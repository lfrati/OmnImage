from omnimage import MemmapDataset, Sampler, OmnImageDataset, store_memmap
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

# from time import time
# from numba import cuda, njit, prange
# from utils import flatten

####################################
######## COMPUTE DISTANCES #########
####################################


def pairwise_cosine(X):
    norms = np.einsum("ij,ij->i", X, X)
    np.sqrt(norms, norms)
    X /= norms[:, np.newaxis]
    dists = 1 - np.dot(X, X.T)
    return dists


####################################
######## TEST PERFORMANCE ##########
####################################


def flatten(l):
    return [i for seq in l for i in seq]


def sample_distances(
    distances,
    ntasks,
    train_samples,  # ,=15,
    class_size,  # =20,
    tot_classes,  # =659,
):
    # distances : (13180, 13180) where 13180 = 659 tasks * 20 examples
    # array of task numbers
    tasks = sorted(np.random.choice(np.arange(tot_classes), ntasks, replace=False))
    # array of arrays of indices for each task
    task_ids = [np.arange(class_size * task, class_size * (task + 1)) for task in tasks]
    # array of arrays of indices to use for training
    train_tasks = np.array(
        [
            sorted(np.random.choice(ids, train_samples, replace=False))
            for ids in task_ids
        ]
    )
    # array of arrays of the indices not used for training
    test = np.array(
        [i for train, ids in zip(train_tasks, task_ids) for i in ids if i not in train]
    )
    # turn into flat array to index into the distances matrix
    train = flatten(train_tasks)
    # extract sub-matrix s.t. rows = test tasks, columns = distance to train tasks
    return distances[np.ix_(test, train)], train, test


def test_distances(distances, ntasks, train_samples, class_size, tot_classes):
    test_samples = class_size - train_samples
    # supp has shape: (class_size * ntasks, train_samples * ntasks)
    # e.g. train_samples = 15, class_size = 1000, ntasks = 10 -> (9850, 150)
    supp, train, test = sample_distances(
        distances,
        ntasks=ntasks,
        train_samples=train_samples,
        class_size=class_size,
        tot_classes=tot_classes,
    )
    # We have a matrix of distances where each row is a "test image" and each column its
    # cosine distance from a "train image". The first test_Sampels rows contains test
    # images corresponding to the first train task so it's "label" is 0, and so on.
    labels = [task for task in range(ntasks) for i in range(class_size - train_samples)]
    # If the model was perfect when sorting the distance and taking the top train_samples
    # ones they should all belong to the correct class.
    # nneighbors contains the closest train_examples samples for each test image
    # NOTE: divide by train_samples to turn indexes into class labels
    nneighbors = (np.argsort(supp, axis=1) // train_samples)[:, :train_samples]
    # bincount + argmax = fun. A sane person would have used Counter + some trickery.
    # NOTE: bincount is a sweet poison but wastefull AF np.bincount([10000000])
    preds = [np.argmax(np.bincount(row)) for row in nneighbors]
    accuracy = sum([pred == label for pred, label in zip(preds, labels)]) / len(labels)
    return accuracy


#%%

import torch

#%%

NSAMPLES = 20
DOWNLOAD_LOC = "./data"
MMAP_FILE = f"{DOWNLOAD_LOC}/mmap_OmnImage84_{NSAMPLES}.dat"
dataset = OmnImageDataset(data_dir=DOWNLOAD_LOC, samples=NSAMPLES)
store_memmap(MMAP_FILE, dataset)
mmdataset = MemmapDataset(folder=DOWNLOAD_LOC, samples=NSAMPLES)
sampler = Sampler(mmdataset, nsamples_per_class=NSAMPLES)
print(sampler)

# for _ in range(10):
#     N = np.random.randint(1, NSAMPLES)
#     ims, labels = sampler.sample_class(N=N)
#     assert ims.shape == (N, 3, 84, 84)
#     assert labels.shape == (N,)
#     unique_labels = set([l.item() for l in labels])
#     assert len(unique_labels) == 1

#%%

import matplotlib.pyplot as plt
import torchvision


class DeNormalize(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denorm = DeNormalize(mean=dataset.transform.mean, std=dataset.transform.std)

ims, labels = sampler.sample_class(N=NSAMPLES)

# plt.imshow(denorm(ims[0]).numpy().transpose(1, 2, 0))
# plt.show()

#%%

from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Lambda

transf = Compose(
    [
        ToTensor(),
        Resize(84),
        Normalize(mean=0.13066, std=0.30131),
        Lambda(lambda x: x.expand(3, -1, -1)),
    ]
)
omniglot = torchvision.datasets.Omniglot(root=DOWNLOAD_LOC, transform=transf)


#%%

import torch.nn as nn


class Conv4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_args = {
            "kernel_size": (3, 3),
            "stride": (1, 1),
            # "padding": 1,
            # "padding_mode": "circular",
        }

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, **self.conv_args),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=256, out_channels=256, **self.conv_args),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=256, out_channels=256, **self.conv_args),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=256, out_channels=256, **self.conv_args),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
        )
        self.fc = nn.Linear(in_features=2304, out_features=1000, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


# net = Conv4().cuda()
net = torch.jit.script(Conv4().cuda())
net.eval()

out = net(ims.cuda())
print(out.shape)

#%%


def extract_features(dataset, model, device="cuda"):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        features = [
            model.encoder(ims.to(device)).cpu().numpy() for ims, _ in tqdm(loader)
        ]
    return np.vstack(features)


features = extract_features(dataset, net)

distances = pairwise_cosine(features)

#%%


test_distances(
    distances,
    ntasks=100,
    train_samples=NSAMPLES - 5,
    class_size=NSAMPLES,
    tot_classes=1000,
)

#%%

features = extract_features(omniglot, net)
distances = pairwise_cosine(features)

test_distances(
    distances,
    ntasks=600,
    train_samples=NSAMPLES - 5,
    class_size=NSAMPLES,
    tot_classes=963,
)

#%%

"""
ToDo: iid train on omniglot/omnimage and compare knn performance
"""
