from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor
from tqdm import tqdm, trange

from omnimage import MemmapDataset, OmnImageDataset, Sampler, split, store_memmap

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


class Memo(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.memo = {}

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, i):
        if i not in self.memo:
            self.memo[i] = self.dataset.__getitem__(i)
        return self.memo[i]


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


#%%


def extract_features(dataset, model, device="cuda"):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        features = [
            model.encoder(ims.to(device)).cpu().numpy() for ims, _ in tqdm(loader)
        ]
    return np.vstack(features)


def knn_test(dataset, model, device="cuda"):
    features = extract_features(dataset, model, device)
    distances = pairwise_cosine(features)

    name = dataset.__class__.__name__
    if name == "Memo":
        name = dataset.dataset.__class__.__name__

    print(name, ":")
    results = []
    for i in [50, 100, 200, 300]:
        acc = test_distances(
            distances,
            ntasks=i,
            train_samples=NSAMPLES - 5,
            class_size=NSAMPLES,
            tot_classes=300,
        )
        results.append((i, acc))

    return results


#%%

transf = Compose(
    [
        ToTensor(),
        Resize(84),
        Normalize(mean=0.13066, std=0.30131),
        Lambda(lambda x: x.expand(3, -1, -1)),
    ]
)
omniglot_train = Memo(
    torchvision.datasets.Omniglot(root=DOWNLOAD_LOC, transform=transf, background=False)
)
omniglot_test = Memo(
    torchvision.datasets.Omniglot(root=DOWNLOAD_LOC, transform=transf, background=True)
)

"""
ToDo: iid train on omniglot/omnimage and compare knn performance
"""

## this splits each class into train/test, I need to splits classes themselves
# train, test, train_classes, test_classes = split(dataset, p=0.8, samples=20, seed=4)


def meta_split(dataset, uniq_classes):
    classes = list(range(len(uniq_classes)))
    shuffle(classes)
    train_classes = set(classes[:700])
    test_classes = set(classes[700:])

    train_idxs = []
    test_idxs = []
    for i, (im, cls) in tqdm(enumerate(dataset)):
        if cls.item() in train_classes:
            train_idxs.append(i)
        else:
            test_idxs.append(i)

    train_idxs = sorted(train_idxs)
    test_idxs = sorted(test_idxs)

    return Subset(dataset, train_idxs), Subset(dataset, test_idxs)


omnimage_train, omnimage_test = meta_split(dataset, dataset.uniq_classes)
omnimage_train = Memo(omnimage_train)

#%%

from torch.nn.functional import cross_entropy
from torch.optim import Adam


def train(model, dataset, its, bsize=64, lr=1e-4, device="cuda"):
    loader = DataLoader(dataset, batch_size=bsize, shuffle=True)

    net = torch.jit.script(model.to(device))
    net.train()

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    pbar = trange(its)

    for it in pbar:
        avg_loss = 0
        for i, (ims, targets) in enumerate(loader):
            optim.zero_grad()
            out = net(ims.to(device))
            loss = cross_entropy(out, targets.to(device))
            avg_loss += loss.item()
            pbar.set_description(f"{avg_loss/(i+1):.5f}")
            loss.backward()
            optim.step()
    return net


pretrained_omniglot = train(Conv4(), omniglot_train, 10)
torch.jit.save(pretrained_omniglot, "pretrained_omniglot.pt")

pretrained_omnimage = train(Conv4(), omnimage_train, 15)
torch.jit.save(pretrained_omnimage, "pretrained_omnimage.pt")

#%%


# denorm = DeNormalize(mean=dataset.transform.mean, std=dataset.transform.std)
# ims, labels = sampler.sample_class(N=NSAMPLES)
# plt.imshow(denorm(ims[0]).numpy().transpose(1, 2, 0))
# plt.show()


NSAMPLES = 20
DOWNLOAD_LOC = "./data"
MMAP_FILE = f"{DOWNLOAD_LOC}/mmap_OmnImage84_{NSAMPLES}.dat"
dataset = OmnImageDataset(data_dir=DOWNLOAD_LOC, samples=NSAMPLES)
store_memmap(MMAP_FILE, dataset)
mmdataset = MemmapDataset(folder=DOWNLOAD_LOC, samples=NSAMPLES)
# sampler = Sampler(mmdataset, nsamples_per_class=NSAMPLES)
# print(sampler)

# for _ in range(10):
#     N = np.random.randint(1, NSAMPLES)
#     ims, labels = sampler.sample_class(N=N)
#     assert ims.shape == (N, 3, 84, 84)
#     assert labels.shape == (N,)
#     unique_labels = set([l.item() for l in labels])
#     assert len(unique_labels) == 1

# net = Conv4().cuda()
net = torch.jit.script(Conv4().cuda())
net.eval()

knn_test(omnimage_test, torch.jit.load("pretrained_omnimage.pt"))

knn_test(omniglot_test, torch.jit.load("pretrained_omniglot.pt"))

# knn_test(dataset, net)
# knn_test(mmdataset, net)
# knn_test(omniglot, net)
