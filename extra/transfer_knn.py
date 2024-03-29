import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor
from tqdm import tqdm, trange

from omnimage import MemmapDataset, OmnImageDataset, Sampler, split, store_memmap

NSAMPLES = 20
DOWNLOAD_LOC = "./data"
MMAP_FILE = f"{DOWNLOAD_LOC}/mmap_OmnImage84_{NSAMPLES}.dat"

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


def extract_features(dataset, model, device="cuda"):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        features = [
            model.encoder(ims.to(device)).cpu().numpy() for ims, _ in tqdm(loader)
        ]
    return np.vstack(features)


def knn_test(dataset, model, device="cuda"):
    model = model.to(device)
    features = extract_features(dataset, model, device)
    distances = pairwise_cosine(features)

    name = dataset.__class__.__name__
    if name == "Memo":
        name = dataset.dataset.__class__.__name__

    # print(name, ":")
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

import random


def meta_split(dataset, uniq_classes, seed=4):
    random.seed(seed)
    classes = list(range(len(uniq_classes)))
    random.shuffle(classes)
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


dataset = OmnImageDataset(data_dir=DOWNLOAD_LOC, samples=NSAMPLES)
store_memmap(MMAP_FILE, dataset)
mmdataset = MemmapDataset(folder=DOWNLOAD_LOC, samples=NSAMPLES)
# sampler = Sampler(mmdataset, nsamples_per_class=NSAMPLES)
# print(sampler)

omnimage_train, omnimage_test = meta_split(dataset, dataset.uniq_classes)
omnimage_train = Memo(omnimage_train)

#%%


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


# for _ in range(10):
#     N = np.random.randint(1, NSAMPLES)
#     ims, labels = sampler.sample_class(N=N)
#     assert ims.shape == (N, 3, 84, 84)
#     assert labels.shape == (N,)
#     unique_labels = set([l.item() for l in labels])
#     assert len(unique_labels) == 1

# net = torch.jit.script(Conv4().cuda())
# net.eval()

data = {
    "image_untrained": knn_test(omnimage_test, Conv4()),
    "image_pretrained": knn_test(
        omnimage_test, torch.jit.load("pretrained_omnimage.pt")
    ),
    "glot_untrained": knn_test(omniglot_test, Conv4()),
    "glot_pretrained": knn_test(
        omniglot_test, torch.jit.load("pretrained_omniglot.pt")
    ),
}
print(data)

#%%

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image


class Imagenet84Dataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        assert len(labels) == len(images), f"{len(labels)=} != {len(images)=}"
        self.labels = labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = read_image(img_path) / 255.0
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


imagenet84_folder = Path("/home/Storage/data_dungeon/image_data/ImageNet84/")


def sample_class(cls_idxs: list[int], cls_names: list[str], loc: Path, nsamples: int):
    samples = []
    labels = []
    for cls_idx in tqdm(cls_idxs):
        cls_name = cls_names[cls_idx]
        cls_folder = loc / cls_name
        # sort to make the shuffle deterministic
        ims = sorted(cls_folder.iterdir())
        assert nsamples < len(ims)
        random.shuffle(ims)
        ims = [im.as_posix() for im in ims[:nsamples]]
        samples.extend(ims)
        labels.extend([cls_idx for _ in range(nsamples)])
    return samples, labels


def meta_sample_imagenet(loc: Path, nsamples=20, seed=4):

    cls_names = sorted([cls.name for cls in loc.iterdir()])
    random.seed(seed)
    cls_idxs = list(range(len(cls_names)))
    random.shuffle(cls_idxs)

    train_idxs = set(cls_idxs[:700])
    test_idxs = set(cls_idxs[700:])

    train_samples, train_labels = sample_class(train_idxs, cls_names, loc, nsamples)
    test_samples, test_labels = sample_class(test_idxs, cls_names, loc, nsamples)

    transf = Normalize(
        mean=[0.48261318, 0.45887598, 0.4111194],
        std=[0.2710958, 0.26436415, 0.27893084],
    )

    train = Imagenet84Dataset(train_samples, train_labels, transform=transf)
    test = Imagenet84Dataset(test_samples, test_labels, transform=transf)

    return train, test


train84, test84 = meta_sample_imagenet(imagenet84_folder)


# def means_stds():
#     ims = np.array([train84[i][0].numpy() for i in range(len(train84))])
#     print(np.mean(ims, axis=(0, 2, 3)))
#     print(np.std(ims, axis=(0, 2, 3)))
# means_stds()

#%%

# random_pretrained_omnimage = train(Conv4(), train84, 20)
# torch.jit.save(random_pretrained_omnimage, "random_pretrained_omnimage.pt")

NREPS = 20

data = {
    "image_untrained": [knn_test(omnimage_test, Conv4()) for _ in range(NREPS)],
    "image_pretrained": [
        knn_test(omnimage_test, torch.jit.load("pretrained_omnimage.pt"))
        for _ in range(NREPS)
    ],
    "glot_untrained": [knn_test(omniglot_test, Conv4()) for _ in range(NREPS)],
    "glot_pretrained": [
        knn_test(omniglot_test, torch.jit.load("pretrained_omniglot.pt"))
        for _ in range(NREPS)
    ],
    "image_random_untrained": [knn_test(test84, Conv4()) for _ in range(NREPS)],
    "image_random_pretrained": [
        knn_test(test84, torch.jit.load("random_pretrained_omnimage.pt"))
        for _ in range(NREPS)
    ],
}


import json

with open("transfer.json", "w") as f:
    f.write(json.dumps(data))

#%%

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def unzip(l):
    return list(zip(*l))


def CI(xs, conf=0.95):
    from scipy.stats import t

    # The king of speed, is biased estimator so bad?
    # xs : [N, samples]
    m = xs.mean(axis=1)
    s = xs.std(axis=1)
    S = xs.shape[1]  # sample size
    dof = S - 1
    t_crit = np.abs(t.ppf((1 - conf) / 2, dof))
    r = s * t_crit / np.sqrt(S)
    return (m - r, m + r)


def process(repeats, marker, label, color, ax, ci=True):
    ys = np.array([unzip(res)[1] for res in repeats])
    xs = unzip(repeats[0])[0]
    mean = np.mean(ys, axis=0)
    ax.plot(xs, mean, marker, label=label, color=color)
    if ci:
        upper, lower = CI(ys.T)
        ax.fill_between(xs, upper, lower, alpha=0.2, color=color)


fig, ax = plt.subplots(figsize=(8, 8))
process(
    data["glot_pretrained"], marker=".-", label="OmniGlot pretrained", color="C0", ax=ax
)
process(data["glot_untrained"], marker=".--", label="OmniGlot", color="C0", ax=ax)

process(
    data["image_pretrained"],
    marker=".-",
    label="OmnImage pretrained",
    color="C1",
    ax=ax,
)
process(data["image_untrained"], marker=".--", label="OmnImage", color="C1", ax=ax)

process(
    data["image_random_pretrained"],
    marker=".-",
    label="rand-ImageNet pretrained",
    color="C2",
    ax=ax,
)
process(
    data["image_random_untrained"],
    marker=".--",
    label="rand-ImageNet",
    color="C2",
    ax=ax,
)
plt.grid()
plt.legend()
plt.xlabel("# Tasks", fontsize=14)
plt.ylabel("KNN Accuracy", fontsize=14)
plt.ylim(0, 1)
plt.title("KNN learnability/transfer in the 20 images regime")
plt.tight_layout()
plt.savefig("knn_transfer.png")
plt.close()


#%%


cross_data = {
    "image_pretrained_glot": [
        knn_test(omnimage_test, torch.jit.load("pretrained_omniglot.pt"))
        for _ in range(NREPS)
    ],
    "glot_pretrained_image": [
        knn_test(omniglot_test, torch.jit.load("pretrained_omnimage.pt"))
        for _ in range(NREPS)
    ],
    "image_pretrained_random": [
        knn_test(omnimage_test, torch.jit.load("random_pretrained_omnimage.pt"))
        for _ in range(NREPS)
    ],
    "random_pretrained_image": [
        knn_test(test84, torch.jit.load("pretrained_omnimage.pt")) for _ in range(NREPS)
    ],
}


# cross_data["image_pretrained_random"] = [
#     knn_test(omnimage_test, torch.jit.load("random_pretrained_omnimage.pt"))
#     for _ in range(NREPS)
# ]

with open("cross_transfer.json", "w") as f:
    f.write(json.dumps(cross_data))

#%%


fig, ax = plt.subplots(figsize=(8, 8))
process(
    data["glot_pretrained"],
    marker=".-",
    label="OmniGlot pretrained",
    color="C0",
    ax=ax,
)
process(data["glot_untrained"], marker=".--", label="", color="gray", ax=ax, ci=False)
process(
    cross_data["glot_pretrained_image"],
    marker=".--",
    label="OmnImage train$\\rightarrow$ OmniGlot test",
    color="C0",
    ax=ax,
)
process(
    data["image_pretrained"],
    marker=".-",
    label="OmnImage pretrained",
    color="C1",
    ax=ax,
)
process(
    cross_data["image_pretrained_glot"],
    marker=".--",
    label="OmniGlot train $\\rightarrow$ OmnImage test",
    color="C1",
    ax=ax,
)
process(
    cross_data["image_pretrained_random"],
    marker=".--",
    label="rand-ImageNet train $\\rightarrow$ OmnImage test",
    color="C2",
    ax=ax,
)
process(data["image_untrained"], marker=".--", label="", color="gray", ax=ax, ci=False)
plt.grid()
plt.legend()
plt.xlabel("# Tasks", fontsize=14)
plt.ylabel("KNN Accuracy", fontsize=14)
plt.ylim(0, 1)
plt.title("KNN o.o.d. transfer in the 20 images regime")
plt.tight_layout()
plt.savefig("cross_knn_transfer.png")
plt.close()

#%%

fig, ax = plt.subplots(figsize=(8, 8))
process(
    cross_data["random_pretrained_image"],
    marker=".--",
    label="OmnImage train $\\rightarrow$ rand-ImageNet test",
    color="C1",
    ax=ax,
)
process(
    data["image_random_pretrained"],
    marker=".-",
    label="rand-ImageNet pretrained",
    color="C2",
    ax=ax,
)
process(
    data["image_random_untrained"],
    marker=".--",
    label="rand-ImageNet",
    color="gray",
    ci=False,
    ax=ax,
)
plt.grid()
plt.legend()
plt.xlabel("# Tasks", fontsize=14)
plt.ylabel("KNN Accuracy", fontsize=14)
# plt.ylim(0, 0.4)
plt.title("KNN o.o.d. transfer in the 20 images regime")
plt.tight_layout()
plt.savefig("cross_knn_transfer_images.png")
plt.close()
