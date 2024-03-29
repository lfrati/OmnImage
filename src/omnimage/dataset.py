import mimetypes
import os
from pathlib import Path
import random

from tqdm import trange

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import Normalize

from .download import download_dataset


def get_images(path):
    def _get_files(p, fs, extensions=None):
        res = [
            p / f
            for f in fs
            if not f.startswith(".")
            and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
        ]
        return res

    extensions = set(
        k for k, v in mimetypes.types_map.items() if v.startswith("image/")
    )
    res = []
    for p, _, f in os.walk(path):
        res += _get_files(Path(p), f, extensions)
    return sorted(res)


class OmnImageDataset(Dataset):
    def __init__(
        self, data_dir, samples=20, normalize=True, device="cpu", memoize=False
    ):
        self.data_dir = data_dir
        self.img_dir = f"{data_dir}/OmnImage84_{samples}"
        if not Path(self.img_dir).exists():
            download_dataset(samples=samples, download_root=data_dir)
        else:
            print(f"Found {self.img_dir}. Skipping download.")
        get_class = lambda x: x.parent.name
        self.normalize = normalize
        # means and std computed from the 100 version
        self.transform = Normalize(
            mean=[0.48751324, 0.46667117, 0.41095525],
            std=[0.26073888, 0.2528451, 0.2677635],
        )
        self.images = sorted(get_images(self.img_dir))
        self.classes = [get_class(im) for im in self.images]
        self.uniq_classes = list(sorted(set(self.classes)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.uniq_classes)}
        self.labels = [self.class_to_idx[get_class(im)] for im in self.images]
        self.memoize = memoize
        self.memo = {}
        self.device = device
        # read_images doesn't work with Path
        self.images = [path.as_posix() for path in self.images]
        assert self.device in [
            "cpu",
            "cuda",
        ], f"Device must be either 'cpu' or 'cuda': {device} found."

    def __len__(self):
        return len(self.images)

    def _get(self, idx):
        image = read_image(self.images[idx]) / 255
        label = self.labels[idx]
        image = image.to(self.device)
        label = torch.tensor(label).to(self.device)
        if self.normalize:
            image = self.transform(image)
        return image, label

    def __getitem__(self, idx):
        if self.memoize:
            if idx not in self.memo:
                self.memo[idx] = self._get(idx)
            return self.memo[idx]
        else:
            return self._get(idx)

    def to_memmap(self, path):
        n = len(self)
        c, h, w = self[0][0].shape
        fp = np.memmap(path, dtype="float32", mode="w+", shape=(n, c, h, w))
        print(f"Creating memmap file {path}")
        for i in trange(n):
            img, _ = self[i]
            fp[i] = img.numpy()
        fp.flush()

    def __repr__(self):
        name = self.__class__.__name__
        nimages = len(self.images)
        nclasses = len(self.uniq_classes)
        return f"{name}\n    images: {nimages:_}\n   classes: {nclasses:_}"


def split(dataset, p=0.8, samples=20, seed=None):
    # randomly split the dataset between train/test
    # e.g. samples=3, nclasses=100, p=0.8
    # labels is a list of ints
    rng = random.Random(seed)
    labels = [dataset[i][1] for i in trange(len(dataset))]
    assert is_sorted(labels)
    classes = list(set(labels))  # [0,1,2,...,100]
    ntrain = int(len(classes) * p)  # 100*0.8 = 80
    rng.shuffle(classes)
    train_classes = sorted(classes[:ntrain])  # [0,3,4,...,93] : 80
    test_classes = sorted(classes[ntrain:])  # [1,2,5,...,100] : 20
    train_idxs = [
        (i * samples) + j for i in train_classes for j in range(samples)
    ]  # [0,1,2,9,10,11,...,276]
    test_idxs = [
        (i * samples) + j for i in test_classes for j in range(samples)
    ]  # [3,4,5,6,7,8,5,5,5,...,300]
    return (
        Subset(dataset, train_idxs),
        Subset(dataset, test_idxs),
        train_classes,
        test_classes,
    )


def is_sorted(l):
    return all(i <= j for i, j in zip(l, l[1:]))


class Sampler:
    def __init__(self, dataset, nsamples_per_class=20):
        # assumes indexes are sorted per class 00..0011..1122...
        # assert is_sorted([dataset[i][1] for i in range(len(dataset))])
        self.dataset = dataset
        self.samples = np.arange(len(dataset))
        self.class_idxs = self.samples.reshape(-1, nsamples_per_class)
        self.classes = np.arange(self.class_idxs.shape[0])

    def __repr__(self):
        return f"Sampler: {len(self.dataset)} SAMPLES, {len(self.classes)} CLASSES"

    def get(self, idxs):
        # get batch of ims,labels from a list of indices
        ims = [self.dataset[i][0] for i in idxs]
        lbs = [self.dataset[i][1] for i in idxs]
        # return torch.stack(ims), torch.stack(lbs)
        return torch.stack(ims), torch.tensor(lbs)

    def sample_class(self, N=20):
        # get N ims of a single class, used for inner loop
        # NOTE: always gets the same N ims
        cls = np.random.choice(self.classes)
        return self.get(self.class_idxs[cls][:N])

    def sample_random(self, N=64):
        # get N at random from the whole dataset, used for the outer loop
        samples = np.random.choice(self.samples, size=N, replace=False)
        return self.get(samples)
