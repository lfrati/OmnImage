import mimetypes
import os
from pathlib import Path
import random
from time import monotonic

from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import Normalize
from tqdm import trange


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
    def __init__(self, img_dir, normalize=True, device="cpu", memoize=False):
        get_class = lambda x: x.parent.name
        self.normalize = normalize
        # means and std computed from the 100 version
        self.transform = Normalize(
            mean=[0.48751324, 0.46667117, 0.41095525],
            std=[0.26073888, 0.2528451, 0.2677635],
        )
        self.img_dir = img_dir
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
        name = dataset.__class__.__name__
        nimages = len(self.images)
        nclasses = len(self.uniq_classes)
        return f"{name}\n    images: {nimages:_}\n   classes: {nclasses:_}"


def split(dataset, p=0.8, samples=20, verbose=False, seed=None):
    # randomly split the dataset between train/test
    # e.g. samples=3, nclasses=100, p=0.8
    # labels is a list of ints #[]
    if verbose:
        print("Preparing splits...")
        range_fun = trange
    else:
        range_fun = range
    rng = random.Random(seed)
    labels = [dataset[i][1] for i in range_fun(len(dataset))]
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
    if verbose:
        print(f"Splits ready: Train:{len(train_idxs)} Test:{len(test_idxs)}")
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
        self.balanced = np.copy(self.class_idxs).T
        self.rng = np.random.default_rng()
        self.rng.permuted(self.balanced, axis=0, out=self.balanced)
        self.slice = 0

    def __repr__(self):
        return f"Sampler: {len(self.dataset)} SAMPLES, {len(self.classes)} CLASSES, DEVICE={self.device}"

    def get(self, idxs):
        # get batch of ims,labels from a list of indices
        ims = [self.dataset[i][0] for i in idxs]
        lbs = [self.dataset[i][1] for i in idxs]
        return torch.stack(ims), torch.stack(lbs)

    def sample_class(self, N=20):
        # get N ims of a single class, used for inner loop
        # NOTE: always gets the same N ims
        cls = np.random.choice(self.classes)
        return self.get(self.class_idxs[cls][:N])

    def sample_random(self, N=64):
        # get N at random from the whole dataset, used for the outer loop
        samples = np.random.choice(self.samples, size=N, replace=False)
        return self.get(samples)

    def sample(self, inner_size=20, outer_size=64, expand=True):
        # get 20 ims for the inner and 20+64 ims for the outer
        inner_ims, inner_labels = self.sample_class(inner_size)
        outer_ims, outer_labels = self.sample_random(outer_size)
        outer_ims = torch.cat([inner_ims, outer_ims])
        outer_labels = torch.cat([inner_labels, outer_labels])
        if expand:
            inner_ims = inner_ims.unsqueeze(1)
        return inner_ims, inner_labels, outer_ims, outer_labels

    def sample_balanced(self):
        # get a sample with exactly 1 example from every class
        idxs = self.balanced[self.slice]
        samples = self.get(idxs)
        if self.slice >= len(self.balanced) - 1:
            self.slice = 0
            self.rng.permuted(self.balanced, axis=0, out=self.balanced)
            # NOTE: this will change idxs, use them before shuffling
        else:
            self.slice += 1
        return samples


def async_loader(loader):
    data_iter = iter(loader)
    nbatches = len(loader)

    # start loading the first batch
    next_batch = next(data_iter)
    # with pin_memory=True and non_blocking=True, this will copy data to GPU non blockingly
    next_batch = [_.cuda(non_blocking=True) for _ in next_batch]

    for i in range(nbatches):
        batch = next_batch
        if i + 1 < nbatches:
            # start copying data of next batch
            next_batch = next(data_iter)
            next_batch = [el.cuda(non_blocking=True) for el in next_batch]

        yield batch


#%%


"""
We are going to store the dataset shape in the memorymapped file.
Problem:  Problem the dataset has variable length and I don't want to store the size in the name.
Solution: Use a fixed size header of (N,C,H,W) = 4 x np.uint32 = 16 bytes
          First read the haeder, get shape and then read the rest of the file with 16 offset.
Problem:  The dataset is stored as np.float32 (for compatibility with the network weights).
          The shape is stored as np.uint32 (no way I'm storing integers as floats >_>)
Solution: - Read the uint32 shape to float32 format (don't convert, just read the bytes)
          - Flatten and concatenate header/data
          - Store
          - Retrieve
          - Read the shape from the 16 bytes
          - Re-interpret it as uint32
"""


HEADERBYTES = 16

CHW = "np.dtype[np.float32]"
FourD = tuple[int, int, int, int]
NDArrayNCHW = "np.ndarray[FourD, CHW]"


def store_memmap(path: str, dataset: Dataset) -> None:
    N = len(dataset)
    im, _ = dataset[0]
    im_shape = im.shape
    im_size = np.prod(im_shape).item()
    shape = np.array((N, *im_shape)).astype(np.uint32)
    data_size = np.prod(shape).item()
    # read as float
    float_shape = np.frombuffer(shape.tobytes(), dtype=np.float32)
    assert float_shape.nbytes == HEADERBYTES
    tot_shape = float_shape.size + data_size

    mmfp = np.memmap(path, dtype="float32", mode="w+", shape=tot_shape)
    start = float_shape.size
    mmfp[:start] = float_shape
    for i in trange(N):
        img, _ = dataset[i]
        mmfp[start : start + im_size] = img.numpy().flatten()
        start += im_size
    mmfp.flush()


def retrieve_memmap(path: str) -> NDArrayNCHW:
    retrieved = np.memmap(path, dtype="float32", mode="r+")
    retrieved_float_shape = retrieved[:4]
    int_shape = tuple(np.frombuffer(retrieved_float_shape.tobytes(), dtype=np.uint32))
    retrieved_data = np.memmap(
        path, dtype="float32", offset=16, mode="c", shape=int_shape
    )
    return retrieved_data


dataset = OmnImageDataset("./data/OmnImage84_20")

MMAP_FILE = "./data/mmap_OmnImage_20.dat"
DATA_LOC = "./data/OmnImage_20"

store_memmap(MMAP_FILE, dataset)
retrieved_data = retrieve_memmap(MMAP_FILE)

print("Verifying retrieved data...")
for i in trange(len(dataset)):
    orig, _ = dataset[i]
    retr = retrieved_data[i]
    assert np.allclose(
        orig, retr
    ), f"Element {i} don't match between dataset and retrieved."
print("OK")


#%%


class MemmapDataset(Dataset):
    def __init__(self, path: str):

        # means and stds are computed on the 100 verion
        self.means = torch.tensor([0.4875, 0.4667, 0.4110])
        self.stds = torch.tensor([0.2607, 0.2528, 0.2678])

        retrieved_float_shape = np.memmap(path, dtype="float32", mode="r+")[:4]
        self.shape = tuple(
            np.frombuffer(retrieved_float_shape.tobytes(), dtype=np.uint32)
        )
        self.images = np.memmap(
            path, dtype="float32", offset=16, mode="c", shape=self.shape
        )
        self.path = path
        self.samples_per_class = self.shape[0] // 1_000

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        img = self.images[i]
        label = i // self.samples_per_class
        return torch.from_numpy(img), label

    def to_image(self, ten):
        im = rearrange(ten, "c h w -> h w c")
        im = (im * self.stds) + self.means
        im = im.clip(0, 1)
        return im

    def __repr__(self):
        r = f"{self.__class__.__name__}:\n"
        r += f"\t shape: {self.shape}\n"
        r += f"\t  file: {self.path}\n"
        return r


mmdataset = MemmapDataset(MMAP_FILE)
print(mmdataset)

ten, label = mmdataset[0]
img = mmdataset.to_image(ten)

plt.imshow(img)
plt.show()

#%%

dataset = OmnImageDataset("./data/OmnImage84_20", memoize=True)
memdataset = MemmapDataset(MMAP_FILE)

print("Dataset shape", memdataset.shape)
print("No memoization")
for i in trange(len(dataset)):
    img, label = dataset[i]
    assert img.shape == (3, 84, 84)
print("With memoization")
for i in trange(len(dataset)):
    img, label = dataset[i]
    assert img.shape == (3, 84, 84)
print("Memory mapped")
for i in trange(len(memdataset)):
    img, _ = memdataset[i]
    assert img.shape == (3, 84, 84)


#%%


def test(dataset, shuffle, trials, nworkers):
    print(f"bsize          async    block")
    for bsize in [32, 64, 128, 256, 512]:
        loader = DataLoader(
            dataset,
            batch_size=bsize,
            pin_memory=True,
            shuffle=shuffle,
            num_workers=nworkers,
        )
        noblock_times = []
        for i in range(trials):
            start = monotonic()
            for i, (ims, labels) in enumerate(async_loader(loader)):
                _ = ims.std()
            end = monotonic()
            noblock_times.append(end - start)
        block_times = []
        for i in range(trials):
            start = monotonic()
            for i, (ims, labels) in enumerate(loader):
                ims = ims.cuda()
                _ = ims.std()
            end = monotonic()
            block_times.append(end - start)
        # print(noblock, block)
        noblock = np.mean(noblock_times)
        block = np.mean(block_times)
        better = (noblock - block) / block
        print(f"{bsize:>4} : {better*100:+.2f}% ({noblock:.2f} vs {block:.2f})")


# no memoization, 8 workers
dataset = OmnImageDataset("OmnImage84_100")
test(dataset, shuffle=True, trials=5, nworkers=8)
# bsize          async    block
#   32 : -1.47% (4.99 vs 5.06)
#   64 : +1.55% (4.40 vs 4.33)
#  128 : +0.95% (4.43 vs 4.39)
#  256 : +2.39% (4.40 vs 4.30)
#  512 : -0.65% (4.23 vs 4.26)

# # memoization, 1 workers
dataset = OmnImageDataset("OmnImage84_100", memoize=True)
# preload
for i in trange(len(dataset)):
    v = dataset[i]
test(dataset, shuffle=True, trials=5, nworkers=0)
# bsize          async    block
#   32 : -16.69% (3.09 vs 3.71)
#   64 : -19.91% (2.90 vs 3.62)
#  128 : -14.99% (2.96 vs 3.48)
#  256 : -25.83% (2.79 vs 3.76)
#  512 : -15.62% (2.50 vs 2.96)

# memory mapped, 1 worker
dataset = MemmapDataset(MMAP_FILE)
test(dataset, shuffle=True, trials=5, nworkers=0)
# bsize          async    block
#   32 : -3.79% (2.05 vs 2.13)
#   64 : -32.42% (1.55 vs 2.30)
#  128 : -30.24% (1.81 vs 2.60)
#  256 : -23.68% (2.06 vs 2.70)
#  512 : -21.40% (2.07 vs 2.63)


#%%

# train, test, tr_cls, te_cls = split(dataset, verbose=True, seed=4)
# print(len(train), len(test))

# device = "cuda"
# sampler = OMLSampler(train, device)
# print(sampler)

# # OMLSampler(dataset) to use the whole dataset instead

# # inner/outer sample sizes
# I, O = 15, 32
# inner_ims, inner_labels, outer_ims, outer_labels = sampler.sample(
#     inner_size=I, outer_size=O
# )
# print(inner_ims.shape)
# print(inner_labels.shape)
# print(outer_ims.shape)
# print(outer_labels.shape)
# assert (
#     inner_ims.shape[0] == I
# ), f"Inner loop mismatch: Expected {I} found {inner_ims.shape[0]}"
# assert (
#     outer_ims.shape[0] == I + O
# ), f"Outer loop mismatch: Expected {I+O} found {outer_ims.shape[0]}"

# ims, labels = sampler.sample_balanced()
# assert ims.device.type == device
# assert len(set([label.item() for label in labels])) == len(labels)
