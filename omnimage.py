import hashlib
import mimetypes
import os
from pathlib import Path
import zipfile
import random

import numpy as np
import requests
import torch
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import Normalize
from tqdm import trange

__all__ = ["OmnImageDataset"]

FILES = {
    "url": "https://uvm.edu/~lfrati",
    20: {
        "filename": "OmnImage84_20.zip",
        "md5": "30aa0b55fc6b3bccd06aaa6615661ee8",
    },
    100: {
        "filename": "OmnImage84_100.zip",
        "md5": "3869650152622568a7356146307c414e",
    },
    "sample": {
        "filename": "ImagenetSample.zip",
        "md5": "971eddceacb7e929cfbe55d041e9f794",
    },
}


def download_url(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully!")
    else:
        print("Failed to download file. Status code:", response.status_code)


def calculate_md5(filename: str, expected: str) -> bool:
    with open(filename, "rb") as file:
        md5_hash = hashlib.md5(file.read())
        digest = md5_hash.hexdigest()
        if digest == expected:
            return True
        print(f"md5sums don't match! Expected {expected}, got {digest}")
        return False


def extract_zip(from_path: str, to_path: str) -> None:
    with zipfile.ZipFile(
        from_path,
        "r",
        compression=zipfile.ZIP_STORED,
    ) as zip:
        zip.extractall(to_path)


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


def download_dataset(
    num_samples: int,
    data_dir: str,
) -> None:
    info = FILES[num_samples]
    zip_name = info["filename"]
    folder_name = zip_name.replace(".zip", "")
    url = f"{FILES['url']}/{zip_name}"
    md5sum = info["md5"]

    data_folder = Path(data_dir).resolve()
    assert data_folder.exists()
    zip_path = data_folder / zip_name
    folder_path = data_folder / folder_name

    print("Saving to:", data_folder)
    print("Filename:", zip_name)

    if zip_path.exists():
        print("Zip exists already.")
    else:
        print(f"Downloading:\n  {url}\nto\n  {zip_path}")
        download_url(url, zip_path)
        print("Downloaded.\n")

    assert calculate_md5(zip_path, md5sum)

    if folder_path.exists():
        print("Folder exists already.")
    else:
        print(f"Extracting:\n  {zip_path}\nto\n  {data_folder}")
        extract_zip(zip_path, data_folder.as_posix())
        print("Extracted.")


class OmnImageDataset(Dataset):
    def __init__(
        self, data_dir, samples=20, normalize=True, device="cpu", memoize=False
    ):
        self.data_dir = data_dir
        self.img_dir = f"{data_dir}/OmnImage84_{samples}"
        if not Path(self.img_dir).exists():
            download_dataset(num_samples=samples, data_dir=data_dir)
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
    def __init__(self, dataset: OmnImageDataset, nsamples_per_class: int = 20):
        # assumes indexes are sorted per class 00..0011..1122...
        # assert is_sorted([dataset[i][1] for i in range(len(dataset))])
        self.dataset = dataset
        self.len = len(dataset)
        self.samples = np.arange(self.len)
        self.class_idxs = self.samples.reshape(-1, nsamples_per_class)
        self.classes = np.arange(self.class_idxs.shape[0])

    def __repr__(self):
        return f"Sampler: {self.len} SAMPLES, {len(self.classes)} CLASSES"

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

# CHW = "np.dtype[np.float32]"
# FourD = tuple[int, int, int, int]
# NDArrayNCHW = "np.ndarray[FourD, CHW]"


def store_memmap(path: str, dataset: Dataset) -> None:
    if Path(path).exists():
        print(f"Found {path}. Skipping store_memmap.")
        return
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


def retrieve_memmap(path: str):  # -> NDArrayNCHW:
    retrieved = np.memmap(path, dtype="float32", mode="r+")
    retrieved_float_shape = retrieved[:4]
    int_shape = tuple(np.frombuffer(retrieved_float_shape.tobytes(), dtype=np.uint32))
    retrieved_data = np.memmap(
        path, dtype="float32", offset=16, mode="c", shape=int_shape
    )
    return retrieved_data


class MemmapDataset(Dataset):
    def __init__(self, folder: str, samples=20):
        # means and stds are computed on the 100 verion
        self.means = torch.tensor([0.4875, 0.4667, 0.4110])
        self.stds = torch.tensor([0.2607, 0.2528, 0.2678])
        path = f"{folder}/mmap_OmnImage84_{samples}.dat"

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
        # im = rearrange(ten, "c h w -> h w c")
        im = ten.permute(1, 2, 0)
        im = (im * self.stds) + self.means
        im = im.clip(0, 1)
        return im

    def __repr__(self):
        r = f"{self.__class__.__name__}:\n"
        r += f"\t shape: {self.shape}\n"
        r += f"\t  file: {self.path}\n"
        return r
