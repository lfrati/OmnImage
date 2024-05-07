import requests
import zipfile
import hashlib
import os
import mimetypes
from pathlib import Path
import numpy as np
import torch
from tqdm import trange
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Normalize

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
        print(f"Extracting:\n  {zip_path}\nto\n  {folder_path}")
        extract_zip(zip_path, folder_path)
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
