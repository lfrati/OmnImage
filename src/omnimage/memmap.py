from pathlib import Path

from tqdm import trange

from einops import rearrange
import numpy as np
import torch
from torch.utils.data import Dataset

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


def retrieve_memmap(path: str): # -> NDArrayNCHW:
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
        im = rearrange(ten, "c h w -> h w c")
        im = (im * self.stds) + self.means
        im = im.clip(0, 1)
        return im

    def __repr__(self):
        r = f"{self.__class__.__name__}:\n"
        r += f"\t shape: {self.shape}\n"
        r += f"\t  file: {self.path}\n"
        return r


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
