import mimetypes
import os
from hashlib import blake2b
from pathlib import Path
from typing import List, Optional

import cv2
from numba import i8, u8
from PIL import Image

import torch


def model2name(net):
    return f"{net.__module__}_{weights2hash(net)}"


def par2bytes(p):
    return p.detach().cpu().numpy().tobytes()


def weights2hash(model, dsize=8):
    # compute hash of a torch.nn.Module weights or a list of tensors

    h = blake2b(digest_size=dsize)
    # state = {name:par2bytes(p) for name, p in net.named_parameters()}
    # names = sorted(state.keys()) # sort names for reproducibility
    # for name in names:
    #   b = state[name]
    #   h.update(b)
    if issubclass(model.__class__, torch.nn.Module):
        model = model.parameters()
    for p in model:
        h.update(par2bytes(p))
    return h.hexdigest()


def read_folder(folder: Path) -> List[Path]:
    if type(folder) is str:
        folder = Path(folder)
    return sorted(folder.iterdir())


def is_rgb(path):
    return Image.open(path).mode == "RGB"


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
    for p, d, f in os.walk(path):
        res += _get_files(Path(p), f, extensions)
    return sorted(res)


def load(im: Path) -> u8[:, :, :]:
    return cv2.cvtColor(
        cv2.imread(im.as_posix()),
        cv2.COLOR_BGR2RGB,
    )


def load_resize(im: Path, size: i8 = 64) -> u8[:, :]:
    # ugly AF but fast
    # NOTE: cv2 reads images as BGR!
    return cv2.resize(
        cv2.imread(im.as_posix()),
        (size, size),
        interpolation=cv2.INTER_AREA,
    )


def paths2tensors(paths: List[Path], size: Optional[i8] = None):
    # ugly AF but 3 times faster than Image.open
    # toten = torchvision.transforms.ToTensor()
    # pil = torch.stack([toten(Image.open(im)) for im in tqdm(paths)])
    # NOTE: need to transpose chan. dim. to front
    if size is None:
        tens = torch.stack(
            [torch.from_numpy(load(im).transpose(2, 0, 1)) / 255 for im in paths]
        )
    else:
        tens = torch.stack(
            [
                torch.from_numpy(load_resize(im, size=size).transpose(2, 0, 1)) / 255
                for im in paths
            ]
        )
    return tens
