from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vgg19_bn
from tqdm import tqdm

from utils import paths2tensors, read_folder, model2name, paths2tensors_par


def hook_it_up(net, layer):
    activations = None

    def my_hook(m, i, o):
        nonlocal activations
        activations = o

    handle = layer.register_forward_hook(my_hook)
    net.eval()

    def forward(inp):
        with torch.no_grad():
            _ = net(inp)
        return activations

    return forward, handle


def store_activations(net, name, classes, device, to_hook=None):
    folder = Path(name)
    folder.mkdir(parents=True, exist_ok=True)
    net = net.to(device)
    forward, handle = hook_it_up(net, to_hook)
    for cls in tqdm(classes):
        fname = folder / (cls.name + ".npy")
        if not fname.exists():
            ims = read_folder(cls)
            inps = paths2tensors_par(ims, size=84).to(device)
            feats = forward(inps).detach().cpu().numpy()
            np.save(fname, feats)
        else:
            print(fname, "exists")


#%%


if __name__ == "__main__":

    from torchvision.models import vgg19_bn
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir", type=str, default="imagenet", help="path to folder w/ imagenet data"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run the model on",
        choices=["cpu", "mps", "cuda"],
    )
    args = parser.parse_args()

    net = vgg19_bn(pretrained=True, progress=True)
    to_hook = net.classifier[3]
    name = model2name(net)  # torchvision.models.vgg_ce631fc9ca0278a2
    classes = read_folder(args.dir)
    store_activations(net, name, classes, device=args.device, to_hook=to_hook)
