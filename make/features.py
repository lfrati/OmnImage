import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.models import vgg19_bn
from tqdm import tqdm

from utils import paths2tensors_par, read_folder


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
    forward, _ = hook_it_up(net, to_hook)
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ims", type=str, default="imagenet", help="path to folder w/ imagenet data"
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
    name = "output/feats"
    classes = read_folder(args.ims)
    store_activations(net, name, classes, device=args.device, to_hook=to_hook)


if __name__ == "__main__":
    main()
