import numpy as np
from pathlib import Path
import torch
from torchvision.models import vgg19_bn
from subpair import extract
import matplotlib.pyplot as plt
import cv2
from einops import rearrange


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def pairwise_cosine_numpy(X):
    norms = np.einsum("ij,ij->i", X, X)
    np.sqrt(norms, norms)
    X /= norms[:, np.newaxis]
    dists = 1 - np.dot(X, X.T)
    return dists


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

    return (forward, handle)


def load_resize(im, size):
    # ugly AF but fast
    # NOTE: cv2 reads images as BGR!
    return cv2.resize(
        cv2.cvtColor(
            cv2.imread(im.as_posix()),
            cv2.COLOR_BGR2RGB,
        ),
        (size, size),
        interpolation=cv2.INTER_AREA,
    )


def paths2tensors(paths, size):
    tens = torch.stack(
        [
            torch.from_numpy(load_resize(im, size=size).transpose(2, 0, 1)) / 255
            for im in paths
        ]
    )
    return tens


#%%

sample = Path("./lab/sample/n01440764")
paths = np.array(list(sample.iterdir()))  # (1300,)

net = vgg19_bn(pretrained=True, progress=True)
to_hook = net.classifier[3]
forward, _ = hook_it_up(net, to_hook)

inps = paths2tensors(paths, 84)  # (1300, 3, 84, 84)

feats = forward(inps).detach().cpu().numpy()  # (1300, 4096)
centroid = feats.mean(axis=0)  # (4096,)
distances = pairwise_cosine_numpy(feats)  # (1300, 1300)

best, stats = extract(distances, P=200, S=20, K=50, M=3, O=2, its=10_000)

# plt.plot(stats["fits"]
# plt.grid()
# plt.ylabel("Subset distances")
# plt.xlabel("Iterations")
# plt.title("Minimal subset extraction (lower is better)")
# plt.tight_layout()
# plt.savefig("min_subset.png", dpi=300)
# plt.show()

best_ims = np.stack([load_resize(path, 84) for path in paths[best]])
best_grid = rearrange(best_ims, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=5)

l1_to_centroid = np.abs(feats - centroid).sum(axis=1)
closest_to_centroid = np.argsort(l1_to_centroid)[:20]
# closest_to_centroid = np.argsort(l1_to_centroid)[::-1][:20]
print(l1_to_centroid[closest_to_centroid])
closest_ims = np.stack([load_resize(path, 84) for path in paths[closest_to_centroid]])
closest_grid = rearrange(closest_ims, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=5)


fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
axs[0].imshow(best_grid)
axs[0].set_title("Evolved clique")
axs[0].axis("off")
axs[1].imshow(closest_grid)
axs[1].set_title("Closest to centroid")
axs[1].axis("off")
plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.)
plt.show()
