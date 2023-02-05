from pathlib import Path

# import cv2
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.utils import make_grid
from tqdm import tqdm


def get_wnids():
    with open("./wnids.txt", "r") as f:
        lines = f.readlines()
    wnid2name = {}
    for line in lines:
        wnid, _, name = line.strip().split(" ")
        print(wnid, name)
        wnid2name[wnid] = name
    return wnid2name


wnid2name = get_wnids()

print(wnid2name)


def get_omnicliques():
    with open("./OmnImage_20.txt", "r") as f:
        lines = f.readlines()

    from collections import defaultdict

    wnid2clique = defaultdict(list)

    for line in lines:
        path = Path(line)
        wnid = path.parent
        wnid2clique[str(wnid)].append(str(path).strip())

    return dict(wnid2clique)


wnid2clique = get_omnicliques()

#%%

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# import torchvision
# bicubic = torchvision.transforms.InterpolationMode.BICUBIC


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


imagenet_dir = Path("/home/Storage/data_dungeon/image_data/ILSVRC2012/")


def save_grid(img, name, wnid):
    plt.figure(figsize=(10, 8))
    img = img.detach()
    img = F.to_pil_image(img)
    plt.imshow(np.asarray(img))
    # plt.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    txt = plt.text(
        10,
        25,
        name.replace("_", " "),
        fontsize=32,
        color="black",
        bbox={"facecolor": "white", "pad": 4, "alpha": 0.8},
    )
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()
    plt.savefig(f"./cliques/{wnid}-{name}.jpg", dpi=200)
    plt.close()


for wnid, clique in tqdm(wnid2clique.items()):
    # print(wnid)
    ims = []
    for path in clique:
        im_path = str(imagenet_dir / path)
        # im = cv2.resize(
        #     cv2.cvtColor(
        #         cv2.imread(im_path),
        #         cv2.COLOR_BGR2RGB,
        #     ),
        #     (84, 84),
        #     interpolation=cv2.INTER_AREA,
        # )
        # ims.append(torch.from_numpy(im.transpose(2, 0, 1) / 255))
        im = read_image(im_path)
        if im.shape[0] == 1:
            im = im.repeat(3, 1, 1)
        im84 = F.resize(im, (84, 84))
        ims.append(im84)

    name = wnid2name[wnid]
    grid = make_grid(ims, nrow=5)
    save_grid(grid, name, wnid)
    # break
