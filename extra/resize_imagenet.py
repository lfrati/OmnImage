import re
from pathlib import Path

import cv2
from tqdm import tqdm

from utils import load_resize

from pathlib import Path


#%%


imgnet_folder = Path("/home/Storage/data_dungeon/image_data/ILSVRC2012/")
small_folder = Path("/home/Storage/data_dungeon/image_data/ImageNet84/")

classes = list(imgnet_folder.iterdir())

for cls in tqdm(classes):
    new_cls_folder = small_folder / cls.name
    # print(new_cls_folder)
    new_cls_folder.mkdir(parents=True, exist_ok=True)
    for im in cls.iterdir():
        new_path = new_cls_folder / im.name
        # print("from:", im)
        # print("to:  ", new_path)
        small_im = load_resize(im, size=84)
        cv2.imwrite(new_path.as_posix(), small_im)
