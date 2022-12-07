import re
from pathlib import Path

import cv2
from tqdm import tqdm

from utils import load_resize


def generate(im_path, output_folder, imagenet_dir="imagenet", im_size=84):
    """
    Read im_path from original imagenet, resize and move to new folder

    Parameters
    ----------

    im_path : list[Path]
        images to read

    imagenet_dir : Path
        location of original imagenet images (referenced by im_path)

    output_folder: Path
        output folder to store the resized images to (e.g. OmnImage64)

    im_size: int
        the desired size of the output images

    Returns
    -------
    None

    """
    cls, file = im_path.parts[-2:]
    new_path = Path(output_folder) / cls / file
    new_path.parent.mkdir(parents=True, exist_ok=True)

    src = (imagenet_dir / im_path).resolve()
    dest = str(new_path.resolve())

    small_im = load_resize(src, size=im_size)
    cv2.imwrite(dest, small_im)


#%%

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ims",
        type=str,
        help="list of paths in the subset",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="OmnImage_20.txt",
        help="list of paths in the subset",
    )
    parser.add_argument(
        "--imsize",
        type=int,
        default=84,
        help="size to resize image to",
    )
    args = parser.parse_args()
    print(args)

    filename = Path(args.subset).name
    pattern = r"[./]?OmnImage_(\d+).txt"
    match = re.match(pattern, filename)
    if match is None:
        raise RuntimeError(
            f"Wrong subset file format.\n Got:      {args.subset}\n Expected: {pattern}"
        )
    N = int(match.group(1))

    with open(args.subset, "r") as f:
        data = f.read().splitlines()
        data = [Path(p) for p in data]
    print("Subset size:", len(data))
    output_folder = f"output/OmnImage{args.imsize}_{N}"
    for im_path in tqdm(data):
        generate(
            im_path=im_path,
            output_folder=output_folder,
            imagenet_dir=args.ims,
            im_size=args.imsize,
        )
