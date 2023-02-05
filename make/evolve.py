import argparse
from os.path import join
from pathlib import Path
import pickle
import sys
from typing import List

import numpy as np

from subpair import extract, pairwise_cosine


def read_folder(folder: Path) -> List[Path]:
    if isinstance(folder, str):
        folder = Path(folder)
    return sorted(folder.iterdir())


def existing_dir(arg):
    path = Path(arg).resolve()
    if not path.is_dir():
        raise argparse.ArgumentError(f"Not a valid directory: {arg} ({path})")
    return path


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ims",
        type=existing_dir,
        default="imagenet",
        help="path to folder w/ imagenet data",
    )
    parser.add_argument(
        "--feats",
        type=existing_dir,
        help="path to folder w/ extracted features",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="How many samples to take from each class",
    )
    parser.add_argument(
        "--its",
        type=int,
        default=3_000,
        help="How many iterations to run the evolutionary algorithm for",
    )

    args = parser.parse_args(args)
    print(args)
    im_folder = args.ims
    hist_folder = Path("output/hist")
    hist_folder.mkdir(parents=True, exist_ok=True)

    with open(f"output/OmnImage_{args.samples}.txt", "w") as output_file:
        for f in args.feats.iterdir():
            cls = f.stem
            print(f"Processing class {cls}")
            folder = read_folder(im_folder / cls)

            # drop top folder name
            images = np.array([join(*im.parts[1:]) for im in folder])

            X = np.load(f)
            distances = pairwise_cosine(X)
            best, stats = extract(distances, P=200, S=20, K=50, M=3, O=2, its=args.its)

            selected = np.sort(best)
            for i in selected:
                img = images[i]
                print(f"Adding {img}")
                output_file.write(img + "\n")
            with open(hist_folder / f"{cls}.pkl", "wb") as p:
                pickle.dump((best, stats), p)


if __name__ == "__main__":
    sys.exit(main())
