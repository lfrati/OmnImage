import argparse
import logging
from pathlib import Path
import tarfile
import tempfile

import numpy as np
import torch
from torchvision import io
from torchvision.models import VGG19_BN_Weights, vgg19_bn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as fn
from tqdm import tqdm


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


if __name__ == "__main__":
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFMT)

    parser = argparse.ArgumentParser(description="Extract features from an Imagenet21k class using VGG19_BN")
    parser.add_argument(
        "--tars",
        type=str,
        required=True,
        help="the tar file to extract and resize",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="folder to read the tars from",
    )

    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="path to store the tars to",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(args)
    logging.info(f"Device: {device}")

    source = Path(args.source)
    dest = Path(args.dest)

    with open(args.tars, "r") as f:
        tars = f.read().splitlines()
        tars = [source / tar for tar in tars]

    BICUBIC = InterpolationMode.BICUBIC

    for i, tar in enumerate(tars):
        # class e.g. n03592669
        cls = tar.stem

        feats_path = f"{args.dest}/{cls}.npy"
        # since some images are discarded I need to store a list of image names
        # so that I can match features to their images
        ims_info_path = f"{args.dest}/{cls}.txt"

        if Path(ims_info_path).exists():
            logging.warning(f"{ims_info_path} exists. Skipping {cls}")
            continue

        logging.info(f"Processing class {cls}")

        im_names = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tardata = tarfile.open(tar)  # extract the tar to get orginal images
            tardata.extractall(path=tmpdir)  # into a temporary directory
            tardata.close()

            # NOTE: sort for reproducibility!
            images = sorted(list(Path(tmpdir).iterdir()))  # get images from the tmp dir

            N = len(images)
            resized_ims = []
            for j, im_path in enumerate(images):
                im = io.read_image(im_path.as_posix())  # read image values

                crop_size = min(im.shape[-2:])
                square_im = fn.center_crop(im, [crop_size, crop_size])  # make square
                smaller_im = fn.resize(
                    square_im, [84, 84], interpolation=BICUBIC, antialias=True
                )

                # discard alpha channels and black/white images
                if len(smaller_im.shape) != 3 or smaller_im.shape[0] != 3:
                    logging.warning(
                        f"Skipped {j+1:>4d}/{N}: {im_path} (img shape = {smaller_im.shape})"
                    )
                    continue
                else:
                    resized_ims.append(smaller_im / 255.0)
                    im_names.append(im_path.name)
                    # logging.info(f"Resized {j+1:>4d}/{N}: {im_path} {smaller_im.shape}")

        if len(resized_ims) < 100:
            logging.warning(
                f"Skipping class {cls}: Only {len(resized_ims)} images remaining"
            )
            continue

        logging.info(f"Extracting features for class {cls}")
        net = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT, progress=True)
        to_hook = net.classifier[3]

        net = net.to(device)
        forward, _ = hook_it_up(net, to_hook)
        feats = []
        for im in tqdm(resized_ims):
            inp = im.to(device).unsqueeze(0)  # add fake batch dimension
            feats.append(forward(inp).detach().cpu().numpy())

        feats = np.vstack(feats)  # list of N tensors (1, nfeats) -> np.array (N,nfeats)

        np.save(feats_path, feats)
        logging.info(f"Numpy features stored in {feats_path}")
        with open(ims_info_path, "w") as f:
            for im_name in im_names:
                f.write(f"{im_name}\n")
        logging.info(f"Image names stored in {ims_info_path}")

        # sanity check that file names and features match
        feats_retrieved = np.load(feats_path)
        with open(ims_info_path, "r") as f:
            names_retrieved = f.read().splitlines()
        assert len(names_retrieved) == feats_retrieved.shape[0]

        logging.info(f"Completed {i+1}/{len(tars)} : {cls}.")
