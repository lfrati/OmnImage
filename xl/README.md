## Step 1 - Extract features

The Imagenet21k (downloaded from https://image-net.org/download-images.php) contains ~~21K~~ 19167 tar files that contain somewhere between 1 and 1k images.

We use `process.py` to extract features from each of those images using a pretrained [VGG network](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19_bn.html) from PyTorch.

As an additional cleaning step we discard:
- images with an alpha channel
- images in grayscale
- classes with < 100 after the above steps

This leaves us with 14436 classes.
To process all the 14k tars in a reasonable time we split them in groups of 200 and process each group in parallel.
To generate the groups use:
```bash
ls <IMAGENETPATH>/winter21_whole > all_tars.txt
mkdir tars_groups
cd tars_groups
split -l 200 ../all_tars.txt
cd ..
```
Use `bash launcher.sh` to call process.py passing each group as an argument (might need to adjust source and destination paths)
