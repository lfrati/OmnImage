![](assets/banner.png)

# OmnImage

The [OmniImage](https://drive.google.com/drive/folders/1tpm4LY3gEUlpK7h54uHqKOcw353np05s?usp=sharing) dataset contains a 1000 classes with 20|100 [images](https://drive.google.com/file/d/1rR7Xh4sxXVY9im_DrjZwEf2oydmx8YWL/view) each, downsized to 32x32|64x64|84x84 pixels.

# Why?

- [MNIST](https://en.wikipedia.org/wiki/MNIST_database) has 10 classes and thousands of examples for each class.
- [ImageNet](https://www.image-net.org/) has 1000 classes with thousands of example each.
- [Omniglot](https://github.com/brendenlake/omniglot) has 1623 classes with tens of examples each.

<p align="center">
  <img align="center" width="400" alt="Dataset Area" src="https://user-images.githubusercontent.com/3115640/204098543-1bc2406f-487f-4c06-8b4f-224c0e2e2840.png">
</p>

However, while ImageNet contains natural images MNIST and Omniglot only contain examples of handwritten digits/characters.

[OmniImage](https://drive.google.com/drive/folders/1tpm4LY3gEUlpK7h54uHqKOcw353np05s?usp=sharing) is a class-consistent subset of [ImageNet](https://www.image-net.org/) images that mirrors the shallow-and-wide dataset shape of [Omniglot](https://github.com/brendenlake/omniglot).

What does consistent mean?
While characters are fairly easy to "characterize" (minus some bad handwriting) natural images can vary wildly. We try to reduce the noise in our dataset by extracting the 20|100 subset of each class that is most similar to each other.

We do this by using [evolutionary pairwise cosines subset minimization](https://github.com/lfrati/subpair). For each of the 1000 classes we compute the pairwise cosine distance between all the the examples (features extracted using a pretrained VGG model). We then evolve the minimal subset for each class and add those examples to our dataset.

See [this link](https://drive.google.com/file/d/1rR7Xh4sxXVY9im_DrjZwEf2oydmx8YWL/view) for an overview of the 20000 version.
