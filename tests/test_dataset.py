from time import monotonic
import matplotlib.pyplot as plt

from dataset import OmnImageDataset
from memmap import store_memmap, retrieve_memmap, MemmapDataset
from tqdm import trange
import numpy as np

dataset = OmnImageDataset("./data/OmnImage84_20")

MMAP_FILE = "./data/mmap_OmnImage_20.dat"
DATA_LOC = "./data/OmnImage_20"

store_memmap(MMAP_FILE, dataset)
retrieved_data = retrieve_memmap(MMAP_FILE)

print("Verifying retrieved data...")
for i in trange(len(dataset)):
    orig, _ = dataset[i]
    retr = retrieved_data[i]
    assert np.allclose(
        orig, retr
    ), f"Element {i} don't match between dataset and retrieved."
print("OK")


#%%

mmdataset = MemmapDataset(MMAP_FILE)
print(mmdataset)

ten, label = mmdataset[0]
img = mmdataset.to_image(ten)

plt.imshow(img)
plt.show()

#%%

dataset = OmnImageDataset("./data/OmnImage84_20", memoize=True)
memdataset = MemmapDataset(MMAP_FILE)

print("Dataset shape", memdataset.shape)
print("No memoization")
for i in trange(len(dataset)):
    img, label = dataset[i]
    assert img.shape == (3, 84, 84)
print("With memoization")
for i in trange(len(dataset)):
    img, label = dataset[i]
    assert img.shape == (3, 84, 84)
print("Memory mapped")
for i in trange(len(memdataset)):
    img, _ = memdataset[i]
    assert img.shape == (3, 84, 84)


#%%


def test(dataset, shuffle, trials, nworkers):
    print(f"bsize          async    block")
    for bsize in [32, 64, 128, 256, 512]:
        loader = DataLoader(
            dataset,
            batch_size=bsize,
            pin_memory=True,
            shuffle=shuffle,
            num_workers=nworkers,
        )
        noblock_times = []
        for i in range(trials):
            start = monotonic()
            for i, (ims, labels) in enumerate(async_loader(loader)):
                _ = ims.std()
            end = monotonic()
            noblock_times.append(end - start)
        block_times = []
        for i in range(trials):
            start = monotonic()
            for i, (ims, labels) in enumerate(loader):
                ims = ims.cuda()
                _ = ims.std()
            end = monotonic()
            block_times.append(end - start)
        # print(noblock, block)
        noblock = np.mean(noblock_times)
        block = np.mean(block_times)
        better = (noblock - block) / block
        print(f"{bsize:>4} : {better*100:+.2f}% ({noblock:.2f} vs {block:.2f})")


# no memoization, 8 workers
dataset = OmnImageDataset("OmnImage84_100")
test(dataset, shuffle=True, trials=5, nworkers=8)
# bsize          async    block
#   32 : -1.47% (4.99 vs 5.06)
#   64 : +1.55% (4.40 vs 4.33)
#  128 : +0.95% (4.43 vs 4.39)
#  256 : +2.39% (4.40 vs 4.30)
#  512 : -0.65% (4.23 vs 4.26)

# # memoization, 1 workers
dataset = OmnImageDataset("OmnImage84_100", memoize=True)
# preload
for i in trange(len(dataset)):
    v = dataset[i]
test(dataset, shuffle=True, trials=5, nworkers=0)
# bsize          async    block
#   32 : -16.69% (3.09 vs 3.71)
#   64 : -19.91% (2.90 vs 3.62)
#  128 : -14.99% (2.96 vs 3.48)
#  256 : -25.83% (2.79 vs 3.76)
#  512 : -15.62% (2.50 vs 2.96)

# memory mapped, 1 worker
dataset = MemmapDataset(MMAP_FILE)
test(dataset, shuffle=True, trials=5, nworkers=0)
# bsize          async    block
#   32 : -3.79% (2.05 vs 2.13)
#   64 : -32.42% (1.55 vs 2.30)
#  128 : -30.24% (1.81 vs 2.60)
#  256 : -23.68% (2.06 vs 2.70)
#  512 : -21.40% (2.07 vs 2.63)


#%%

# train, test, tr_cls, te_cls = split(dataset, verbose=True, seed=4)
# print(len(train), len(test))

# device = "cuda"
# sampler = OMLSampler(train, device)
# print(sampler)

# # OMLSampler(dataset) to use the whole dataset instead

# # inner/outer sample sizes
# I, O = 15, 32
# inner_ims, inner_labels, outer_ims, outer_labels = sampler.sample(
#     inner_size=I, outer_size=O
# )
# print(inner_ims.shape)
# print(inner_labels.shape)
# print(outer_ims.shape)
# print(outer_labels.shape)
# assert (
#     inner_ims.shape[0] == I
# ), f"Inner loop mismatch: Expected {I} found {inner_ims.shape[0]}"
# assert (
#     outer_ims.shape[0] == I + O
# ), f"Outer loop mismatch: Expected {I+O} found {outer_ims.shape[0]}"

# ims, labels = sampler.sample_balanced()
# assert ims.device.type == device
# assert len(set([label.item() for label in labels])) == len(labels)
