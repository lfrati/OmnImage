from omnimage import *

import pytest

DOWNLOAD_LOC = "./data"


@pytest.mark.parametrize("samples", [20, 100])
def test_download(samples):
    print()
    DATA_LOC = f"{DOWNLOAD_LOC}/OmnImage84_{samples}"
    download_dataset(samples=samples, download_root="data")
    dataset = OmnImageDataset(DATA_LOC)
    print(dataset)
    assert len(dataset) == samples * 1_000


@pytest.mark.parametrize("samples", [20, 100])
def test_memmap(samples):
    print()
    DATA_LOC = f"{DOWNLOAD_LOC}/OmnImage84_{samples}"
    MMAP_FILE = f"{DOWNLOAD_LOC}mmap_OmnImage84_{samples}.dat"
    dataset = OmnImageDataset(DATA_LOC)
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


@pytest.mark.parametrize("samples", [20, 100])
def test_memmap_dataset(samples):
    MMAP_FILE = f"{DOWNLOAD_LOC}mmap_OmnImage84_{samples}.dat"
    mmdataset = MemmapDataset(MMAP_FILE)
    print()
    print(mmdataset)
    assert len(mmdataset) == samples * 1_000

    for i in trange(len(mmdataset)):
        orig, _ = mmdataset[i]
        assert orig.shape == (3, 84, 84)


@pytest.mark.parametrize("samples", [20, 100])
def test_split(samples):
    MMAP_FILE = f"{DOWNLOAD_LOC}mmap_OmnImage84_{samples}.dat"
    mmdataset = MemmapDataset(MMAP_FILE)
    print(len(mmdataset))
    p = 0.8
    train, test, _, _ = split(mmdataset, p=p, samples=samples, seed=4)
    ntrain = int(len(mmdataset) * p)
    ntest = len(mmdataset) - ntrain
    assert len(train) == ntrain
    assert len(test) == ntest


@pytest.mark.parametrize("samples", [20, 100])
def test_sampler_class(samples):
    MMAP_FILE = f"{DOWNLOAD_LOC}mmap_OmnImage84_{samples}.dat"
    mmdataset = MemmapDataset(MMAP_FILE)
    sampler = Sampler(mmdataset)
    print(sampler)

    N = 20
    for _ in range(10):
        ims, labels = sampler.sample_class(N=N)
        assert ims.shape == (N, 3, 84, 84)
        assert labels.shape == (N,)
        unique_labels = set([l.item() for l in labels])
        assert len(unique_labels) == 1


@pytest.mark.parametrize("samples", [20, 100])
def test_sampler_random(samples):
    MMAP_FILE = f"{DOWNLOAD_LOC}mmap_OmnImage84_{samples}.dat"
    mmdataset = MemmapDataset(MMAP_FILE)

    sampler = Sampler(mmdataset)
    print(sampler)

    N = 64
    ims, labels = sampler.sample_random(N=N)
    assert ims.shape == (N, 3, 84, 84)
    assert len(labels) == N
