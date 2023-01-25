import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from tqdm import tqdm

from subpair import extract


# ~/../Storage/data_dungeon/image_data
loc = Path("torchvision.models.vgg_ce631fc9ca0278a2")

classes = list(loc.iterdir())

class_names = np.array([cls.stem for cls in classes])

mean_acts = np.stack([np.load(cls).mean(axis=0) for cls in tqdm(classes)])


#%%


def pairwise_cosine_numpy(X):
    norms = np.einsum("ij,ij->i", X, X)
    np.sqrt(norms, norms)
    X /= norms[:, np.newaxis]
    dists = 1 - np.dot(X, X.T)
    return dists


pairwise_mean = pairwise_cosine_numpy(mean_acts)

# np.save("pairwise_mean_classes", pairwise_mean)
# np.save("pairwise_mean_classes_names", class_names)

#%%


distances = np.load("../OmnImage/pairwise_mean_classes.npy")
best, stats = extract(distances, P=200, S=300, K=50, M=3, O=2, its=10_000)

print("Final fit:", stats["fits"][-1])

plt.plot(stats["fits"])
plt.grid()
plt.ylabel("Subset distances")
plt.xlabel("Iterations")
plt.title("Minimal subset extraction (lower is better)")
plt.tight_layout()
plt.savefig("min_subset.png", dpi=300)
plt.show()

test_classes = sorted(best)


with open("test_class_names.json", "w") as f:
    f.write(json.dumps(class_names[test_classes].tolist()))
