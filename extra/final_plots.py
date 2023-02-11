import json
import numpy as np

with open("transfer.json", "r") as f:
    data = json.loads(f.read())

with open("./cross_transfer.json", "r") as f:
    cross_data = json.loads(f.read())

#%%

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def unzip(l):
    return list(zip(*l))


def CI(xs, conf=0.95):
    from scipy.stats import t

    # The king of speed, is biased estimator so bad?
    # xs : [N, samples]
    m = xs.mean(axis=1)
    s = xs.std(axis=1)
    S = xs.shape[1]  # sample size
    dof = S - 1
    t_crit = np.abs(t.ppf((1 - conf) / 2, dof))
    r = s * t_crit / np.sqrt(S)
    return (m - r, m + r)


def process(repeats, marker, label, color, ax, ci=True, annotate=True):
    ys = np.array([unzip(res)[1] for res in repeats])
    xs = unzip(repeats[0])[0]
    mean = np.mean(ys, axis=0)
    ax.plot(
        xs,
        mean,
        marker,
        label=label,
        color=color,
    )

    if annotate:
        ax.annotate(f"{mean[-1]:.2f}", (xs[-1], mean[-1]), color=color)
    if ci:
        upper, lower = CI(ys.T)
        ax.fill_between(xs, upper, lower, alpha=0.2, color=color)


fig, ax = plt.subplots(figsize=(8, 8))
process(
    data["glot_pretrained"],
    marker=".-",
    label="          OmniGlot train $\\rightarrow$ Omniglot test",
    color="C0",
    ax=ax,
)
process(
    data["glot_untrained"],
    marker=".--",
    label="random init w/o train $\\rightarrow$ OmniGlot test",
    color="C0",
    ax=ax,
)

process(
    data["image_pretrained"],
    marker=".-",
    label="       OmnImage train $\\rightarrow$ OmnImage test",
    color="C1",
    ax=ax,
)
process(
    data["image_untrained"],
    marker=".--",
    label="random init w/o train $\\rightarrow$ OmnImage test",
    color="C1",
    ax=ax,
)

process(
    data["image_random_pretrained"],
    marker=".-",
    label="rand-ImageNet train $\\rightarrow$ rand-ImageNet test",
    color="C2",
    ax=ax,
)
process(
    data["image_random_untrained"],
    marker=".--",
    label="random init w/o train$\\rightarrow$ rand-ImageNet test",
    color="C2",
    ax=ax,
)
plt.grid()
plt.legend(fontsize="medium")
plt.xlabel("# Classes", fontsize=16)
plt.ylabel("KNN Accuracy", fontsize=16)
plt.ylim(0, 1)
# plt.title("KNN learnability in the 20 images regime")
plt.tight_layout()
# plt.savefig("knn_transfer.png")
plt.savefig("knn_transfer.pdf")
# plt.close()
plt.show()


#%%


fig, ax = plt.subplots(figsize=(8, 6))
# process(
#     data["glot_pretrained"],
#     marker=".-",
#     label="OmniGlot train $\\rightarrow$ OmniGlot test",
#     color="C0",
#     ax=ax,
# )
# process(data["glot_untrained"], marker=".--", label="", color="gray", ax=ax, ci=False)
# process(
#     cross_data["glot_pretrained_image"],
#     marker=".--",
#     label="OmnImage train$\\rightarrow$ OmniGlot test",
#     color="C0",
#     ax=ax,
# )
process(
    data["image_pretrained"],
    marker=".-",
    label="OmnImage train $\\rightarrow$ OmnImage test",
    color="C1",
    ax=ax,
)
process(
    cross_data["image_pretrained_glot"],
    marker=".--",
    label="OmniGlot train $\\rightarrow$ OmnImage test",
    color="C0",
    ax=ax,
)
process(
    cross_data["image_pretrained_random"],
    marker=".--",
    label="rand-ImageNet train $\\rightarrow$ OmnImage test",
    color="C2",
    ax=ax,
)
process(data["image_untrained"], marker=".--", label="", color="gray", ax=ax, ci=False)
plt.grid()
plt.legend(fontsize="large")
plt.xlabel("# Classes", fontsize=16)
plt.ylabel("KNN Accuracy", fontsize=16)
plt.ylim(0, 0.7)
plt.tight_layout()
plt.savefig("cross_knn_transfer.pdf")
plt.show()

#%%


def process2(repeats, marker, label, color, ax, ci=True):
    ys = np.array([unzip(res)[1] for res in repeats])
    xs = unzip(repeats[0])[0]
    mean = np.mean(ys, axis=0)
    ax.plot(
        xs,
        mean,
        marker,
        # label=label + f", final= {mean[-1]:.4f}",
        label=label,
        color=color,
    )

    if color == "C1":
        ax.annotate(f"{mean[-1]:.3f}", (xs[-1] * 0.945, mean[-1] * 1.1), color=color)
    if color == "C2":
        ax.annotate(f"{mean[-1]:.3f}", (xs[-1] * 0.945, mean[-1] * 0.84), color=color)
    if color == "gray":
        ax.annotate(f"{mean[-1]:.3f}", (xs[-1] * 0.945, mean[-1] * 1.15), color=color)
    if ci:
        upper, lower = CI(ys.T)
        ax.fill_between(xs, upper, lower, alpha=0.2, color=color)


fig, ax = plt.subplots(figsize=(8, 4))
process2(
    cross_data["random_pretrained_image"],
    marker=".--",
    label="       OmnImage train $\\rightarrow$ rand-ImageNet test",
    color="C1",
    ax=ax,
)
process2(
    data["image_random_pretrained"],
    marker=".-",
    label="  rand-ImageNet test $\\rightarrow$ rand-ImageNet test",
    color="C2",
    ax=ax,
)
process2(
    data["image_random_untrained"],
    marker=".--",
    label="random init w/o train $\\rightarrow$ rand-ImageNet test",
    color="gray",
    ci=False,
    ax=ax,
)
plt.grid()
plt.legend(fontsize="large")
plt.xlabel("# Classes", fontsize=16)
plt.ylabel("KNN Accuracy", fontsize=16)
plt.ylim(0, 0.30)
# plt.ylim(0, 1)
# plt.title("KNN o.o.d. transfer in the 20 images regime")
plt.tight_layout()
plt.savefig("cross_knn_transfer_images.pdf")
plt.show()

#%%

fig, ax = plt.subplots(figsize=(8, 8))
process(
    data["glot_pretrained"],
    marker=".-",
    label="OmniGlot train $\\rightarrow$ OmniGlot test",
    color="C0",
    ax=ax,
)
process(data["glot_untrained"], marker=".--", label="", color="gray", ax=ax, ci=False)
process(
    cross_data["glot_pretrained_image"],
    marker=".--",
    label="OmnImage train$\\rightarrow$ OmniGlot test",
    color="C1",
    ax=ax,
)
process(
    data["image_pretrained"],
    marker=".-",
    label="OmnImage train $\\rightarrow$ OmnImage test",
    color="C1",
    ax=ax,
)
process(
    cross_data["image_pretrained_glot"],
    marker=".--",
    label="OmniGlot train $\\rightarrow$ OmnImage test",
    color="C0",
    ax=ax,
)
process(
    cross_data["image_pretrained_random"],
    marker=".--",
    label="rand-ImageNet train $\\rightarrow$ OmnImage test",
    color="C2",
    ax=ax,
)
process(data["image_untrained"], marker=".--", label="", color="gray", ax=ax, ci=False)
plt.grid()
plt.legend(fontsize="large")
plt.xlabel("# Classes", fontsize=16)
plt.ylabel("KNN Accuracy", fontsize=16)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("full_cross_knn_transfer.pdf")
plt.show()
