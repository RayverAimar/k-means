"""Demo: cluster a synthetic dataset and save a side-by-side plot.

Run from the repo root:
    python demo.py

Outputs ``kmeans_demo.png`` in the current directory.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from kmeans import KMeans


def main() -> None:
    X, y_true = make_blobs(
        centers=4,
        n_samples=600,
        n_features=2,
        cluster_std=1.1,
        shuffle=True,
        random_state=40,
    )

    k = len(np.unique(y_true))
    model = KMeans(K=k, max_iters=150, random_state=42, plot_steps=False)
    y_pred = model.predict(X)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("KMeans clustering — synthetic blobs (k=4)", fontsize=14, fontweight="bold")

    ax_left.scatter(X[:, 0], X[:, 1], c="#888888", s=18, alpha=0.85)
    ax_left.set_title("Before — raw points")
    ax_left.set_xlabel("x1")
    ax_left.set_ylabel("x2")
    ax_left.grid(alpha=0.2)

    palette = plt.cm.tab10(np.linspace(0, 1, k))
    for cluster_idx in range(k):
        mask = y_pred == cluster_idx
        ax_right.scatter(
            X[mask, 0],
            X[mask, 1],
            color=palette[cluster_idx],
            s=22,
            alpha=0.85,
            label=f"cluster {cluster_idx}",
        )

    centroids = np.array(model.centroids)
    ax_right.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=220,
        color="black",
        edgecolor="white",
        linewidth=1.8,
        label="centroids",
        zorder=5,
    )
    ax_right.set_title(f"After — assigned clusters + centroids")
    ax_right.set_xlabel("x1")
    ax_right.set_ylabel("x2")
    ax_right.legend(loc="best", framealpha=0.9)
    ax_right.grid(alpha=0.2)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig("kmeans_demo.png", dpi=140, bbox_inches="tight")
    print("Saved kmeans_demo.png")


if __name__ == "__main__":
    main()
