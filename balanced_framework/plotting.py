from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_confusion_matrix_png(
    cm: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))

    thresh = cm.max() * 0.5 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

