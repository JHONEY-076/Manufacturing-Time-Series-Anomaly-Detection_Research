from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/section_4_3_unified/ppt_aligned/masking_amount_methods_performance.png"

x = [46, 84, 114]
xlabels = ["46 / 150\n(30.7%)", "84 / 150\n(56.0%)", "114 / 150\n(76.0%)"]
methods = {
    "Masked GT-GAN": {
        "color": "#2563eb",
        "precision": [0.9741, 0.9740, 0.9653],
        "recall": [0.6351, 0.6318, 0.5642],
        "f1": [0.7689, 0.7664, 0.7122],
        "auroc": [0.9636, 0.9672, 0.9610],
        "ap": [0.8459, 0.8519, 0.8242],
        "fp": [5, 5, 6], "fn": [108, 109, 129],
    },
    "Masked Diffusion": {
        "color": "#16a34a",
        "precision": [0.9069, 0.9953, 0.9848],
        "recall": [0.7568, 0.7162, 0.6588],
        "f1": [0.8250, 0.8330, 0.7895],
        "auroc": [0.9852, 0.9888, 0.9850],
        "ap": [0.9125, 0.9324, 0.9107],
        "fp": [23, 1, 3], "fn": [72, 84, 101],
    },
    "FMD": {
        "color": "#ea580c",
        "precision": [0.8296, 0.8863, 0.9189],
        "recall": [0.8716, 0.8953, 0.8041],
        "f1": [0.8501, 0.8908, 0.8577],
        "auroc": [0.9913, 0.9921, 0.9908],
        "ap": [0.9411, 0.9553, 0.9422],
        "fp": [53, 34, 21], "fn": [38, 31, 58],
    },
}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
fig, axes = plt.subplots(2, 3, figsize=(12, 7.3))
metrics = [
    ("precision", "Precision", (0.50, 1.02)),
    ("recall", "Recall", (0.50, 1.02)),
    ("f1", "F1-score", (0.50, 1.02)),
    ("auroc", "AUROC", (0.94, 1.005)),
    ("ap", "Average Precision (AP)", (0.80, 1.005)),
]

for ax, (key, title, ylim) in zip(axes.flat[:5], metrics):
    for name, result in methods.items():
        ax.plot(x, result[key], color=result["color"], marker="o", linewidth=2,
                markersize=5, label=name)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Masked Cells")
    ax.grid(True, alpha=0.25)

ax = axes.flat[5]
for name, result in methods.items():
    ax.plot(x, result["fp"], color=result["color"], marker="o", linewidth=2)
    ax.plot(x, result["fn"], color=result["color"], marker="s", linewidth=2, linestyle="--")
ax.set_title("False Positive and False Negative Counts")
ax.set_ylabel("Window Count")
ax.set_xticks(x, xlabels)
ax.set_xlabel("Masked Cells")
ax.grid(True, alpha=0.25)

method_handles = [Line2D([0], [0], color=m["color"], marker="o", linewidth=2, label=name)
                  for name, m in methods.items()]
style_handles = [Line2D([0], [0], color="#333333", marker="o", linewidth=2, label="FP"),
                 Line2D([0], [0], color="#333333", marker="s", linewidth=2,
                        linestyle="--", label="FN")]
fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
           ncol=3, frameon=False)
axes.flat[5].legend(handles=style_handles, loc="upper right", frameon=True)
fig.suptitle("Performance by Masking Amount and Method", fontsize=14, y=1.01)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(OUT)
