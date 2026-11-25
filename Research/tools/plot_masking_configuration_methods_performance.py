from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/section_4_3_unified/ppt_aligned/mask_configuration_methods_performance.png"

x = [0, 1]
xlabels = ["Temporal-focused", "Feature-focused"]
methods = {
    "Masked GT-GAN": {
        "color": "#2563eb",
        "precision": [0.9415, 0.9672], "recall": [0.5439, 0.5980],
        "f1": [0.6895, 0.7390], "auroc": [0.9578, 0.9669],
        "ap": [0.8047, 0.8423], "fp": [10, 6], "fn": [135, 119],
    },
    "Masked Diffusion": {
        "color": "#16a34a",
        "precision": [0.9195, 0.9683], "recall": [0.8108, 0.7230],
        "f1": [0.8618, 0.8279], "auroc": [0.9896, 0.9850],
        "ap": [0.9386, 0.9152], "fp": [21, 7], "fn": [56, 82],
    },
    "FMD": {
        "color": "#ea580c",
        "precision": [0.8019, 0.8638], "recall": [0.8750, 0.7500],
        "f1": [0.8368, 0.8029], "auroc": [0.9896, 0.9837],
        "ap": [0.9249, 0.9019], "fp": [64, 35], "fn": [37, 74],
    },
}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
fig, axes = plt.subplots(2, 3, figsize=(12, 7.3))
metrics = [
    ("precision", "Precision", (0.50, 1.02)),
    ("recall", "Recall", (0.50, 1.02)),
    ("f1", "F1-score", (0.50, 1.02)),
    ("auroc", "AUROC", (0.94, 1.005)),
    ("ap", "Average Precision (AP)", (0.78, 0.97)),
]

for ax, (key, title, ylim) in zip(axes.flat[:5], metrics):
    for name, result in methods.items():
        ax.plot(x, result[key], color=result["color"], marker="o", linewidth=2,
                markersize=6, label=name)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.set_xticks(x, xlabels, rotation=12, ha="right")
    ax.set_xlabel("Masking Configuration")
    ax.grid(True, alpha=0.25)

ax = axes.flat[5]
for name, result in methods.items():
    ax.plot(x, result["fp"], color=result["color"], marker="o", linewidth=2)
    ax.plot(x, result["fn"], color=result["color"], marker="s", linewidth=2, linestyle="--")
ax.set_title("False Positive and False Negative Counts")
ax.set_ylabel("Window Count")
ax.set_xticks(x, xlabels, rotation=12, ha="right")
ax.set_xlabel("Masking Configuration")
ax.grid(True, alpha=0.25)

method_handles = [Line2D([0], [0], color=m["color"], marker="o", linewidth=2, label=name)
                  for name, m in methods.items()]
style_handles = [Line2D([0], [0], color="#333333", marker="o", linewidth=2, label="FP"),
                 Line2D([0], [0], color="#333333", marker="s", linewidth=2,
                        linestyle="--", label="FN")]
fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
           ncol=3, frameon=False)
axes.flat[5].legend(handles=style_handles, loc="upper right", frameon=True)
fig.suptitle("Performance by Masking Configuration and Method", fontsize=14, y=1.01)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(OUT)
