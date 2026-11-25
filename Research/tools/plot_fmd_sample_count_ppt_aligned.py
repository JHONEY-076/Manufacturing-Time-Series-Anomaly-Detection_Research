from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/section_4_3_unified/ppt_aligned/generated_count.csv"
OUT = ROOT / "output/section_4_3_unified/ppt_aligned/fmd_sample_count_performance.png"

df = pd.read_csv(DATA)
df["added_anomalies"] = df["added_anomalies"].astype(int)
df = df.sort_values("added_anomalies")
x = df["added_anomalies"]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={"width_ratios": [1.55, 1]})

metric_styles = {
    "precision": ("Precision", "#2563eb", "o"),
    "recall": ("Recall", "#dc2626", "s"),
    "f1": ("F1-score", "#16a34a", "^"),
    "auroc": ("AUROC", "#7c3aed", "D"),
    "ap": ("AP", "#ea580c", "P"),
}
for column, (label, color, marker) in metric_styles.items():
    axes[0].plot(x, df[column], label=label, color=color, marker=marker,
                 linewidth=2, markersize=5)
axes[0].set_title("Performance by Number of Added Anomaly Samples")
axes[0].set_xlabel("Added Anomaly Samples")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0.45, 1.02)
axes[0].set_xticks(x)
axes[0].grid(True, alpha=0.25)
axes[0].legend(ncol=2, frameon=True, loc="lower right")

positions = range(len(df))
width = 0.36
axes[1].bar([p - width / 2 for p in positions], df["fp"], width=width, label="FP", color="#0284c7")
axes[1].bar([p + width / 2 for p in positions], df["fn"], width=width, label="FN", color="#f97316")
axes[1].set_title("False Positive and False Negative Counts")
axes[1].set_xlabel("Added Anomaly Samples")
axes[1].set_ylabel("Window Count")
axes[1].set_xticks(list(positions), [str(n) for n in x])
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend(frameon=True)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(OUT)
