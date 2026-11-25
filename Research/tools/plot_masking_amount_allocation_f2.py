from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output/section_4_3_unified/compare_1000_f2/masking_f2.csv"
OUT = RESULTS.parent / "masking_amount_f2.png"
os.environ["MPLCONFIGDIR"] = str(RESULTS.parent / ".mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(RESULTS)
amount = df[df["experiment_family"] == "Total masked amount"].sort_values("masked_cells")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

for metric, label, color, marker in [
    ("precision", "Precision", "#4C78A8", "o"),
    ("recall", "Recall", "#F58518", "s"),
    ("f1", "F1-score", "#54A24B", "^"),
    ("f2", "F2-score", "#E45756", "D"),
    ("ap", "AP", "#B279A2", "P"),
]:
    axes[0].plot(amount["masked_cells"], amount[metric], label=label,
                 color=color, marker=marker, linewidth=2, markersize=6)
axes[0].set_title("Performance by Total Masking Amount")
axes[0].set_xlabel("Masking Configuration")
axes[0].set_ylabel("Score")
axes[0].set_xticks(amount["masked_cells"], [
    "4 steps × 1 feature\n46/150 (30.7%)",
    "8 steps × 2 features\n84/150 (56.0%)",
    "12 steps × 3 features\n114/150 (76.0%)",
])
axes[0].set_ylim(0.65, 1.02)
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False, ncol=2, loc="lower right")

x = list(range(len(amount)))
width = 0.34
fp_bars = axes[1].bar([i-width/2 for i in x], amount["fp"], width,
                      label="FP", color="#4C78A8")
fn_bars = axes[1].bar([i+width/2 for i in x], amount["fn"], width,
                      label="FN", color="#E07B61")
axes[1].set_title("False Positives and False Negatives")
axes[1].set_ylabel("Number of Windows")
axes[1].set_xlabel("Masked Proportion")
axes[1].set_xticks(x, ["30.7%", "56.0%", "76.0%"])
axes[1].set_ylim(0, max(amount["fp"].max(), amount["fn"].max()) * 1.25)
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend(frameon=False)
for bars in [fp_bars, fn_bars]:
    axes[1].bar_label(bars, padding=3, fontsize=9)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(OUT)
