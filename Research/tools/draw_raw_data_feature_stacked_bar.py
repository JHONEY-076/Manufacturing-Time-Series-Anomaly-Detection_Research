from pathlib import Path
import os

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAW_DIR = ROOT / "data" / "raw_data"
OUT_DIR = RAW_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RAW_DIR / "train" / "Training_Data.csv")

features = ["Speed", "Length", "RealPower", "SetPower", "GateOnTime"]
top_n = 5
colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#BAB0AC"]

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.color": "#D8D8D8",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#BDBDBD",
        "axes.linewidth": 1.0,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
    }
)

fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

for row, feature in enumerate(features):
    counts = train[feature].value_counts(dropna=False)
    top_counts = counts.head(top_n).copy()
    other_count = counts.iloc[top_n:].sum()
    if other_count > 0:
        top_counts.loc["Other"] = other_count

    total = top_counts.sum()
    left = 0.0
    for idx, (value, count) in enumerate(top_counts.items()):
        percent = count / total * 100
        label = "Other" if value == "Other" else f"{value:g}"
        ax.barh(
            row,
            percent,
            left=left,
            height=0.64,
            color=colors[idx % len(colors)],
            edgecolor="white",
            linewidth=1.0,
        )

        if percent >= 7:
            ax.text(
                left + percent / 2,
                row,
                f"{label}\n{percent:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                weight="bold",
            )
        elif percent >= 5:
            ax.text(
                left + percent / 2,
                row,
                f"{label}\n{percent:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                weight="bold",
            )
        left += percent

ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Proportion within each feature (%)")
ax.set_title("Original Training Data: Feature Value Composition", weight="bold")
ax.grid(axis="x", alpha=0.9)
ax.grid(axis="y", visible=False)

caption = (
    "Each horizontal bar sums to 100%. Segments show the most frequent values "
    "within each feature; remaining values are grouped as Other. "
    "Labels below 5% are omitted for readability."
)
fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=10, color="#555555")

png_path = OUT_DIR / "raw_data_feature_value_stacked_bar.png"
svg_path = OUT_DIR / "raw_data_feature_value_stacked_bar.svg"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print(png_path)
print(svg_path)
