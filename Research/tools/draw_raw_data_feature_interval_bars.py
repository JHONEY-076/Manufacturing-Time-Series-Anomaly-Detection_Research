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
colors = {
    "Speed": "#4C78A8",
    "Length": "#F58518",
    "RealPower": "#54A24B",
    "SetPower": "#B279A2",
    "GateOnTime": "#E45756",
}

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.color": "#D8D8D8",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#BDBDBD",
        "axes.linewidth": 1.0,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
    }
)


def format_interval(interval: pd.Interval) -> str:
    left = f"{interval.left:g}"
    right = f"{interval.right:g}"
    return f"{left}-{right}"


fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
axes = axes.ravel()

for ax, feature in zip(axes[:5], features):
    values = train[feature].dropna()
    unique_count = values.nunique()
    bins = min(10, max(4, unique_count))

    binned = pd.cut(values, bins=bins, include_lowest=True)
    counts = binned.value_counts(sort=False)
    labels = [format_interval(interval) for interval in counts.index]

    bars = ax.bar(
        labels,
        counts.values,
        color=colors[feature],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.82,
    )

    max_count = counts.max()
    for bar, count in zip(bars, counts.values):
        if count / max_count >= 0.08:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    ax.set_title(f"{feature} interval distribution", weight="bold")
    ax.set_xlabel("Value interval")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.9)
    ax.grid(axis="x", visible=False)

axes[5].axis("off")
axes[5].text(
    0.02,
    0.98,
    "\n".join(
        [
            "Interval Graph Summary",
            "",
            "Data: original training set",
            f"Rows: {len(train):,}",
            "",
            "Each feature is divided into",
            "equal-width value intervals.",
            "",
            "This view emphasizes where",
            "the original process data are",
            "concentrated across ranges.",
        ]
    ),
    ha="left",
    va="top",
    fontsize=12,
    linespacing=1.4,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7", edgecolor="#D0D0D0"),
)

fig.suptitle("Original Training Data: Interval-Based Feature Distributions", fontsize=16, weight="bold")

png_path = OUT_DIR / "raw_data_feature_interval_bars.png"
svg_path = OUT_DIR / "raw_data_feature_interval_bars.svg"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print(png_path)
print(svg_path)
