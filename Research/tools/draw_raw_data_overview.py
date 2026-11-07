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

train_path = RAW_DIR / "train" / "Training_Data.csv"
ng3_path = RAW_DIR / "test" / "WeldingTest_03_NG_labeled.csv"
ng4_path = RAW_DIR / "test" / "WeldingTest_04_NG_labeled.csv"

train = pd.read_csv(train_path)
ng3 = pd.read_csv(ng3_path)
ng4 = pd.read_csv(ng4_path)

plot_columns = ["Speed", "Length", "RealPower", "SetPower", "GateOnTime"]
summary_columns = ["Speed", "Length", "SetPower", "GateOnTime"]

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.color": "#D0D0D0",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#BDBDBD",
        "axes.linewidth": 1.0,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
axes = axes.ravel()

palette = {
    "Speed": "#4C78A8",
    "Length": "#F58518",
    "RealPower": "#54A24B",
    "SetPower": "#B279A2",
    "GateOnTime": "#E45756",
}

for ax, col in zip(axes[:5], plot_columns):
    values = train[col].dropna()
    unique_count = values.nunique()
    bins = min(max(unique_count, 8), 35)
    ax.hist(
        values.to_numpy(),
        bins=bins,
        color=palette[col],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.72,
    )
    ax.set_title(f"Train distribution: {col}", fontsize=12, weight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.margins(x=0.04)

summary_ax = axes[5]
summary_ax.axis("off")

summary_lines = [
    "Raw Data Overview",
    "",
    f"Train rows: {len(train):,}",
    f"Train variables: {len(train.columns)}",
    f"NG3 rows: {len(ng3):,}  | anomaly labels: {int(ng3['label'].sum()):,}",
    f"NG4 rows: {len(ng4):,}  | anomaly labels: {int(ng4['label'].sum()):,}",
    "",
    "Discrete process settings in train:",
]

for col in summary_columns:
    top_values = train[col].value_counts().head(3)
    top_text = ", ".join(f"{idx:g} ({cnt:,})" for idx, cnt in top_values.items())
    summary_lines.append(f"- {col}: {train[col].nunique()} unique; top {top_text}")

summary_lines.extend(
    [
        "",
        "Interpretation:",
        "Most process variables are concentrated",
        "around a few recipe-like operating values.",
    ]
)

summary_ax.text(
    0.02,
    0.98,
    "\n".join(summary_lines),
    va="top",
    ha="left",
    fontsize=11,
    family="DejaVu Sans",
    linespacing=1.35,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7", edgecolor="#D0D0D0"),
)

fig.suptitle(
    "Original Welding Data Characteristics",
    fontsize=16,
    weight="bold",
)

png_path = OUT_DIR / "raw_data_overview_distributions.png"
svg_path = OUT_DIR / "raw_data_overview_distributions.svg"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print(png_path)
print(svg_path)
