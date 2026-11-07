from pathlib import Path
import os

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_data"
OUT = ROOT / "pictures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = [
    "Speed",
    "Length",
    "RealPower",
    "SetFrequency",
    "SetDuty",
    "SetPower",
    "GateOnTime",
]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "WorkingTime" in df.columns:
        df["WorkingTime"] = pd.to_datetime(df["WorkingTime"].astype(str).str.strip(), errors="coerce")
    return df


def savefig(name: str) -> None:
    for ext in ("png", "svg"):
        plt.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_overview(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(14, 12), sharex=True)
    x = np.arange(len(train))
    for ax, feature in zip(axes, FEATURES):
        ax.plot(x, train[feature], color="#1f77b4", linewidth=0.6)
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time index")
    fig.suptitle("Original Training Data: Multivariate Time-Series Overview", fontsize=15, fontweight="bold")
    savefig("original_training_timeseries_overview")


def plot_standardized_overlay(train: pd.DataFrame) -> None:
    sample = train.iloc[:: max(len(train) // 5000, 1)].copy()
    x = np.arange(len(sample))
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for feature, color in zip(FEATURES, colors):
        values = sample[feature].astype(float)
        std = values.std(ddof=0)
        z = (values - values.mean()) / std if std > 0 else values - values.mean()
        ax.plot(x, z, label=feature, linewidth=0.9, alpha=0.85, color=color)
    ax.set_title("Original Training Data: Standardized Feature Overlay", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sampled time index")
    ax.set_ylabel("Standardized value")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, frameon=False, fontsize=9)
    savefig("original_training_standardized_feature_overlay")


def plot_ok_ng_comparison() -> None:
    ok = read_csv(RAW / "test" / "WeldingTest_01_OK.csv")
    ng = read_csv(RAW / "test" / "WeldingTest_03_NG_labeled.csv")
    compare_features = ["Length", "RealPower", "GateOnTime"]
    fig, axes = plt.subplots(len(compare_features), 1, figsize=(22, 9), sharex=False)
    for ax, feature in zip(axes, compare_features):
        ax.plot(np.arange(len(ok)), ok[feature], label="OK sample", color="#1f77b4", linewidth=1.0)
        ax.plot(np.arange(len(ng)), ng[feature], label="NG sample", color="#d62728", linewidth=1.0, alpha=0.88)
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Time index")
    fig.suptitle("Original Test Data: Representative OK vs NG Time-Series", fontsize=16, fontweight="bold")
    savefig("original_ok_ng_representative_timeseries")


def shade_anomalies(ax: plt.Axes, labels: pd.Series) -> None:
    values = labels.fillna(0).astype(int).to_numpy()
    starts = np.where((values == 1) & (np.r_[0, values[:-1]] == 0))[0]
    ends = np.where((values == 1) & (np.r_[values[1:], 0] == 0))[0]
    for start, end in zip(starts, ends):
        ax.axvspan(start, end, color="#d62728", alpha=0.12, linewidth=0)


def plot_labeled_ng(name: str, filename: str) -> None:
    df = read_csv(RAW / "test" / filename)
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(df))
    for ax, feature in zip(axes[:3], ["Length", "RealPower", "GateOnTime"]):
        ax.plot(x, df[feature], color="#262626", linewidth=0.8)
        shade_anomalies(ax, df["label"])
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.25)
    axes[3].step(x, df["label"], where="mid", color="#d62728", linewidth=1.0)
    axes[3].set_ylabel("Label")
    axes[3].set_xlabel("Time index")
    axes[3].set_yticks([0, 1])
    axes[3].grid(True, alpha=0.25)
    fig.suptitle(f"Original Labeled NG Data: {name} Anomaly Regions", fontsize=15, fontweight="bold")
    savefig(f"original_{name.lower()}_labeled_anomaly_regions")


def plot_feature_line_panels(train: pd.DataFrame) -> None:
    sample = train.iloc[:: max(len(train) // 6000, 1)].copy()
    x = np.arange(len(sample))
    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, sample[feature], color="#1f77b4", linewidth=1.0)
        ax.set_title(f"Original Training Data: {feature} Line Plot", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sampled time index")
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.25)
        savefig(f"original_line_{feature.lower()}")


def plot_ok_ng_feature_line_panels() -> None:
    ok = read_csv(RAW / "test" / "WeldingTest_01_OK.csv")
    ng = read_csv(RAW / "test" / "WeldingTest_03_NG_labeled.csv")
    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(len(ok)), ok[feature], label="OK", color="#1f77b4", linewidth=1.0)
        ax.plot(np.arange(len(ng)), ng[feature], label="NG", color="#d62728", linewidth=1.0, alpha=0.85)
        ax.set_title(f"Original Test Data: OK vs NG {feature} Line Plot", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time index")
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        savefig(f"original_line_ok_ng_{feature.lower()}")


def plot_stacked_window_style() -> None:
    ok1 = read_csv(RAW / "test" / "WeldingTest_01_OK.csv")
    ok2 = read_csv(RAW / "test" / "WeldingTest_02_OK.csv")
    ng = read_csv(RAW / "test" / "WeldingTest_03_NG_labeled.csv")
    feature = "RealPower"
    window = 220
    series_list = [
        ("OK window 1", ok1[feature].iloc[300 : 300 + window].reset_index(drop=True), None),
        ("OK window 2", ok2[feature].iloc[700 : 700 + window].reset_index(drop=True), None),
        ("NG window", ng[feature].iloc[560 : 560 + window].reset_index(drop=True), ng["label"].iloc[560 : 560 + window].reset_index(drop=True)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 5.4), sharex=True)
    fig.patch.set_facecolor("#f8fafc")

    for idx, (ax, (title, values, labels)) in enumerate(zip(axes, series_list)):
        x = np.arange(len(values))
        ax.set_facecolor("white")
        ax.plot(x, values, color="#1f77b4", linewidth=1.8)
        if labels is not None and labels.sum() > 0:
            anomaly = labels.astype(bool).to_numpy()
            ax.plot(x[anomaly], values[anomaly], color="#d62728", linewidth=2.2)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.22)
        ax.tick_params(axis="both", labelsize=8, length=3, colors="#475569")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(1.0)
        if idx < 2:
            ax.text(
                0.5,
                -0.28,
                "...",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=18,
                color="#111827",
                fontweight="bold",
            )

    axes[-1].set_xlabel("Time index", fontsize=10)
    fig.suptitle("Original Manufacturing Time-Series Windows", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.42)
    savefig("original_stacked_window_anomaly_style")


def plot_compact_three_feature_raw_series() -> None:
    df = read_csv(RAW / "test" / "WeldingTest_03_NG_labeled.csv")
    features = ["Length", "RealPower", "GateOnTime"]
    x = np.arange(len(df))

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.2), sharex=True)
    for ax, feature in zip(axes, features):
        ax.plot(x, df[feature], color="#1f77b4", linewidth=1.15)
        ax.set_ylabel(feature, fontsize=8)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=8, length=3, width=0.8)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")
            spine.set_linewidth(1.0)

    axes[-1].set_xlabel("Time index", fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.13, hspace=0.22)
    savefig("original_compact_three_feature_raw_series")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    train = read_csv(RAW / "train" / "Training_Data.csv")
    plot_training_overview(train)
    plot_standardized_overlay(train)
    plot_ok_ng_comparison()
    plot_labeled_ng("WeldingTest03NG", "WeldingTest_03_NG_labeled.csv")
    plot_labeled_ng("WeldingTest04NG", "WeldingTest_04_NG_labeled.csv")
    plot_stacked_window_style()
    plot_compact_three_feature_raw_series()


if __name__ == "__main__":
    main()
