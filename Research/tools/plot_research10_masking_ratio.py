from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
RESULT_PATH = ROOT / "data" / "research10" / "results" / "research10_masking_ratio.csv"
PICTURE_DIR = ROOT / "pictures"
PICTURE_DIR.mkdir(parents=True, exist_ok=True)

RATIO_ORDER = ["small", "default", "large"]
COLORS = {
    "precision": "#2f6f9f",
    "recall": "#3c9d5d",
    "f1": "#c94f4f",
    "f2": "#8064a2",
    "auprc": "#d97941",
    "false_negative": "#b91c1c",
    "false_positive": "#6b7280",
}


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(masking_ratio=pd.Categorical(df["masking_ratio"], RATIO_ORDER, ordered=True))
        .sort_values("masking_ratio")
        .reset_index(drop=True)
    )


def save_score_bars(df: pd.DataFrame) -> None:
    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("f2", "F2-score"),
        ("auprc", "AUPRC"),
    ]
    x = np.arange(len(RATIO_ORDER))
    width = 0.14
    offsets = np.linspace(-2, 2, len(metrics)) * width

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for offset, (metric, label) in zip(offsets, metrics):
        values = df[metric].astype(float).to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=COLORS[metric],
            edgecolor="#1f2933",
            linewidth=0.6,
        )

    ax.set_ylim(0.60, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_ORDER)
    ax.set_xlabel("Masking ratio")
    ax.set_ylabel("Final test score")
    ax.grid(axis="y", color="#d7dee8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PICTURE_DIR / "research10_masking_ratio_score_bars.jpg", dpi=300)
    plt.close(fig)


def save_error_bars(df: pd.DataFrame) -> None:
    x = np.arange(len(RATIO_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(
        x - width / 2,
        df["false_negative"].astype(int).to_numpy(),
        width=width,
        label="FN",
        color=COLORS["false_negative"],
        edgecolor="#1f2933",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        df["false_positive"].astype(int).to_numpy(),
        width=width,
        label="FP",
        color=COLORS["false_positive"],
        edgecolor="#1f2933",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_ORDER)
    ax.set_xlabel("Masking ratio")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", color="#d7dee8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(PICTURE_DIR / "research10_masking_ratio_fn_fp_bars.jpg", dpi=300)
    plt.close(fig)


def save_ratio_size_bars(df: pd.DataFrame) -> None:
    x = np.arange(len(RATIO_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(
        x - width / 2,
        df["temporal_block_ratio"].astype(float).to_numpy(),
        width=width,
        label="Temporal ratio",
        color="#4f83bd",
        edgecolor="#1f2933",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        df["feature_group_ratio"].astype(float).to_numpy(),
        width=width,
        label="Feature ratio",
        color="#70ad47",
        edgecolor="#1f2933",
        linewidth=0.6,
    )
    ax.set_ylim(0, 0.70)
    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_ORDER)
    ax.set_xlabel("Masking ratio")
    ax.set_ylabel("Ratio")
    ax.grid(axis="y", color="#d7dee8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(PICTURE_DIR / "research10_masking_ratio_size_bars.jpg", dpi=300)
    plt.close(fig)


def main() -> None:
    df = ordered(pd.read_csv(RESULT_PATH))
    save_score_bars(df)
    save_error_bars(df)
    save_ratio_size_bars(df)
    print(PICTURE_DIR / "research10_masking_ratio_score_bars.jpg")
    print(PICTURE_DIR / "research10_masking_ratio_fn_fp_bars.jpg")
    print(PICTURE_DIR / "research10_masking_ratio_size_bars.jpg")


if __name__ == "__main__":
    main()
