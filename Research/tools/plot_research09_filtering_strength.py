from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
RESULT_PATH = ROOT / "data" / "research09" / "results" / "research09_filtering_strength.csv"
PICTURE_DIR = ROOT / "pictures"
PICTURE_DIR.mkdir(parents=True, exist_ok=True)

STRENGTH_ORDER = ["loose", "default", "strict"]
COUNT_ORDER = [750, 1000]
COLORS = {
    750: "#2f6f9f",
    1000: "#d97941",
}


def save_score_bars(df: pd.DataFrame) -> None:
    metrics = [
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("f2", "F2-score"),
        ("auprc", "AUPRC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    x = np.arange(len(STRENGTH_ORDER))
    width = 0.24

    for ax, (metric, label) in zip(axes, metrics):
        for offset, count in zip([-width / 2, width / 2], COUNT_ORDER):
            values = [
                float(
                    df.loc[
                        (df["augmentation_count"] == count)
                        & (df["filtering_strength"] == strength),
                        metric,
                    ].iloc[0]
                )
                for strength in STRENGTH_ORDER
            ]
            ax.bar(
                x + offset,
                values,
                width=width,
                label=str(count),
                color=COLORS[count],
                edgecolor="#1f2933",
                linewidth=0.6,
            )
        ax.set_title(label, fontsize=13, pad=8)
        ax.set_ylim(0.55, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(STRENGTH_ORDER)
        ax.grid(axis="y", color="#d7dee8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Final test score")
    axes[2].set_ylabel("Final test score")
    axes[2].set_xlabel("Filtering strength")
    axes[3].set_xlabel("Filtering strength")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Generated windows",
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PICTURE_DIR / "research09_filtering_strength_score_bars.jpg", dpi=300)
    plt.close(fig)


def save_error_bars(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    x = np.arange(len(STRENGTH_ORDER))
    width = 0.24
    error_colors = {
        "false_negative": "#b91c1c",
        "false_positive": "#6b7280",
    }

    for ax, count in zip(axes, COUNT_ORDER):
        count_df = df[df["augmentation_count"] == count]
        for offset, metric, label in [
            (-width / 2, "false_negative", "FN"),
            (width / 2, "false_positive", "FP"),
        ]:
            values = [
                int(count_df.loc[count_df["filtering_strength"] == strength, metric].iloc[0])
                for strength in STRENGTH_ORDER
            ]
            ax.bar(
                x + offset,
                values,
                width=width,
                label=label,
                color=error_colors[metric],
                edgecolor="#1f2933",
                linewidth=0.6,
            )
        ax.set_title(f"{count} generated windows", fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(STRENGTH_ORDER)
        ax.set_xlabel("Filtering strength")
        ax.grid(axis="y", color="#d7dee8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Number of samples")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(PICTURE_DIR / "research09_filtering_strength_fn_fp_bars.jpg", dpi=300)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RESULT_PATH)
    save_score_bars(df)
    save_error_bars(df)
    print(PICTURE_DIR / "research09_filtering_strength_score_bars.jpg")
    print(PICTURE_DIR / "research09_filtering_strength_fn_fp_bars.jpg")


if __name__ == "__main__":
    main()
