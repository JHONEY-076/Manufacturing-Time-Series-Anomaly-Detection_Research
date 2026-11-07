from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["GT-GAN", "Diffusion", "Masking GT-GAN", "Masking Diffusion"]
METRICS = ["precision", "recall", "f1", "auprc"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "auprc": "AUPRC",
}
COLORS = {
    "precision": "#4C78A8",
    "recall": "#F58518",
    "f1": "#54A24B",
    "auprc": "#E45756",
}


def format_axes(ax: plt.Axes, title: str, baseline: pd.Series | None = None) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Generated anomaly count")
    ax.set_ylabel("Final test score")
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="y", alpha=0.25)
    if baseline is not None:
        ax.axhline(float(baseline["f1"]), color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            0.99,
            float(baseline["f1"]) + 0.005,
            "Original RF F1",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def plot_combined(df: pd.DataFrame, baseline: pd.Series, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharey=True)
    axes = axes.ravel()

    for ax, method in zip(axes, METHODS):
        subset = df[df["method"] == method].sort_values("generated_anomaly_used")
        counts = subset["generated_anomaly_used"].astype(int).to_numpy()
        x = np.arange(len(counts))
        width = 0.19
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(METRICS))

        for metric, offset in zip(METRICS, offsets):
            ax.bar(
                x + offset,
                subset[metric].to_numpy(),
                width=width,
                color=COLORS[metric],
                label=METRIC_LABELS[metric],
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in counts])
        format_axes(ax, method, baseline)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Model Performance by Generated Anomaly Count", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_dir / "model_bar_metrics_by_generated_count.png", dpi=220)
    plt.close(fig)


def plot_individual(df: pd.DataFrame, baseline: pd.Series, out_dir: Path) -> None:
    individual_dir = out_dir / "method_by_method"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        subset = df[df["method"] == method].sort_values("generated_anomaly_used")
        counts = subset["generated_anomaly_used"].astype(int).to_numpy()
        x = np.arange(len(counts))
        width = 0.19
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(METRICS))

        fig, ax = plt.subplots(figsize=(9, 5.2))
        for metric, offset in zip(METRICS, offsets):
            bars = ax.bar(
                x + offset,
                subset[metric].to_numpy(),
                width=width,
                color=COLORS[metric],
                label=METRIC_LABELS[metric],
                alpha=0.9,
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in counts])
        format_axes(ax, f"{method} Performance by Generated Count", baseline)
        ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
        fig.tight_layout()
        filename = method.lower().replace(" ", "_").replace("-", "") + "_bar_metrics_by_generated_count.png"
        fig.savefig(individual_dir / filename, dpi=220)
        plt.close(fig)


def main() -> None:
    root = Path.cwd()
    result_path = root / "data" / "research03" / "results" / "supervised_model_augmented_performance.csv"
    out_dir = root / "data" / "research03" / "figures"

    df = pd.read_csv(result_path)
    baseline = df[df["method"] == "Real anomaly only"].iloc[0]
    generated_df = df[df["method"].isin(METHODS)].copy()

    plot_combined(generated_df, baseline, out_dir)
    plot_individual(generated_df, baseline, out_dir)

    print("saved: data/research03/figures/model_bar_metrics_by_generated_count.png")
    print("saved: data/research03/figures/method_by_method/*_bar_metrics_by_generated_count.png")


if __name__ == "__main__":
    main()
