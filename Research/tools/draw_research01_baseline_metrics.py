from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("precision", "Precision", "#4C78A8"),
    ("recall", "Recall", "#F58518"),
    ("f1", "F1-score", "#54A24B"),
    ("auroc", "AUROC", "#72B7B2"),
    ("auprc", "AUPRC", "#E45756"),
]


def main() -> None:
    root = Path.cwd()
    summary_path = root / "data" / "research01" / "results" / "research01_randomforest_baseline_summary.json"
    out_dir = root / "data" / "research01" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    result = summary["result"]
    labels = [label for _, label, _ in METRICS]
    values = [float(result[key]) for key, _, _ in METRICS]
    colors = [color for _, _, color in METRICS]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=10)

    ax.set_title("Original Anomaly Seed Baseline Performance", fontsize=15, fontweight="bold")
    ax.set_ylabel("Final test score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    note = (
        f"RandomForest, normal train={result['normal_train_used']:,}, "
        f"real anomaly seed={result['real_anomaly_seed_used']:,}, "
        f"threshold={result['threshold_selected_on_real_validation']:.2f}"
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    png_path = out_dir / "original_seed_baseline_metrics.png"
    svg_path = out_dir / "original_seed_baseline_metrics.svg"
    fig.savefig(png_path, dpi=240)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
