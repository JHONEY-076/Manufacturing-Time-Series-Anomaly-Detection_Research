from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "GT-GAN": "#1F77B4",
    "Diffusion": "#17BECF",
    "Masking GT-GAN": "#FF7F0E",
    "Masking Diffusion": "#D62728",
    "Time warping": "#2CA02C",
    "Magnitude warping": "#9467BD",
    "Noise injection": "#E377C2",
    "Frequency domain": "#8C564B",
    "Comprehensive": "#BCBD22",
}

ORDER = [
    "GT-GAN",
    "Diffusion",
    "Masking GT-GAN",
    "Masking Diffusion",
    "Time warping",
    "Magnitude warping",
    "Noise injection",
    "Frequency domain",
    "Comprehensive",
]


def radial_distance(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - center, axis=1)


def main() -> None:
    result_dir = ROOT / "data" / "research02" / "results"
    figure_dir = ROOT / "data" / "research02" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

    coords = pd.read_csv(result_dir / "overall_tsne_real_generative_traditional.csv")
    original_points = coords.loc[coords["method"] == "Original", ["tsne_x", "tsne_y"]].to_numpy()
    original_center = original_points.mean(axis=0)
    original_spread = radial_distance(original_points, original_center).mean()

    rows = []
    for method in ORDER:
        method_points = coords.loc[coords["method"] == method, ["tsne_x", "tsne_y"]].to_numpy()
        distances = radial_distance(method_points, original_center)
        rows.append(
            {
                "method": method,
                "family": coords.loc[coords["method"] == method, "family"].iloc[0],
                "mean_radial_distance": distances.mean(),
                "median_radial_distance": np.median(distances),
                "radial_distance_ratio": distances.mean() / original_spread,
                "n": len(method_points),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(result_dir / "overall_tsne_radial_distance_ratio_summary.csv", index=False)

    plt.rcParams.update(
        {
            "axes.edgecolor": "#C7D0D9",
            "axes.linewidth": 1.0,
            "font.size": 9,
            "grid.color": "#E4EAF0",
            "grid.linewidth": 0.9,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    x = np.arange(len(summary))
    ratios = summary["radial_distance_ratio"].to_numpy()
    bars = ax.bar(
        x,
        ratios,
        width=0.42,
        color=[COLORS[method] for method in summary["method"]],
        edgecolor="#FFFFFF",
        linewidth=0.8,
        alpha=0.92,
    )
    ax.axhline(1.0, color="#263645", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(len(summary) - 0.15, 1.03, "Original spread = 1.0", ha="right", va="bottom", fontsize=8, weight="bold")

    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{ratio:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
            color="#17212B",
        )

    ax.set_title("Overall t-SNE Radial Distance from Original-Anomaly Center", fontsize=13, weight="bold", pad=10)
    ax.text(
        0.5,
        1.01,
        "Generated and transformation-augmented spread normalized by the Original anomaly spread",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#52616F",
    )
    ax.set_ylabel("Method radial distance / Original radial distance", fontsize=9, weight="bold", color="#34495E")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=28, ha="right", fontsize=8.5, weight="bold")
    ax.set_ylim(0, max(ratios.max() * 1.2, 1.35))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    caption = (
        "Ratios closer to 1 indicate that the generated or augmented samples have a radial spread similar "
        "to the Original anomaly windows in the shared t-SNE space."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8.5, color="#52616F")

    png_path = figure_dir / "overall_tsne_radial_distance_ratio_all_methods.png"
    svg_path = figure_dir / "overall_tsne_radial_distance_ratio_all_methods.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)
    print(result_dir / "overall_tsne_radial_distance_ratio_summary.csv")


if __name__ == "__main__":
    main()
