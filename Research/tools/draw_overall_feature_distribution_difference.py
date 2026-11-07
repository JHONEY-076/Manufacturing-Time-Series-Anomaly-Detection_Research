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
from matplotlib.colors import LinearSegmentedColormap

from draw_overall_tsne_real_generative_traditional import (
    load_npz_x,
)


METHOD_ORDER = [
    "GT-GAN",
    "Diffusion",
    "Masking GT-GAN",
    "Masking Diffusion",
]
METRIC_COLUMNS = ["mean_abs_diff", "std_abs_diff", "corr_abs_diff"]
METRIC_LABELS = ["Mean", "Std", "Corr"]


def window_feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_2d = x.reshape(-1, x.shape[-1])
    mean = x_2d.mean(axis=0)
    std = x_2d.std(axis=0)
    corr = np.nan_to_num(np.corrcoef(x_2d.T))
    return mean, std, corr


def feature_distribution_metrics(real_x: np.ndarray, generated_x: np.ndarray, method: str) -> dict[str, float | int | str]:
    real_mean, real_std, real_corr = window_feature_stats(real_x)
    gen_mean, gen_std, gen_corr = window_feature_stats(generated_x)
    return {
        "method": method,
        "n_real": len(real_x),
        "n_generated": len(generated_x),
        "mean_abs_diff": float(np.mean(np.abs(real_mean - gen_mean))),
        "std_abs_diff": float(np.mean(np.abs(real_std - gen_std))),
        "corr_abs_diff": float(np.mean(np.abs(real_corr - gen_corr))),
    }


def main() -> None:
    split_dir = ROOT / "data" / "augmentation_split"
    generated_dir = ROOT / "data" / "research02" / "generated"
    figure_dir = ROOT / "data" / "research02" / "figures"
    result_dir = ROOT / "data" / "research02" / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

    real_x = load_npz_x(split_dir / "anomaly_train_seed_windows.npz")
    generated_sets = {
        "GT-GAN": load_npz_x(generated_dir / "gtgan_series_windows.npz"),
        "Diffusion": load_npz_x(generated_dir / "diffusion_series_windows.npz"),
        "Masking GT-GAN": load_npz_x(generated_dir / "gtgan_masked_windows.npz"),
        "Masking Diffusion": load_npz_x(generated_dir / "diffusion_masked_windows.npz"),
    }

    summary = pd.DataFrame(
        [feature_distribution_metrics(real_x, generated_sets[method], method) for method in METHOD_ORDER]
    )
    summary.to_csv(result_dir / "overall_feature_distribution_difference_summary.csv", index=False)

    values = summary.set_index("method").loc[METHOD_ORDER, METRIC_COLUMNS].to_numpy()
    cmap = LinearSegmentedColormap.from_list("difference", ["#D8F3E8", "#F4D8C7", "#FF4B35"])

    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0, vmax=max(values.max(), 1e-9))

    ax.set_title("Feature-Level Distribution Difference", fontsize=15, weight="bold", pad=24)
    ax.text(
        0.5,
        1.035,
        "Absolute difference between original anomaly windows and each method",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#52616F",
    )
    ax.set_xticks(np.arange(len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=10, weight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=9.5, weight="bold")
    ax.tick_params(length=0)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}".rstrip("0").rstrip("."),
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#22313F",
            )

    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.035)
    colorbar.set_label("Absolute difference", fontsize=8.5, color="#52616F")
    colorbar.ax.tick_params(labelsize=8)

    fig.text(
        0.5,
        -0.025,
        "Lighter cells are closer to the original anomaly distribution.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#52616F",
    )

    png_path = figure_dir / "overall_feature_distribution_difference_heatmap.png"
    svg_path = figure_dir / "overall_feature_distribution_difference_heatmap.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)
    print(result_dir / "overall_feature_distribution_difference_summary.csv")


if __name__ == "__main__":
    main()
