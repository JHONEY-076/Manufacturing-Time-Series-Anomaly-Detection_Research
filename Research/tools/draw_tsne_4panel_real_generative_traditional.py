from __future__ import annotations

from pathlib import Path
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from draw_overall_tsne_real_generative_traditional import (
    SAMPLE_PER_GROUP,
    RANDOM_STATE,
    comprehensive_augmentation,
    flatten_windows,
    frequency_domain,
    load_npz_x,
    magnitude_warp,
    noise_injection,
    sample_rows,
    time_warp,
)


def embed_groups(groups: list[tuple[str, np.ndarray]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    x_all = np.vstack([flatten_windows(x) for _, x in groups])
    labels = np.concatenate([[name] * len(x) for name, x in groups])
    x_scaled = StandardScaler().fit_transform(x_all)
    perplexity = min(35, max(5, (len(x_scaled) - 1) // 3))
    embedded = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    ).fit_transform(x_scaled)
    return embedded, labels


def draw_panel(
    ax: plt.Axes,
    title: str,
    groups: list[tuple[str, np.ndarray]],
    embedded: np.ndarray,
    labels: np.ndarray,
    colors: dict[str, str],
    markers: dict[str, str],
    axis_limits: tuple[float, float, float, float],
) -> None:
    for name, _ in groups:
        points = embedded[labels == name]
        is_original = name == "Original"
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=118 if is_original else 88,
            marker=markers.get(name, "o"),
            color=colors[name],
            alpha=0.92 if not is_original else 0.72,
            edgecolors="white",
            linewidths=1.05,
            label=name,
        )
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    x_min, x_max, y_min, y_max = axis_limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8, frameon=True, loc="best", markerscale=1.25)


def main() -> None:
    split_dir = ROOT / "data" / "augmentation_split"
    generated_dir = ROOT / "data" / "research02" / "generated"
    figure_dir = ROOT / "data" / "research02" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

    real_anomaly = load_npz_x(split_dir / "anomaly_train_seed_windows.npz")

    original = sample_rows(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 1)

    gtgan = sample_rows(load_npz_x(generated_dir / "gtgan_series_windows.npz"), SAMPLE_PER_GROUP, RANDOM_STATE + 2)
    diffusion = sample_rows(
        load_npz_x(generated_dir / "diffusion_series_windows.npz"), SAMPLE_PER_GROUP, RANDOM_STATE + 3
    )
    masking_gtgan = sample_rows(
        load_npz_x(generated_dir / "gtgan_masked_windows.npz"), SAMPLE_PER_GROUP, RANDOM_STATE + 4
    )
    masking_diffusion = sample_rows(
        load_npz_x(generated_dir / "diffusion_masked_windows.npz"), SAMPLE_PER_GROUP, RANDOM_STATE + 5
    )

    time = time_warp(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 101)
    magnitude = magnitude_warp(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 102)
    noise = noise_injection(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 103)
    frequency = frequency_domain(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 104)
    comprehensive = comprehensive_augmentation(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 105)

    colors = {
        "Original": "#FFD400",
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
    markers = {
        "Original": "o",
        "GT-GAN": "^",
        "Diffusion": "^",
        "Masking GT-GAN": "^",
        "Masking Diffusion": "^",
        "Time warping": "s",
        "Magnitude warping": "s",
        "Noise injection": "s",
        "Frequency domain": "s",
        "Comprehensive": "s",
    }

    panel_groups = [
        (
            "(a) Plain generative models",
            [
                ("Original", original),
                ("GT-GAN", gtgan),
                ("Diffusion", diffusion),
            ],
        ),
        (
            "(b) Masking-based generative models",
            [
                ("Original", original),
                ("Masking GT-GAN", masking_gtgan),
                ("Masking Diffusion", masking_diffusion),
            ],
        ),
        (
            "(c) Transformation-based augmentations",
            [
                ("Original", original),
                ("Time warping", time),
                ("Magnitude warping", magnitude),
                ("Noise injection", noise),
                ("Frequency domain", frequency),
            ],
        ),
        (
            "(d) Representative augmentation comparison",
            [
                ("Original", original),
                ("Masking Diffusion", masking_diffusion),
                ("Magnitude warping", magnitude),
                ("Frequency domain", frequency),
                ("Comprehensive", comprehensive),
            ],
        ),
    ]

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#BDBDBD",
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    axes = axes.ravel()
    global_group_map: dict[str, np.ndarray] = {}
    for _title, groups in panel_groups:
        for name, values in groups:
            global_group_map.setdefault(name, values)

    global_groups = list(global_group_map.items())
    embedded, labels = embed_groups(global_groups, RANDOM_STATE)

    x_all = embedded[:, 0]
    y_all = embedded[:, 1]
    x_pad = max((x_all.max() - x_all.min()) * 0.05, 1.0)
    y_pad = max((y_all.max() - y_all.min()) * 0.05, 1.0)
    axis_limits = (
        float(x_all.min() - x_pad),
        float(x_all.max() + x_pad),
        float(y_all.min() - y_pad),
        float(y_all.max() + y_pad),
    )

    for idx, (title, groups) in enumerate(panel_groups):
        draw_panel(axes[idx], title, groups, embedded, labels, colors, markers, axis_limits)

    fig.suptitle(
        "t-SNE Distribution Comparison by Augmentation Type",
        fontsize=17,
        weight="bold",
    )
    caption = (
        "All panels use coordinates from one shared t-SNE embedding of all compared groups. "
        f"All groups are sampled to n={SAMPLE_PER_GROUP}. "
        "Original denotes real anomaly seed windows. "
        "Transformation methods are generated from original anomaly seed windows."
    )
    fig.text(0.01, -0.015, caption, ha="left", va="top", fontsize=10, color="#555555")

    png_path = figure_dir / "tsne_4panel_real_generative_traditional.png"
    svg_path = figure_dir / "tsne_4panel_real_generative_traditional.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
