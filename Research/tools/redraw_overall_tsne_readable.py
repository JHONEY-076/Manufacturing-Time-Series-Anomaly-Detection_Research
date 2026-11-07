"""Create a large, readable overall t-SNE plot from the actual window arrays."""

from pathlib import Path
import os

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from draw_overall_tsne_real_generative_traditional import (
    RANDOM_STATE,
    SAMPLE_PER_GROUP,
    comprehensive_augmentation,
    flatten_windows,
    frequency_domain,
    load_npz_x,
    magnitude_warp,
    noise_injection,
    sample_rows,
    time_warp,
)


def main() -> None:
    split_dir = ROOT / "data" / "augmentation_split"
    generated_dir = ROOT / "data" / "research02" / "generated"
    result_dir = ROOT / "data" / "research02" / "results"
    output_dir = ROOT / "pictures" / "overall_tsne_readable"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

    normal = load_npz_x(split_dir / "normal_train_windows.npz")
    anomaly = load_npz_x(split_dir / "anomaly_train_seed_windows.npz")

    groups: list[tuple[str, str, np.ndarray]] = [
        ("Original normal", "Original", sample_rows(normal, SAMPLE_PER_GROUP, RANDOM_STATE)),
        ("Original anomaly", "Original", sample_rows(anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 1)),
        (
            "GT-GAN",
            "Generative AI",
            sample_rows(load_npz_x(generated_dir / "gtgan_series_windows.npz"), SAMPLE_PER_GROUP, RANDOM_STATE + 2),
        ),
        (
            "Diffusion",
            "Generative AI",
            sample_rows(load_npz_x(generated_dir / "diffusion_series_windows.npz"), SAMPLE_PER_GROUP, RANDOM_STATE + 3),
        ),
        (
            "Masking GT-GAN",
            "Generative AI",
            sample_rows(load_npz_x(generated_dir / "gtgan_masked_windows.npz"), SAMPLE_PER_GROUP, RANDOM_STATE + 4),
        ),
        (
            "Masking Diffusion",
            "Generative AI",
            sample_rows(load_npz_x(generated_dir / "diffusion_masked_windows.npz"), SAMPLE_PER_GROUP, RANDOM_STATE + 5),
        ),
        ("Time warping", "Transformation", time_warp(anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 101)),
        ("Magnitude warping", "Transformation", magnitude_warp(anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 102)),
        ("Noise injection", "Transformation", noise_injection(anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 103)),
        ("Frequency domain", "Transformation", frequency_domain(anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 104)),
        (
            "Comprehensive",
            "Transformation",
            comprehensive_augmentation(anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 105),
        ),
    ]

    if any(len(values) != SAMPLE_PER_GROUP for _, _, values in groups):
        raise ValueError("Every displayed group must contain exactly 247 windows")

    flattened = np.vstack([flatten_windows(values) for _, _, values in groups])
    methods = np.concatenate([[name] * len(values) for name, _, values in groups])
    families = np.concatenate([[family] * len(values) for _, family, values in groups])

    scaled = StandardScaler().fit_transform(flattened)
    embedded = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    ).fit_transform(scaled)

    coords = pd.DataFrame(
        {
            "tsne_x": embedded[:, 0],
            "tsne_y": embedded[:, 1],
            "method": methods,
            "family": families,
        }
    )
    coords_path = result_dir / "overall_tsne_readable_coordinates.csv"
    coords.to_csv(coords_path, index=False)

    colors = {
        "Original normal": "#B8B8B8",
        "Original anomaly": "#111111",
        "GT-GAN": "#1F77B4",
        "Diffusion": "#17BECF",
        "Masking GT-GAN": "#FF7F0E",
        "Masking Diffusion": "#D62728",
        "Time warping": "#2CA02C",
        "Magnitude warping": "#9467BD",
        "Noise injection": "#E377C2",
        "Frequency domain": "#8C564B",
        "Comprehensive": "#A8AD00",
    }
    marker_by_family = {"Original": "o", "Generative AI": "^", "Transformation": "s"}
    sizes = {"Original normal": 78, "Original anomaly": 118}

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.edgecolor": "#AEB8C2",
            "axes.linewidth": 1.5,
            "grid.color": "#DDE3E9",
            "grid.linewidth": 1.1,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        }
    )
    fig, ax = plt.subplots(figsize=(16.5, 12.5))

    plot_order = [
        "Original normal",
        "Time warping",
        "Magnitude warping",
        "Noise injection",
        "Frequency domain",
        "Comprehensive",
        "GT-GAN",
        "Diffusion",
        "Masking GT-GAN",
        "Masking Diffusion",
        "Original anomaly",
    ]
    for name in plot_order:
        subset = coords[coords["method"] == name]
        family = subset["family"].iloc[0]
        is_anomaly = name == "Original anomaly"
        is_normal = name == "Original normal"
        ax.scatter(
            subset["tsne_x"],
            subset["tsne_y"],
            s=sizes.get(name, 96 if family == "Generative AI" else 82),
            marker=marker_by_family[family],
            color=colors[name],
            alpha=0.62 if is_normal else (0.97 if is_anomaly else 0.86),
            edgecolors="white",
            linewidths=1.1 if is_anomaly else 0.75,
            label=f"{name} ({family})",
            zorder=30 if is_anomaly else (3 if is_normal else 10),
        )

    x_min, x_max = coords["tsne_x"].min(), coords["tsne_x"].max()
    y_min, y_max = coords["tsne_y"].min(), coords["tsne_y"].max()
    x_pad = max((x_max - x_min) * 0.045, 1.0)
    y_pad = max((y_max - y_min) * 0.045, 1.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.55)
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("t-SNE 1", fontsize=18, fontweight="medium", labelpad=10)
    ax.set_ylabel("t-SNE 2", fontsize=18, fontweight="medium", labelpad=10)
    ax.set_title(
        "Overall t-SNE: Original, Generative AI, and Transformation-Augmented Windows",
        fontsize=24,
        fontweight="bold",
        pad=18,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.072),
        ncol=3,
        frameon=True,
        fontsize=13,
        title=f"Samples per group: {SAMPLE_PER_GROUP}",
        title_fontsize=14,
        markerscale=1.25,
        columnspacing=1.6,
        handletextpad=0.7,
        borderpad=0.9,
        labelspacing=0.7,
    )
    fig.text(
        0.5,
        0.022,
        "All groups share one t-SNE space after standardizing flattened windows. "
        "Transformation-based augmentations use the original anomaly windows as their source.",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#4D5863",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.29)

    png_path = output_dir / "overall_tsne_readable.png"
    svg_path = output_dir / "overall_tsne_readable.svg"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    print(png_path)
    print(svg_path)
    print(coords_path)


if __name__ == "__main__":
    main()
