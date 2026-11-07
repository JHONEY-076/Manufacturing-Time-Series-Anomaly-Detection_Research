from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
FEATURE_NAMES = ["RealPower", "GateOnTime", "SetPower", "Speed", "Length"]

METHOD_FILES = {
    "GT-GAN": "gtgan_series_windows.npz",
    "Masking GT-GAN": "gtgan_masked_windows.npz",
    "Diffusion": "diffusion_series_windows.npz",
    "Masking Diffusion": "diffusion_masked_windows.npz",
}

COLORS = {
    "Real": "#4C78A8",
    "GT-GAN": "#F58518",
    "Masking GT-GAN": "#54A24B",
    "Diffusion": "#E45756",
    "Masking Diffusion": "#72B7B2",
}


def load_npz_x(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)["X"].astype(np.float32)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def sample_rows(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=n, replace=False)]


def plot_shared_pca(real_x: np.ndarray, generated_sets: dict[str, np.ndarray], figure_dir: Path) -> None:
    sampled = {
        method: sample_rows(x, len(real_x), RANDOM_STATE + i)
        for i, (method, x) in enumerate(generated_sets.items())
    }
    labels = ["Real"] * len(real_x)
    matrices = [flatten_windows(real_x)]
    for method, x in sampled.items():
        labels.extend([method] * len(x))
        matrices.append(flatten_windows(x))

    x_all = StandardScaler().fit_transform(np.vstack(matrices))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(x_all)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for label in ["Real", *sampled.keys()]:
        pts = coords[labels == label]
        size = 22 if label == "Real" else 18
        alpha = 0.82 if label == "Real" else 0.48
        ax.scatter(pts[:, 0], pts[:, 1], s=size, alpha=alpha, label=label, color=COLORS[label])

    ax.set_title("Shared PCA Distribution of Real and Generated Anomaly Windows", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(figure_dir / "pca_shared_real_vs_generated.png", dpi=220)
    plt.close(fig)


def nearest_neighbor_rows(real_x: np.ndarray, generated_sets: dict[str, np.ndarray]) -> pd.DataFrame:
    real_flat = flatten_windows(real_x)
    rows = []
    for i, (method, x) in enumerate(generated_sets.items()):
        gen = sample_rows(x, len(real_x), RANDOM_STATE + i)
        combined = StandardScaler().fit_transform(np.vstack([real_flat, flatten_windows(gen)]))
        real_z = combined[: len(real_flat)]
        gen_z = combined[len(real_flat) :]
        dist = NearestNeighbors(n_neighbors=1).fit(real_z).kneighbors(gen_z)[0].ravel()
        rows.extend({"method": method, "gen_to_real_nn_distance": float(value)} for value in dist)
    return pd.DataFrame(rows)


def plot_nn_boxplot(nn_df: pd.DataFrame, figure_dir: Path) -> None:
    methods = list(METHOD_FILES)
    data = [nn_df.loc[nn_df["method"] == method, "gen_to_real_nn_distance"].to_numpy() for method in methods]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    box = ax.boxplot(data, labels=methods, patch_artist=True, showfliers=False)
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.55)
    ax.set_title("Generated-to-Real Nearest Neighbor Distance", fontweight="bold")
    ax.set_ylabel("Distance in standardized flattened-window space")
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", rotation=12)
    plt.tight_layout()
    plt.savefig(figure_dir / "generated_to_real_nn_distance_boxplot.png", dpi=220)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, figure_dir: Path) -> None:
    metrics = metrics.copy()
    metrics["coverage_pct"] = metrics["real_coverage_within_0.25_median_real_dist"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    specs = [
        ("centroid_distance", "Centroid Distance", "lower is better"),
        ("gen_to_real_nn_mean", "Generated-to-Real NN Mean", "lower is better"),
        ("coverage_pct", "Real Coverage (%)", "higher is better"),
    ]
    for ax, (col, title, subtitle) in zip(axes, specs):
        values = metrics.set_index("method").loc[list(METHOD_FILES), col]
        ax.bar(values.index, values.values, color=[COLORS[m] for m in values.index], alpha=0.78)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(figure_dir / "distribution_alignment_metric_bars.png", dpi=220)
    plt.close(fig)


def plot_feature_mean_heatmap(real_x: np.ndarray, generated_sets: dict[str, np.ndarray], figure_dir: Path) -> None:
    real_2d = real_x.reshape(-1, real_x.shape[-1])
    real_mean = real_2d.mean(axis=0)
    rows = []
    for method, x in generated_sets.items():
        gen_mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        rows.append(gen_mean - real_mean)

    values = np.vstack(rows)
    limit = max(abs(values.min()), abs(values.max()))

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(len(METHOD_FILES)))
    ax.set_yticklabels(list(METHOD_FILES))
    ax.set_title("Generated - Real Feature Mean Difference", fontweight="bold")
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(c, r, f"{values[r, c]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82)
    plt.tight_layout()
    plt.savefig(figure_dir / "feature_mean_difference_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path.cwd()
    real_x = load_npz_x(root / "data" / "augmentation_split" / "anomaly_train_seed_windows.npz")
    generated_dir = root / "data" / "research02" / "generated"
    figure_dir = root / "data" / "research02" / "figures"
    result_dir = root / "data" / "research02" / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    generated_sets = {
        method: load_npz_x(generated_dir / filename)
        for method, filename in METHOD_FILES.items()
    }

    metrics = pd.read_csv(result_dir / "tsne_distribution_comparison_metrics.csv")
    plot_shared_pca(real_x, generated_sets, figure_dir)

    nn_df = nearest_neighbor_rows(real_x, generated_sets)
    nn_df.to_csv(result_dir / "generated_to_real_nn_distances.csv", index=False)
    plot_nn_boxplot(nn_df, figure_dir)

    plot_metric_bars(metrics, figure_dir)
    plot_feature_mean_heatmap(real_x, generated_sets, figure_dir)

    print("saved: data/research02/figures/pca_shared_real_vs_generated.png")
    print("saved: data/research02/figures/generated_to_real_nn_distance_boxplot.png")
    print("saved: data/research02/figures/distribution_alignment_metric_bars.png")
    print("saved: data/research02/figures/feature_mean_difference_heatmap.png")
    print("saved: data/research02/results/generated_to_real_nn_distances.csv")


if __name__ == "__main__":
    main()
