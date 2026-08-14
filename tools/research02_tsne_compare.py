from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


METHOD_FILES = {
    "GT-GAN": "gtgan_series_windows.npz",
    "Masking GT-GAN": "gtgan_masked_windows.npz",
    "Diffusion": "diffusion_series_windows.npz",
    "Masking Diffusion": "diffusion_masked_windows.npz",
}


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def sample_rows(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=n, replace=False)]


def feature_quality(real_x: np.ndarray, generated_x: np.ndarray) -> dict[str, float]:
    real_2d = real_x.reshape(-1, real_x.shape[-1])
    gen_2d = generated_x.reshape(-1, generated_x.shape[-1])
    real_corr = np.nan_to_num(np.corrcoef(real_2d.T))
    gen_corr = np.nan_to_num(np.corrcoef(gen_2d.T))
    return {
        "mean_abs_diff": float(np.mean(np.abs(real_2d.mean(axis=0) - gen_2d.mean(axis=0)))),
        "std_abs_diff": float(np.mean(np.abs(real_2d.std(axis=0) - gen_2d.std(axis=0)))),
        "corr_abs_diff": float(np.mean(np.abs(real_corr - gen_corr))),
    }


def proximity_metrics(real_flat: np.ndarray, generated_flat: np.ndarray) -> dict[str, float]:
    scaler = StandardScaler()
    combined = scaler.fit_transform(np.vstack([real_flat, generated_flat]))
    real_z = combined[: len(real_flat)]
    gen_z = combined[len(real_flat) :]

    gen_to_real = NearestNeighbors(n_neighbors=1).fit(real_z).kneighbors(gen_z)[0].ravel()
    real_to_gen = NearestNeighbors(n_neighbors=1).fit(gen_z).kneighbors(real_z)[0].ravel()
    centroid_dist = float(np.linalg.norm(real_z.mean(axis=0) - gen_z.mean(axis=0)))

    distances = pairwise_distances(real_z, real_z)
    nonzero = distances[distances > 0]
    scale = float(np.median(nonzero)) if len(nonzero) else 1.0
    close_threshold = scale * 0.25

    return {
        "gen_to_real_nn_mean": float(gen_to_real.mean()),
        "gen_to_real_nn_median": float(np.median(gen_to_real)),
        "real_to_gen_nn_mean": float(real_to_gen.mean()),
        "real_to_gen_nn_median": float(np.median(real_to_gen)),
        "centroid_distance": centroid_dist,
        "real_coverage_within_0.25_median_real_dist": float(np.mean(real_to_gen <= close_threshold)),
    }


def run_tsne(x: np.ndarray) -> np.ndarray:
    perplexity = min(30, max(5, (len(x) - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    ).fit_transform(StandardScaler().fit_transform(x))


def plot_pairwise_tsne(real_x: np.ndarray, generated_sets: dict[str, np.ndarray], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = axes.ravel()

    for i, (method, generated_x) in enumerate(generated_sets.items()):
        gen_sample = sample_rows(generated_x, len(real_x), RANDOM_STATE + i)
        x_all = np.vstack([flatten_windows(real_x), flatten_windows(gen_sample)])
        embedded = run_tsne(x_all)
        real_points = embedded[: len(real_x)]
        gen_points = embedded[len(real_x) :]

        ax = axes[i]
        ax.scatter(gen_points[:, 0], gen_points[:, 1], s=26, alpha=0.48, color="#FAA43A", label="Generated Data")
        ax.scatter(real_points[:, 0], real_points[:, 1], s=20, alpha=0.78, color="#5DA5DA", label="Real Data")
        ax.set_title(f"{method}\nReal n={len(real_x)}, Generated n={len(gen_sample)}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(False)

    plt.tight_layout()
    out_path = figure_dir / "tsne_2x2_pairwise_real_vs_generated.png"
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    root = Path.cwd()
    real_path = root / "data" / "augmentation_split" / "anomaly_train_seed_windows.npz"
    generated_dir = root / "data" / "research02" / "generated"
    result_dir = root / "data" / "research02" / "results"
    figure_dir = root / "data" / "research02" / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    real_x = np.load(real_path, allow_pickle=True)["X"].astype(np.float32)
    generated_sets = {
        method: np.load(generated_dir / filename, allow_pickle=True)["X"].astype(np.float32)
        for method, filename in METHOD_FILES.items()
    }

    rows = []
    real_flat = flatten_windows(real_x)
    for i, (method, generated_x) in enumerate(generated_sets.items()):
        gen_sample = sample_rows(generated_x, len(real_x), RANDOM_STATE + i)
        row = {"method": method, "n_real": len(real_x), "n_generated_used": len(gen_sample)}
        row.update(feature_quality(real_x, generated_x))
        row.update(proximity_metrics(real_flat, flatten_windows(gen_sample)))
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(result_dir / "tsne_distribution_comparison_metrics.csv", index=False)
    plot_pairwise_tsne(real_x, generated_sets, figure_dir)

    sort_cols = [
        "gen_to_real_nn_mean",
        "real_to_gen_nn_mean",
        "centroid_distance",
        "mean_abs_diff",
        "std_abs_diff",
        "corr_abs_diff",
    ]
    print(metrics.sort_values(sort_cols).to_string(index=False))
    print("saved: data/research02/figures/tsne_2x2_pairwise_real_vs_generated.png")
    print("saved: data/research02/results/tsne_distribution_comparison_metrics.csv")


if __name__ == "__main__":
    main()
