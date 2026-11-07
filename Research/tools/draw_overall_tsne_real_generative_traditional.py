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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
SAMPLE_PER_GROUP = 247


def ensure_window_3d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        return x.astype(np.float32)
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0, :, :].astype(np.float32)
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[:, :, :, 0].astype(np.float32)
    squeezed = np.squeeze(x)
    if squeezed.ndim == 3:
        return squeezed.astype(np.float32)
    raise ValueError(f"Expected 3D window data, got shape {x.shape}")


def load_npz_x(path: Path) -> np.ndarray:
    return ensure_window_3d(np.load(path, allow_pickle=True)["X"])


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def sample_rows(x: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    if len(x) <= n_sample:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=n_sample, replace=False)]


def smooth_random_curve(length: int, n_features: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_knots = 4
    knot_x = np.linspace(0, length - 1, n_knots)
    curves = []
    for _ in range(n_features):
        knot_y = rng.normal(loc=1.0, scale=sigma, size=n_knots)
        curves.append(np.interp(np.arange(length), knot_x, knot_y))
    return np.stack(curves, axis=1).astype(np.float32)


def time_warp(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    length = x.shape[1]
    base_time = np.arange(length)
    for _ in range(n_generate):
        source = x[rng.integers(0, len(x))]
        increments = rng.lognormal(mean=0.0, sigma=0.18, size=length)
        warped_time = np.cumsum(increments)
        warped_time = (warped_time - warped_time[0]) / (warped_time[-1] - warped_time[0])
        warped_time = warped_time * (length - 1)
        generated = np.empty_like(source)
        for feature_idx in range(source.shape[1]):
            generated[:, feature_idx] = np.interp(base_time, warped_time, source[:, feature_idx])
        out.append(generated)
    return np.stack(out).astype(np.float32)


def magnitude_warp(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_generate):
        source = x[rng.integers(0, len(x))]
        curve = smooth_random_curve(source.shape[0], source.shape[1], sigma=0.12, seed=seed + i)
        out.append(source * curve)
    return np.stack(out).astype(np.float32)


def noise_injection(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feature_std = x.reshape(-1, x.shape[-1]).std(axis=0).reshape(1, 1, -1)
    out = []
    for _ in range(n_generate):
        source = x[rng.integers(0, len(x))]
        noise = rng.normal(loc=0.0, scale=0.04, size=source.shape) * feature_std
        out.append(source + noise)
    return np.stack(out).astype(np.float32)


def frequency_domain(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    length = x.shape[1]
    for _ in range(n_generate):
        source = x[rng.integers(0, len(x))]
        spectrum = np.fft.rfft(source, axis=0)
        amplitude_scale = rng.normal(loc=1.0, scale=0.08, size=spectrum.shape)
        phase_shift = rng.normal(loc=0.0, scale=0.08, size=spectrum.shape)
        phase_shift[0, :] = 0.0
        if length % 2 == 0:
            phase_shift[-1, :] = 0.0
        perturbed = spectrum * amplitude_scale * np.exp(1j * phase_shift)
        out.append(np.fft.irfft(perturbed, n=length, axis=0))
    return np.stack(out).astype(np.float32)


def comprehensive_augmentation(x: np.ndarray, n_generate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    length = x.shape[1]
    feature_std = x.reshape(-1, x.shape[-1]).std(axis=0).reshape(1, -1)
    base_time = np.arange(length)
    for i in range(n_generate):
        source = x[rng.integers(0, len(x))]
        increments = rng.lognormal(mean=0.0, sigma=0.12, size=length)
        warped_time = np.cumsum(increments)
        warped_time = (warped_time - warped_time[0]) / (warped_time[-1] - warped_time[0])
        warped_time = warped_time * (length - 1)
        generated = np.empty_like(source)
        for feature_idx in range(source.shape[1]):
            generated[:, feature_idx] = np.interp(base_time, warped_time, source[:, feature_idx])
        generated = generated * smooth_random_curve(length, source.shape[1], sigma=0.08, seed=seed + i)
        spectrum = np.fft.rfft(generated, axis=0)
        amplitude_scale = rng.normal(loc=1.0, scale=0.04, size=spectrum.shape)
        phase_shift = rng.normal(loc=0.0, scale=0.04, size=spectrum.shape)
        phase_shift[0, :] = 0.0
        if length % 2 == 0:
            phase_shift[-1, :] = 0.0
        generated = np.fft.irfft(spectrum * amplitude_scale * np.exp(1j * phase_shift), n=length, axis=0)
        noise = rng.normal(loc=0.0, scale=0.02, size=generated.shape) * feature_std
        out.append(generated + noise)
    return np.stack(out).astype(np.float32)


def main() -> None:
    split_dir = ROOT / "data" / "augmentation_split"
    generated_dir = ROOT / "data" / "research02" / "generated"
    figure_dir = ROOT / "data" / "research02" / "figures"
    result_dir = ROOT / "data" / "research02" / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

    real_anomaly = load_npz_x(split_dir / "anomaly_train_seed_windows.npz")

    groups: list[tuple[str, str, np.ndarray]] = [
        ("Original", "Original", sample_rows(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 1)),
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
        ("Time warping", "Transformation", time_warp(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 101)),
        ("Magnitude warping", "Transformation", magnitude_warp(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 102)),
        ("Noise injection", "Transformation", noise_injection(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 103)),
        ("Frequency domain", "Transformation", frequency_domain(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 104)),
        (
            "Comprehensive",
            "Transformation",
            comprehensive_augmentation(real_anomaly, SAMPLE_PER_GROUP, RANDOM_STATE + 105),
        ),
    ]

    x_all = np.vstack([flatten_windows(x) for _, _, x in groups])
    labels = np.concatenate([[name] * len(x) for name, _, x in groups])
    families = np.concatenate([[family] * len(x) for _, family, x in groups])

    x_scaled = StandardScaler().fit_transform(x_all)
    perplexity = min(40, max(5, (len(x_scaled) - 1) // 3))
    embedded = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    ).fit_transform(x_scaled)

    coords = pd.DataFrame(
        {
            "tsne_x": embedded[:, 0],
            "tsne_y": embedded[:, 1],
            "method": labels,
            "family": families,
        }
    )
    coords.to_csv(result_dir / "overall_tsne_real_generative_traditional.csv", index=False)

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
        "Original": "s",
        "Generative AI": "^",
        "Transformation": "s",
    }
    sizes = {
        "Original": 155,
    }

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#BDBDBD",
            "font.size": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(15, 9), constrained_layout=True)

    plot_order = [name for name, _, _ in groups if name != "Original"] + ["Original"]
    for name in plot_order:
        subset = coords[coords["method"] == name]
        family = subset["family"].iloc[0]
        ax.scatter(
            subset["tsne_x"],
            subset["tsne_y"],
            s=sizes.get(name, 82),
            marker=markers[family],
            color=colors[name],
            alpha=0.98 if name == "Original" else 0.82,
            edgecolors="#111111" if name == "Original" else "#FFFFFF",
            linewidths=1.8 if name == "Original" else 0.9,
            label=f"{name} ({family})",
            zorder=20 if name == "Original" else 3,
        )

    ax.set_title(
        "Overall t-SNE: Original, Generative AI, and Transformation-Augmented Windows",
        fontsize=15,
        weight="bold",
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.45)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=10,
        title=f"Samples per group: {SAMPLE_PER_GROUP}",
        title_fontsize=10,
        markerscale=1.35,
    )

    caption = (
        "All groups are embedded in one shared t-SNE space after standardizing flattened windows. "
        "Original denotes real anomaly seed windows. "
        "Transformation-based augmentations are generated from the original anomaly seed windows."
    )
    fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=9, color="#555555")

    png_path = figure_dir / "overall_tsne_real_generative_traditional.png"
    svg_path = figure_dir / "overall_tsne_real_generative_traditional.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)
    print(result_dir / "overall_tsne_real_generative_traditional.csv")


if __name__ == "__main__":
    main()
