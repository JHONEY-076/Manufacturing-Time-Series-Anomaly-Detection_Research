"""Compare saved generation pools at an equal evaluation count; no retraining."""

from pathlib import Path
import json
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

from research05_balanced_aug_experiment import quality_filter_generated_anomalies


FILES = {
    "GT-GAN": "gtgan_series_windows.npz",
    "Diffusion": "diffusion_series_windows.npz",
    "Masking GT-GAN": "gtgan_masked_windows.npz",
    "Masking Diffusion": "diffusion_masked_windows.npz",
}
COUNT = 750
SEED = 42
MAX_LAG = 10


def statistics(x):
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(axis=0), flat.std(axis=0), np.corrcoef(flat.T)


def pooled_window_acf(x):
    """Biased ACF pooled over demeaned windows; never join window boundaries.

    rho_f(k) = sum_i sum_t u_itf u_i,t+k,f / sum_i sum_t u_itf**2,
    where u_itf is centered on the mean of window i, feature f.
    """
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=1, keepdims=True)
    denominator = np.square(centered).sum(axis=(0, 1))
    if np.any(denominator <= 1e-12):
        raise ValueError("ACF is undefined for a feature without within-window variation")
    return np.stack([
        (centered[:, :-lag] * centered[:, lag:]).sum(axis=(0, 1)) / denominator
        for lag in range(1, MAX_LAG + 1)
    ])


def main():
    split = ROOT / "data/augmentation_split"
    generated = ROOT / "data/research02/generated"
    with np.load(split / "anomaly_train_seed_windows.npz", allow_pickle=True) as data:
        real = data["X"].astype(np.float32)
        features = data["features"].tolist()
    with np.load(split / "normal_train_windows.npz", allow_pickle=True) as data:
        normal = data["X"].astype(np.float32)
    pools = {}
    for method, filename in FILES.items():
        with np.load(generated / filename, allow_pickle=True) as data:
            assert data["features"].tolist() == features
            pools[method] = data["X"].astype(np.float32)
    assert all(x.shape == (1000, 30, 5) for x in pools.values())
    assert real.shape == (247, 30, 5)

    indices = np.random.default_rng(SEED).choice(1000, COUNT, replace=False)
    evaluated = {method: x[indices] for method, x in pools.items()}
    evaluated["FMD (default filter)"] = quality_filter_generated_anomalies(
        pools["Masking Diffusion"], real, normal, COUNT
    )
    ref_mean, ref_std, ref_corr = statistics(real)
    ref_acf = pooled_window_acf(real)
    rows = []
    details = {}
    for method, x in evaluated.items():
        assert x.shape == (COUNT, 30, 5) and np.isfinite(x).all()
        mean, std, corr = statistics(x)
        acf = pooled_window_acf(x)
        row = {
            "method": method,
            "n_real": len(real),
            "n_evaluated": len(x),
            "mean_abs_diff": float(np.abs(mean - ref_mean).mean()),
            "std_abs_diff": float(np.abs(std - ref_std).mean()),
            "corr_abs_diff": float(np.abs(corr - ref_corr).mean()),
            "acf_abs_diff": float(np.abs(acf - ref_acf).mean()),
        }
        assert all(np.isfinite(row[k]) for k in row if k.endswith("diff"))
        rows.append(row)
        details[method] = {
            "mean_abs_diff_by_feature": np.abs(mean - ref_mean).tolist(),
            "std_abs_diff_by_feature": np.abs(std - ref_std).tolist(),
            "acf_abs_diff_by_feature": np.abs(acf - ref_acf).mean(axis=0).tolist(),
            "acf_by_lag_and_feature": acf.tolist(),
        }

    metadata = {
        "scope": "Descriptive similarity to the 247 training anomaly windows; not independent generalization or downstream detection performance.",
        "retrained": False,
        "features": features,
        "pool_size_each_method": 1000,
        "evaluation_count_each_method": COUNT,
        "baseline_sampling_seed": SEED,
        "baseline_sample_indices": indices.tolist(),
        "baseline_selection": "Same random index vector, without replacement, from each saved pool.",
        "fmd_selection": "Existing default quality_filter_generated_anomalies, 750 selected from the full 1000-window Masking Diffusion pool; full normal training pool used as filter reference.",
        "fmd_filter_parameters": {"anomaly_radius_multiplier": 1.25, "normal_radius_multiplier": 0.75, "within_anomaly_range_threshold": 0.90},
        "statistics": "Mean/std pooled across windows and observations; population std; mean absolute entrywise error of full 5x5 Pearson correlation matrices, including diagonals, matching the existing quality metric.",
        "acf": "Demean within each window, then pool lagged cross-products and lag-zero squared deviations over all windows. ACF error is averaged over 5 features and observation lags 1..10; no cross-window products. Lags are observation steps, not seconds.",
        "uncertainty": "One fixed saved pool and one sampling seed; no retraining uncertainty estimated.",
        "reference_acf": ref_acf.tolist(),
        "results": rows,
        "feature_details": details,
    }
    output = ROOT / "output/section_4_4_1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation_quality_comparison.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Table A. Generation quality comparison",
        "",
        "Reference: 247 real anomaly training windows. All methods evaluated on 750 windows from saved 1,000-window pools. Lower is closer for every error metric.",
        "",
        "| Method | N | Mean difference | Std difference | Correlation difference | ACF difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['n_evaluated']} | {row['mean_abs_diff']:.5f} | {row['std_abs_diff']:.5f} | {row['corr_abs_diff']:.5f} | {row['acf_abs_diff']:.5f} |")
    lines += [
        "",
        "Non-filtered methods: random sampling without replacement, seed 42. FMD: default filter selects 750 candidates from the full Masking Diffusion pool. No model retraining.",
        "",
        "ACF: pooled within-window autocorrelation at observation lags 1-10, averaged in absolute difference over the five input features. No products cross window boundaries. See the sibling JSON for full definitions and per-feature values.",
        "",
        "These are training-reference similarity results, not proof of independent generalization, diversity, or downstream detection improvement.",
    ]
    (output / "generation_quality_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
