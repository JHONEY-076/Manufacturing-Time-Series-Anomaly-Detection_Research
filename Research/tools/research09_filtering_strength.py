from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research05_balanced_aug_experiment import (
    ensure_window_3d,
    flatten_windows,
    load_npz_x,
    load_npz_xy,
    sample_rows,
)
from research08_augmentation_saturation import (
    AUGMENTATION_COUNTS,
    NORMAL_TRAIN_SIZE,
    RANDOM_STATE,
    evaluate_augmented_model,
)


TARGET_COUNTS = [750, 1000]
FILTERING_STRENGTHS = {
    "loose": {
        "anomaly_radius_multiplier": 1.50,
        "normal_radius_multiplier": 0.50,
        "within_anomaly_range_threshold": 0.80,
    },
    "default": {
        "anomaly_radius_multiplier": 1.25,
        "normal_radius_multiplier": 0.75,
        "within_anomaly_range_threshold": 0.90,
    },
    "strict": {
        "anomaly_radius_multiplier": 1.00,
        "normal_radius_multiplier": 1.00,
        "within_anomaly_range_threshold": 0.95,
    },
}
QUALITY_RANGE_MULTIPLIER = 3.0


def quality_filter_generated_anomalies_with_strength(
    x_generated: np.ndarray,
    x_anomaly_seed: np.ndarray,
    x_normal_train: np.ndarray,
    n_select: int,
    anomaly_radius_multiplier: float,
    normal_radius_multiplier: float,
    within_anomaly_range_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Select generated anomalies under a configurable filtering gate."""
    x_generated = ensure_window_3d(x_generated)
    x_anomaly_seed = ensure_window_3d(x_anomaly_seed)
    x_normal_train = ensure_window_3d(x_normal_train)

    generated_flat = flatten_windows(x_generated)
    anomaly_flat = flatten_windows(x_anomaly_seed)
    normal_flat = flatten_windows(x_normal_train)

    reference_flat = np.concatenate([normal_flat, anomaly_flat], axis=0)
    reference_mean = reference_flat.mean(axis=0, keepdims=True)
    reference_std = reference_flat.std(axis=0, keepdims=True)
    reference_std = np.where(reference_std < 1e-6, 1.0, reference_std)

    generated_z = (generated_flat - reference_mean) / reference_std
    anomaly_z = (anomaly_flat - reference_mean) / reference_std
    normal_z = (normal_flat - reference_mean) / reference_std

    anomaly_centroid = anomaly_z.mean(axis=0, keepdims=True)
    normal_centroid = normal_z.mean(axis=0, keepdims=True)
    dist_to_anomaly = np.linalg.norm(generated_z - anomaly_centroid, axis=1)
    dist_to_normal = np.linalg.norm(generated_z - normal_centroid, axis=1)

    anomaly_mean = anomaly_flat.mean(axis=0, keepdims=True)
    anomaly_std = anomaly_flat.std(axis=0, keepdims=True)
    anomaly_std = np.where(anomaly_std < 1e-6, 1.0, anomaly_std)
    within_anomaly_range = (
        np.abs((generated_flat - anomaly_mean) / anomaly_std) <= QUALITY_RANGE_MULTIPLIER
    ).mean(axis=1)

    anomaly_radius = np.percentile(np.linalg.norm(anomaly_z - anomaly_centroid, axis=1), 95)
    normal_radius = np.percentile(np.linalg.norm(normal_z - normal_centroid, axis=1), 25)
    candidate_mask = (
        (dist_to_anomaly <= anomaly_radius * anomaly_radius_multiplier)
        & (dist_to_normal >= normal_radius * normal_radius_multiplier)
        & (within_anomaly_range >= within_anomaly_range_threshold)
    )

    quality_score = -dist_to_anomaly + 0.5 * dist_to_normal + 10.0 * within_anomaly_range
    candidate_indices = np.where(candidate_mask)[0]
    fallback_used = len(candidate_indices) < n_select
    if fallback_used:
        candidate_indices = np.arange(len(x_generated))

    ranked_indices = candidate_indices[np.argsort(quality_score[candidate_indices])[::-1]]
    selected_indices = ranked_indices[: min(n_select, len(ranked_indices))]

    diagnostics = {
        "candidate_count": int(candidate_mask.sum()),
        "candidate_ratio": float(candidate_mask.mean()),
        "fallback_used": bool(fallback_used),
        "selected_count": int(len(selected_indices)),
        "selected_quality_mean": float(quality_score[selected_indices].mean()),
        "selected_dist_to_anomaly_mean": float(dist_to_anomaly[selected_indices].mean()),
        "selected_dist_to_normal_mean": float(dist_to_normal[selected_indices].mean()),
        "selected_within_anomaly_range_mean": float(within_anomaly_range[selected_indices].mean()),
    }
    return x_generated[selected_indices].astype(np.float32), diagnostics


def main() -> None:
    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    generated_dir = root / "data" / "research02" / "generated"
    result_dir = root / "data" / "research09" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    x_normal_train, _ = load_npz_xy(split_dir / "normal_train_windows.npz")
    x_anomaly_seed, _ = load_npz_xy(split_dir / "anomaly_train_seed_windows.npz")
    x_val, y_val = load_npz_xy(split_dir / "validation_windows.npz")
    x_final, y_final = load_npz_xy(split_dir / "final_test_windows.npz")
    masking_diffusion_x = ensure_window_3d(load_npz_x(generated_dir / "diffusion_masked_windows.npz"))

    rows = []
    for count in TARGET_COUNTS:
        for strength_idx, (strength, params) in enumerate(FILTERING_STRENGTHS.items()):
            seed = RANDOM_STATE + 9000 + count + strength_idx
            augmented_x, diagnostics = quality_filter_generated_anomalies_with_strength(
                x_generated=masking_diffusion_x,
                x_anomaly_seed=x_anomaly_seed,
                x_normal_train=x_normal_train,
                n_select=count,
                **params,
            )
            row = evaluate_augmented_model(
                method=f"Filtered Masking Diffusion ({strength})",
                augmentation_family="generative_filtered",
                augmentation_count=count,
                x_normal_train=x_normal_train,
                x_anomaly_seed=x_anomaly_seed,
                x_augmented=augmented_x,
                x_val=x_val,
                y_val=y_val,
                x_final=x_final,
                y_final=y_final,
                seed=seed,
            )
            row["filtering_strength"] = strength
            row.update(params)
            row.update(diagnostics)
            rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(["augmentation_count", "filtering_strength"])
    results_path = result_dir / "research09_filtering_strength.csv"
    summary_path = result_dir / "research09_filtering_strength_summary.json"
    results_df.to_csv(results_path, index=False)

    best_by_f1 = results_df.loc[results_df["f1"].idxmax()].to_dict()
    best_by_f2 = results_df.loc[results_df["f2"].idxmax()].to_dict()
    best_by_recall = results_df.loc[results_df["recall"].idxmax()].to_dict()
    summary = {
        "objective": "Filtering strength analysis for Filtered Masking Diffusion.",
        "target_counts": TARGET_COUNTS,
        "filtering_strengths": FILTERING_STRENGTHS,
        "default_reference": "research08 default filtering at augmentation_count 750 and 1000.",
        "best_by_f1": best_by_f1,
        "best_by_f2": best_by_f2,
        "best_by_recall": best_by_recall,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Research09 filtering strength results")
    print(results_df)


if __name__ == "__main__":
    if not set(TARGET_COUNTS).issubset(set(AUGMENTATION_COUNTS)):
        raise ValueError("TARGET_COUNTS should be selected from Research08 augmentation counts.")
    main()
