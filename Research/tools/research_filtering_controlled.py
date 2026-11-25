from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_section441_matched_rf as matched
import research_section43_unified as unified
from research05_balanced_aug_experiment import flatten_windows, load_npz_x, load_npz_xy
from research09_filtering_strength import (
    FILTERING_STRENGTHS,
    QUALITY_RANGE_MULTIPLIER,
    quality_filter_generated_anomalies_with_strength,
)


TARGET_COUNT = 750
NORMAL_N = 5000
NORMAL_SEED = 65
SCORE_WEIGHTS = {
    "baseline (1.0, 0.5, 10)": (1.0, 0.5, 10.0),
    "anomaly proximity (2.0, 0.5, 10)": (2.0, 0.5, 10.0),
    "normal separation (1.0, 1.0, 10)": (1.0, 1.0, 10.0),
    "anomaly range (1.0, 0.5, 20)": (1.0, 0.5, 20.0),
}


def weighted_filter(x_generated, x_anomaly_seed, x_normal_all, weights):
    generated = np.asarray(x_generated, dtype=np.float32)
    generated_flat = flatten_windows(generated)
    anomaly_flat = flatten_windows(x_anomaly_seed)
    normal_flat = flatten_windows(x_normal_all)
    reference = np.concatenate([normal_flat, anomaly_flat], axis=0)
    reference_mean = reference.mean(axis=0, keepdims=True)
    reference_std = reference.std(axis=0, keepdims=True)
    reference_std = np.where(reference_std < 1e-6, 1.0, reference_std)
    generated_z = (generated_flat - reference_mean) / reference_std
    anomaly_z = (anomaly_flat - reference_mean) / reference_std
    normal_z = (normal_flat - reference_mean) / reference_std
    anomaly_centroid = anomaly_z.mean(axis=0, keepdims=True)
    normal_centroid = normal_z.mean(axis=0, keepdims=True)
    dist_anomaly = np.linalg.norm(generated_z - anomaly_centroid, axis=1)
    dist_normal = np.linalg.norm(generated_z - normal_centroid, axis=1)
    anomaly_mean = anomaly_flat.mean(axis=0, keepdims=True)
    anomaly_std = anomaly_flat.std(axis=0, keepdims=True)
    anomaly_std = np.where(anomaly_std < 1e-6, 1.0, anomaly_std)
    within_range = (np.abs((generated_flat - anomaly_mean) / anomaly_std) <= QUALITY_RANGE_MULTIPLIER).mean(axis=1)

    params = FILTERING_STRENGTHS["default"]
    anomaly_radius = np.percentile(np.linalg.norm(anomaly_z - anomaly_centroid, axis=1), 95)
    normal_radius = np.percentile(np.linalg.norm(normal_z - normal_centroid, axis=1), 25)
    eligible = (
        (dist_anomaly <= anomaly_radius * params["anomaly_radius_multiplier"])
        & (dist_normal >= normal_radius * params["normal_radius_multiplier"])
        & (within_range >= params["within_anomaly_range_threshold"])
    )
    wa, wn, wr = weights
    score = -wa * dist_anomaly + wn * dist_normal + wr * within_range
    candidates = np.flatnonzero(eligible)
    fallback = len(candidates) < TARGET_COUNT
    if fallback:
        candidates = np.arange(len(generated))
    selected = candidates[np.argsort(score[candidates])[::-1][:TARGET_COUNT]]
    return generated[selected], {
        "candidate_count": int(eligible.sum()),
        "fallback_used": bool(fallback),
        "selected_quality_score_mean": float(score[selected].mean()),
        "selected_dist_to_anomaly_mean": float(dist_anomaly[selected].mean()),
        "selected_dist_to_normal_mean": float(dist_normal[selected].mean()),
        "selected_within_anomaly_range_mean": float(within_range[selected].mean()),
        "weight_anomaly_distance": wa,
        "weight_normal_distance": wn,
        "weight_anomaly_range": wr,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    split = root / "data" / "augmentation_split"
    output = root / "output" / "filtering_controlled_comparison"
    output.mkdir(parents=True, exist_ok=True)
    x_normal_all, _ = load_npz_xy(split / "normal_train_windows.npz")
    x_anomaly, _ = load_npz_xy(split / "anomaly_train_seed_windows.npz")
    x_val_all, y_val_all = load_npz_xy(split / "validation_windows.npz")
    x_test_all, y_test_all = load_npz_xy(split / "final_test_windows.npz")
    train_meta = pd.concat(
        [pd.read_csv(split / "normal_train_metadata.csv"), pd.read_csv(split / "anomaly_train_seed_metadata.csv")],
        ignore_index=True,
    )
    val_meta = pd.read_csv(split / "validation_metadata.csv")
    test_meta = pd.read_csv(split / "final_test_metadata.csv")
    train_rows = matched.source_rows(train_meta)
    val_keep = matched.disjoint_mask(val_meta, train_rows)
    val_rows = matched.source_rows(val_meta.loc[val_keep])
    used_rows = {key: train_rows.get(key, set()) | val_rows.get(key, set()) for key in train_rows.keys() | val_rows.keys()}
    test_keep = matched.disjoint_mask(test_meta, used_rows)
    x_val, y_val = x_val_all[val_keep], y_val_all[val_keep]
    x_test, y_test = x_test_all[test_keep], y_test_all[test_keep]
    normal_indices = np.random.default_rng(NORMAL_SEED).choice(len(x_normal_all), NORMAL_N, replace=False)
    x_normal = x_normal_all[normal_indices]
    pool = load_npz_x(root / "data" / "research02" / "generated" / "diffusion_masked_windows.npz")

    conditions = []
    rng = np.random.default_rng(42)
    random_indices = rng.choice(len(pool), TARGET_COUNT, replace=False)
    conditions.append(("No filter: random 750", pool[random_indices], {"filter_applied": False}))

    baseline, baseline_diag = quality_filter_generated_anomalies_with_strength(
        pool, x_anomaly, x_normal_all, TARGET_COUNT, **FILTERING_STRENGTHS["default"]
    )
    conditions.append(("FMD default", baseline, {"filter_applied": True, "score_weights": "1.0, 0.5, 10.0", **baseline_diag}))

    for label, weights in SCORE_WEIGHTS.items():
        selected, diagnostics = weighted_filter(pool, x_anomaly, x_normal_all, weights)
        conditions.append((f"FMD weights: {label}", selected, {"filter_applied": True, **diagnostics}))

    rows = []
    for name, selected, diagnostics in conditions:
        metrics, _, _ = unified.fit_eval(x_normal, x_anomaly, selected, x_val, y_val, x_test, y_test, unified.MODEL_SEED)
        rows.append({"condition": name, "augmentation_count": len(selected), **metrics, **diagnostics})
        print(f"{name}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, F2={metrics['f2']:.4f}, FN={metrics['fn']}, FP={metrics['fp']}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "filtering_presence_and_weights.csv", index=False, encoding="utf-8-sig")

    strength_rows = []
    for strength, params in FILTERING_STRENGTHS.items():
        selected, diagnostics = quality_filter_generated_anomalies_with_strength(
            pool, x_anomaly, x_normal_all, TARGET_COUNT, **params
        )
        metrics, _, _ = unified.fit_eval(
            x_normal, x_anomaly, selected, x_val, y_val, x_test, y_test, unified.MODEL_SEED
        )
        strength_rows.append(
            {
                "strength": strength,
                "generated_count": len(selected),
                **params,
                **metrics,
                **diagnostics,
            }
        )
        print(
            f"strength {strength}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, F2={metrics['f2']:.4f}, FN={metrics['fn']}, FP={metrics['fp']}",
            flush=True,
        )
    pd.DataFrame(strength_rows).to_csv(
        output / "filtering_strength.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "purpose": "Compare filter presence at a matched selected count and filter score weights under fixed default eligibility gates.",
        "target_count": TARGET_COUNT,
        "generated_candidate_count": int(len(pool)),
        "normal_count": int(len(x_normal)),
        "real_anomaly_seed_count": int(len(x_anomaly)),
        "clean_validation": {"n": int(len(y_val)), "normal": int((y_val == 0).sum()), "anomaly": int((y_val == 1).sum())},
        "clean_final_test": {"n": int(len(y_test)), "normal": int((y_test == 0).sum()), "anomaly": int((y_test == 1).sum())},
        "split_exclusion": "Same source-row-disjoint validation and final-test selection as research_section43_unified.py.",
        "threshold_policy": "Validation F1 with ties resolved by recall then precision.",
        "classifier": "RandomForest, 300 trees, min_samples_leaf=2, balanced_subsample, seed 42.",
        "score_definition": "-wa * distance_to_anomaly + wn * distance_to_normal + wr * within_anomaly_range; default eligibility gates held fixed.",
    }
    (output / "filtering_presence_and_weights.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
