from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_section441_matched_rf as matched
import research_section43_unified as unified
from research05_balanced_aug_experiment import load_npz_x, load_npz_xy
from research09_filtering_strength import FILTERING_STRENGTHS, quality_filter_generated_anomalies_with_strength


COUNTS = [0, 50, 100, 200, 500, 750, 1000]
NORMAL_N = 5000
NORMAL_SEED = 65


def main():
    root = Path(__file__).resolve().parents[1]
    split = root / "data" / "augmentation_split"
    output = root / "output" / "masking_count_unified"
    output.mkdir(parents=True, exist_ok=True)
    normal_all, _ = load_npz_xy(split / "normal_train_windows.npz")
    anomaly, _ = load_npz_xy(split / "anomaly_train_seed_windows.npz")
    val_all, y_val_all = load_npz_xy(split / "validation_windows.npz")
    test_all, y_test_all = load_npz_xy(split / "final_test_windows.npz")
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
    x_val, y_val = val_all[val_keep], y_val_all[val_keep]
    x_test, y_test = test_all[test_keep], y_test_all[test_keep]
    normal_indices = np.random.default_rng(NORMAL_SEED).choice(len(normal_all), NORMAL_N, replace=False)
    x_normal = normal_all[normal_indices]
    pool = load_npz_x(root / "data" / "research02" / "generated" / "diffusion_masked_windows.npz")

    rows = []
    for count in COUNTS:
        selected = None if count == 0 else quality_filter_generated_anomalies_with_strength(
            pool, anomaly, normal_all, count, **FILTERING_STRENGTHS["default"]
        )[0]
        metrics, _, _ = unified.fit_eval(x_normal, anomaly, selected, x_val, y_val, x_test, y_test, unified.MODEL_SEED)
        total_anomaly = len(anomaly) + count
        row = {
            "generated_count": count,
            "total_training_anomaly": total_anomaly,
            "normal_to_anomaly_ratio": NORMAL_N / total_anomaly,
            **metrics,
        }
        rows.append(row)
        print(f"count={count}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AP={metrics['auprc_ap']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(output / "masking_count_unified.csv", index=False, encoding="utf-8-sig")
    summary = {
        "purpose": "Unified masking-diffusion augmentation-count comparison with fixed normal training sample and source-row-disjoint evaluation split.",
        "generated_counts": COUNTS,
        "normal_count": len(x_normal),
        "real_anomaly_seed_count": len(anomaly),
        "clean_validation": {"n": int(len(y_val)), "normal": int((y_val == 0).sum()), "anomaly": int((y_val == 1).sum())},
        "clean_final_test": {"n": int(len(y_test)), "normal": int((y_test == 0).sum()), "anomaly": int((y_test == 1).sum())},
        "filtering": FILTERING_STRENGTHS["default"],
        "threshold_policy": "Validation F1 with ties resolved by recall then precision.",
        "classifier": "RandomForest, 300 trees, min_samples_leaf=2, balanced_subsample, seed 42.",
    }
    (output / "masking_count_unified.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
