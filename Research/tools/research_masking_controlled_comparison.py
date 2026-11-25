from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import research02_generate_data as r02
import research_section441_matched_rf as matched
import research_section43_unified as unified
from research05_balanced_aug_experiment import ensure_window_3d, load_npz_xy
from research09_filtering_strength import quality_filter_generated_anomalies_with_strength


TARGET_COUNT = 750
NORMAL_N = 5000
NORMAL_SEED = 65
FILTERING_PARAMS = {
    "anomaly_radius_multiplier": 1.50,
    "normal_radius_multiplier": 0.50,
    "within_anomaly_range_threshold": 0.80,
}

# (name, temporal block length, feature group size, experiment family)
CONFIGS = [
    ("amount_46", 4, 1, "mask_amount_fixed_composition"),
    ("amount_84", 8, 2, "mask_amount_fixed_composition"),
    ("amount_114", 12, 3, "mask_amount_fixed_composition"),
    ("temporal_heavy_94", 16, 1, "composition_fixed_amount"),
    ("feature_heavy_94", 2, 3, "composition_fixed_amount"),
]


def load_generated_x(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = "x" if "x" in data else "X" if "X" in data else data.files[0]
        return ensure_window_3d(data[key].astype(np.float32))


def main() -> None:
    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    cache_dir = root / "data" / "research10" / "controlled_generated"
    unfiltered = "--unfiltered" in sys.argv[1:]
    result_dir = root / "output" / (
        "masking_controlled_comparison_unfiltered"
        if unfiltered
        else "masking_controlled_comparison"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    x_normal_all, _ = load_npz_xy(split_dir / "normal_train_windows.npz")
    x_anomaly_seed, _ = load_npz_xy(split_dir / "anomaly_train_seed_windows.npz")
    x_val_all, y_val_all = load_npz_xy(split_dir / "validation_windows.npz")
    x_final_all, y_final_all = load_npz_xy(split_dir / "final_test_windows.npz")
    train_meta = pd.concat(
        [
            pd.read_csv(split_dir / "normal_train_metadata.csv"),
            pd.read_csv(split_dir / "anomaly_train_seed_metadata.csv"),
        ],
        ignore_index=True,
    )
    val_meta = pd.read_csv(split_dir / "validation_metadata.csv")
    test_meta = pd.read_csv(split_dir / "final_test_metadata.csv")
    train_rows = matched.source_rows(train_meta)
    val_keep = matched.disjoint_mask(val_meta, train_rows)
    val_rows = matched.source_rows(val_meta.loc[val_keep])
    used_rows = {
        key: train_rows.get(key, set()) | val_rows.get(key, set())
        for key in train_rows.keys() | val_rows.keys()
    }
    test_keep = matched.disjoint_mask(test_meta, used_rows)
    x_val, y_val = x_val_all[val_keep], y_val_all[val_keep]
    x_final, y_final = x_final_all[test_keep], y_final_all[test_keep]
    normal_indices = np.random.default_rng(NORMAL_SEED).choice(
        len(x_normal_all), NORMAL_N, replace=False
    )
    x_normal_train = x_normal_all[normal_indices]
    feature_names = np.array(["RealPower", "GateOnTime", "SetPower", "Speed", "Length"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: list[dict] = []
    generation_summaries: dict[str, dict] = {}
    for name, block_len, group_len, family in CONFIGS:
        temporal_ratio = block_len / 30
        feature_ratio = group_len / 5
        if name == "amount_46":
            pool_path = root / "data" / "research10" / "generated" / "diffusion_masked_small_windows.npz"
        elif name == "amount_84":
            pool_path = root / "data" / "research10" / "generated" / "diffusion_masked_default_windows.npz"
        else:
            pool_path = cache_dir / f"{name}.npz"

        if pool_path.exists():
            generated_x = load_generated_x(pool_path)
            generation_summary = {"loaded_from_cache": True, "generated_path": str(pool_path)}
        else:
            r02.TEMPORAL_BLOCK_RATIO = temporal_ratio
            r02.FEATURE_GROUP_RATIO = feature_ratio
            r02.set_seed()
            mask_config = r02.build_adaptive_mask_config(x_normal_train, x_anomaly_seed, feature_names)
            generated_x = r02.generate_diffusion_masked(
                x_anomaly_seed,
                device=device,
                n_generate=r02.N_GENERATE,
                mask_config=mask_config,
            )
            np.savez_compressed(pool_path, x=generated_x.astype(np.float32))
            generation_summary = {
                "loaded_from_cache": False,
                "generated_path": str(pool_path),
                "adaptive_masking": r02.summarize_adaptive_mask_config(mask_config),
            }

        generated_x = ensure_window_3d(generated_x)
        if unfiltered:
            augmented_x = generated_x
            filter_diagnostics = {"filtering_applied": False}
        else:
            augmented_x, filter_diagnostics = quality_filter_generated_anomalies_with_strength(
                x_generated=generated_x,
                x_anomaly_seed=x_anomaly_seed,
                x_normal_train=x_normal_all,
                n_select=TARGET_COUNT,
                **FILTERING_PARAMS,
            )
        # Keep the classifier, train composition, validation threshold policy, and seed identical.
        metrics, _, _ = unified.fit_eval(
            x_normal_train,
            x_anomaly_seed,
            augmented_x,
            x_val,
            y_val,
            x_final,
            y_final,
            unified.MODEL_SEED,
        )
        row = {
            "method": name,
            "augmentation_family": family,
            "augmentation_count": len(augmented_x),
            **metrics,
        }
        masked_cells = block_len * 5 + group_len * 30 - block_len * group_len
        row.update(
            {
                "experiment_family": family,
                "temporal_block_len": block_len,
                "feature_group_len": group_len,
                "temporal_block_ratio": temporal_ratio,
                "feature_group_ratio": feature_ratio,
                "masked_cells_per_window": masked_cells,
                "total_cells_per_window": 150,
                "masked_cell_fraction": masked_cells / 150,
                **filter_diagnostics,
            }
        )
        rows.append(row)
        generation_summaries[name] = generation_summary
        print(
            f"{name}: masked={masked_cells}/150, P={row['precision']:.4f}, "
            f"R={row['recall']:.4f}, F1={row['f1']:.4f}, F2={row['f2']:.4f}, "
            f"FN={row['fn']}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    csv_path = result_dir / "masking_controlled_comparison.csv"
    results.to_csv(csv_path, index=False)
    summary = {
        "purpose": "Separate masking amount from temporal-versus-feature allocation.",
        "window_shape": {"timesteps": 30, "features": 5, "cells": 150},
        "target_generated_count": int(len(generated_x)),
        "filtering_applied": not unfiltered,
        "filtering": None if unfiltered else FILTERING_PARAMS,
        "threshold_policy": "Best validation F1; same validation/test split and classifier seed for all conditions.",
        "classifier": "RandomForest, 300 trees, min_samples_leaf=2, balanced_subsample.",
        "normal_count": len(x_normal_train),
        "real_anomaly_seed_count": len(x_anomaly_seed),
        "clean_validation": {"n": int(len(y_val)), "normal": int((y_val == 0).sum()), "anomaly": int((y_val == 1).sum())},
        "clean_final_test": {"n": int(len(y_final)), "normal": int((y_final == 0).sum()), "anomaly": int((y_final == 1).sum())},
        "split_exclusion": "Same source-row-disjoint validation and final-test selection as research_section43_unified.py.",
        "generation_summaries": generation_summaries,
        "results_csv": str(csv_path),
    }
    with (result_dir / "masking_controlled_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
