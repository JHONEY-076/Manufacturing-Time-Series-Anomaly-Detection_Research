from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

import research02_generate_data as r02
from research05_balanced_aug_experiment import ensure_window_3d, load_npz_xy
from research08_augmentation_saturation import (
    NORMAL_TRAIN_SIZE,
    RANDOM_STATE,
    evaluate_augmented_model,
)
from research09_filtering_strength import quality_filter_generated_anomalies_with_strength


TARGET_COUNT = 750
FILTERING_STRENGTH = "loose"
FILTERING_PARAMS = {
    "anomaly_radius_multiplier": 1.50,
    "normal_radius_multiplier": 0.50,
    "within_anomaly_range_threshold": 0.80,
}
MASKING_RATIO_CONFIGS = {
    "small": {
        "temporal_block_ratio": 0.15,
        "feature_group_ratio": 0.20,
    },
    "default": {
        "temporal_block_ratio": 0.25,
        "feature_group_ratio": 0.40,
    },
    "large": {
        "temporal_block_ratio": 0.35,
        "feature_group_ratio": 0.60,
    },
}


def load_npz_x(path: Path) -> np.ndarray:
    data = np.load(path)
    if "x" in data:
        return data["x"]
    if "X" in data:
        return data["X"]
    return data[data.files[0]]


def save_npz_x(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32))


def generate_masking_diffusion_pool(
    real_x: np.ndarray,
    normal_x: np.ndarray,
    feature_names: np.ndarray,
    device: str,
    temporal_block_ratio: float,
    feature_group_ratio: float,
    output_path: Path,
) -> tuple[np.ndarray, dict]:
    if output_path.exists():
        generated = ensure_window_3d(load_npz_x(output_path))
        summary = {
            "loaded_from_cache": True,
            "generated_path": str(output_path),
            "temporal_block_ratio": temporal_block_ratio,
            "feature_group_ratio": feature_group_ratio,
            "n_generated": int(len(generated)),
        }
        return generated, summary

    r02.TEMPORAL_BLOCK_RATIO = temporal_block_ratio
    r02.FEATURE_GROUP_RATIO = feature_group_ratio
    r02.set_seed()

    mask_config = r02.build_adaptive_mask_config(normal_x, real_x, feature_names)
    generated = r02.generate_diffusion_masked(
        real_x,
        device=device,
        n_generate=r02.N_GENERATE,
        mask_config=mask_config,
    )
    save_npz_x(output_path, generated)
    summary = {
        "loaded_from_cache": False,
        "generated_path": str(output_path),
        "temporal_block_ratio": temporal_block_ratio,
        "feature_group_ratio": feature_group_ratio,
        "adaptive_masking": r02.summarize_adaptive_mask_config(mask_config),
        "n_generated": int(len(generated)),
    }
    return generated, summary


def main() -> None:
    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    result_dir = root / "data" / "research10" / "results"
    generated_dir = root / "data" / "research10" / "generated"
    result_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    x_normal_train, _ = load_npz_xy(split_dir / "normal_train_windows.npz")
    x_anomaly_seed, _ = load_npz_xy(split_dir / "anomaly_train_seed_windows.npz")
    x_val, y_val = load_npz_xy(split_dir / "validation_windows.npz")
    x_final, y_final = load_npz_xy(split_dir / "final_test_windows.npz")

    feature_names = np.array(["RealPower", "GateOnTime", "SetPower", "Speed", "Length"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    generation_summaries = {}
    for ratio_idx, (masking_ratio, params) in enumerate(MASKING_RATIO_CONFIGS.items()):
        pool_path = generated_dir / f"diffusion_masked_{masking_ratio}_windows.npz"
        generated_x, generation_summary = generate_masking_diffusion_pool(
            real_x=x_anomaly_seed,
            normal_x=x_normal_train,
            feature_names=feature_names,
            device=device,
            output_path=pool_path,
            **params,
        )
        generation_summaries[masking_ratio] = generation_summary

        augmented_x, diagnostics = quality_filter_generated_anomalies_with_strength(
            x_generated=generated_x,
            x_anomaly_seed=x_anomaly_seed,
            x_normal_train=x_normal_train,
            n_select=TARGET_COUNT,
            **FILTERING_PARAMS,
        )
        seed = RANDOM_STATE + 10000 + ratio_idx
        row = evaluate_augmented_model(
            method=f"Filtered Masking Diffusion ({masking_ratio} mask)",
            augmentation_family="generative_filtered_mask_ratio",
            augmentation_count=TARGET_COUNT,
            x_normal_train=x_normal_train,
            x_anomaly_seed=x_anomaly_seed,
            x_augmented=augmented_x,
            x_val=x_val,
            y_val=y_val,
            x_final=x_final,
            y_final=y_final,
            seed=seed,
        )
        row["masking_ratio"] = masking_ratio
        row["filtering_strength"] = FILTERING_STRENGTH
        row.update(params)
        row.update(diagnostics)
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("masking_ratio")
    results_path = result_dir / "research10_masking_ratio.csv"
    summary_path = result_dir / "research10_masking_ratio_summary.json"
    results_df.to_csv(results_path, index=False)

    summary = {
        "objective": "Masking ratio analysis for Filtered Masking Diffusion.",
        "target_count": TARGET_COUNT,
        "filtering_strength": FILTERING_STRENGTH,
        "filtering_params": FILTERING_PARAMS,
        "masking_ratio_configs": MASKING_RATIO_CONFIGS,
        "device": device,
        "normal_train_used_each_method": NORMAL_TRAIN_SIZE,
        "generation_summaries": generation_summaries,
        "best_by_f1": results_df.loc[results_df["f1"].idxmax()].to_dict(),
        "best_by_f2": results_df.loc[results_df["f2"].idxmax()].to_dict(),
        "best_by_recall": results_df.loc[results_df["recall"].idxmax()].to_dict(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Research10 masking ratio results")
    print(results_df)


if __name__ == "__main__":
    main()
