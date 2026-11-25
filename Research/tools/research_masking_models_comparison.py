"""Compare Masked GT-GAN and Masked Diffusion across controlled mask designs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import research02_generate_data as r02
from research_section43_ppt_aligned import (
    NORMAL_N, SENSITIVITY_SEED, fit_eval, load_data, sample_normal,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/section_4_3_unified/ppt_aligned"
POOL_DIR = ROOT / "data/research10/controlled_generated"
DIFF_DIR = ROOT / "data/research10/generated"
FEATURE_NAMES = np.array(["RealPower", "GateOnTime", "SetPower", "Speed", "Length"])
N_GENERATE = 1000
CONFIGS = [
    ("Mask amount 46/150", 4, 1, "Amount", DIFF_DIR / "diffusion_masked_small_windows.npz"),
    ("Mask amount 84/150", 8, 2, "Amount", DIFF_DIR / "diffusion_masked_default_windows.npz"),
    ("Mask amount 114/150", 12, 3, "Amount", POOL_DIR / "amount_114.npz"),
    ("Temporal-focused 94/150", 16, 1, "Allocation", POOL_DIR / "temporal_heavy_94.npz"),
    ("Feature-focused 94/150", 2, 3, "Allocation", POOL_DIR / "feature_heavy_94.npz"),
]


def ensure_3d(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D windows, got {x.shape}")
    return x.astype(np.float32)


def load_pool(path: Path):
    with np.load(path, allow_pickle=True) as f:
        key = "X" if "X" in f else "x"
        return ensure_3d(f[key])


def generate_masked_gtgan(real_x, normal_all, block_len, group_len, name):
    cache = POOL_DIR / f"gtgan_{name.lower().replace(' ', '_').replace('/', '_')}.npz"
    if cache.exists():
        print(f"Loading cached Masked GT-GAN pool: {cache.name}", flush=True)
        return load_pool(cache), {"generated_now": False, "pool": str(cache)}

    r02.TEMPORAL_BLOCK_RATIO = block_len / real_x.shape[1]
    r02.FEATURE_GROUP_RATIO = group_len / real_x.shape[2]
    r02.set_seed()
    config = r02.build_adaptive_mask_config(normal_all, real_x, FEATURE_NAMES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Masked GT-GAN: {name}, block={block_len}, features={group_len}, device={device}", flush=True)
    generated = r02.generate_gtgan_masked(real_x, device=device, n_generate=N_GENERATE, mask_config=config)
    generated = ensure_3d(generated)
    np.savez_compressed(cache, X=generated, y=np.ones(len(generated), dtype=np.int8))
    return generated, {"generated_now": True, "pool": str(cache),
                       "epochs": r02.EPOCHS, "adaptive_masking": r02.summarize_adaptive_mask_config(config)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    normal_all, anomaly, xv, yv, xt, yt = load_data()
    normal = sample_normal(normal_all, NORMAL_N, SENSITIVITY_SEED)
    rows = []
    generation_info = []

    for name, block_len, group_len, family, diffusion_path in CONFIGS:
        cells = block_len * 5 + group_len * 30 - block_len * group_len
        ratio = cells / 150
        diff_pool = load_pool(diffusion_path)
        if len(diff_pool) < N_GENERATE:
            raise ValueError(f"{diffusion_path} has only {len(diff_pool)} samples")
        gt_pool, gt_info = generate_masked_gtgan(anomaly, normal_all, block_len, group_len, name)
        if len(gt_pool) < N_GENERATE:
            raise ValueError(f"Masked GT-GAN pool for {name} has only {len(gt_pool)} samples")

        for method, pool, gen_info in [
            ("Masked GT-GAN", gt_pool, gt_info),
            ("Masked Diffusion", diff_pool, {"generated_now": False, "pool": str(diffusion_path)}),
        ]:
            result = fit_eval(
                f"{method} - {name}", normal, anomaly, pool[:N_GENERATE],
                xv, yv, xt, yt, SENSITIVITY_SEED,
                {"experiment": "masked_model_comparison", "method_family": family,
                 "masked_cells": cells, "masked_ratio": ratio,
                 "temporal_block_steps": block_len, "feature_group_features": group_len},
            )
            rows.append(result)
            generation_info.append({"method": method, "mask": name, **gen_info})
            print(f"{method} {name}: P={result['precision']:.4f}, R={result['recall']:.4f}, "
                  f"F1={result['f1']:.4f}, FP={result['fp']}, FN={result['fn']}", flush=True)

    out_csv = OUT / "masking_models_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary = {
        "experiment": "Masked GT-GAN vs Masked Diffusion across mask amount and allocation",
        "normal_train": NORMAL_N,
        "real_anomaly_train": len(anomaly),
        "generated_per_condition": N_GENERATE,
        "final_test_normal": int((yt == 0).sum()),
        "final_test_anomaly": int((yt == 1).sum()),
        "threshold_rule": "Maximum validation F1; ties resolved by recall then precision.",
        "classifier": {"name": "RandomForestClassifier", "n_estimators": 300,
                       "min_samples_leaf": 2, "class_weight": "balanced_subsample",
                       "random_state": SENSITIVITY_SEED},
        "generation_epochs": r02.EPOCHS,
        "generation_info": generation_info,
        "results": rows,
    }
    (OUT / "masking_models_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
