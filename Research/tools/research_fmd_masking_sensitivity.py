"""Evaluate FMD while varying the underlying Masked Diffusion mask design."""
from pathlib import Path

import numpy as np
import pandas as pd

from research_section43_ppt_aligned import (
    FILTERS, NORMAL_N, SENSITIVITY_SEED, filtered, fit_eval, load_data,
    read_npz, sample_normal,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/section_4_3_unified/ppt_aligned/masking_methods_equal750.csv"
DIFF_DIR = ROOT / "data/research10/generated"
CONTROL_DIR = ROOT / "data/research10/controlled_generated"
POOLS = [
    ("Mask amount 46/150", 46, "Amount", DIFF_DIR / "diffusion_masked_small_windows.npz", "gtgan_mask_amount_46_150.npz"),
    ("Mask amount 84/150", 84, "Amount", DIFF_DIR / "diffusion_masked_default_windows.npz", "gtgan_mask_amount_84_150.npz"),
    ("Mask amount 114/150", 114, "Amount", CONTROL_DIR / "amount_114.npz", "gtgan_mask_amount_114_150.npz"),
    ("Temporal-focused 94/150", 94, "Allocation", CONTROL_DIR / "temporal_heavy_94.npz", "gtgan_temporal-focused_94_150.npz"),
    ("Feature-focused 94/150", 94, "Allocation", CONTROL_DIR / "feature_heavy_94.npz", "gtgan_feature-focused_94_150.npz"),
]
GTGAN_DIR = CONTROL_DIR

def main():
    normal_all, anomaly, xv, yv, xt, yt = load_data()
    normal = sample_normal(normal_all, NORMAL_N, SENSITIVITY_SEED)
    rows = []
    for config_index, (name, masked_cells, family, path, gtgan_file) in enumerate(POOLS):
        pool = read_npz(path)
        gtgan_pool = read_npz(GTGAN_DIR / gtgan_file)
        # Use the same sample indices for the two unfiltered 1,000-sample pools
        # within each mask condition, and keep 750 added samples in all methods.
        sample_idx = np.random.default_rng(4200 + config_index).choice(
            len(pool), 750, replace=False)
        gtgan_idx = np.random.default_rng(4200 + config_index).choice(
            len(gtgan_pool), 750, replace=False)
        for method, candidates, ids, selection in [
            ("Masked GT-GAN", gtgan_pool, gtgan_idx, "Random selection"),
            ("Masked Diffusion", pool, sample_idx, "Random selection"),
        ]:
            result = fit_eval(
                f"{method} - {name}", normal, anomaly, candidates[ids], xv, yv, xt, yt,
                SENSITIVITY_SEED,
                {"experiment": "masking_method_comparison", "method": method,
                 "masking_condition": name,
                 "selection": selection, "masking_family": family,
                 "masked_cells": masked_cells, "masked_proportion": masked_cells / 150,
                 "filter_strength": "None", "selected_count": 750},
            )
            rows.append(result)
            print(f"{method} {name}: P={result['precision']:.4f}, R={result['recall']:.4f}, "
                  f"F1={result['f1']:.4f}, FP={result['fp']}, FN={result['fn']}", flush=True)

        selected, diagnostics = filtered(pool, anomaly, normal_all, 750, "Default")
        result = fit_eval(
            f"FMD - {name}", normal, anomaly, selected, xv, yv, xt, yt,
            SENSITIVITY_SEED,
            {"experiment": "fmd_masking_sensitivity", "method": "FMD",
             "masking_condition": name,
             "selection": "Default FMD ranking", "masking_family": family,
             "masked_cells": masked_cells, "masked_proportion": masked_cells / 150,
             "filter_strength": "Default", **diagnostics},
        )
        rows.append(result)
        print(f"{name}: pass={diagnostics['candidates_passing_filter']}, "
              f"selected={diagnostics['selected_count']}, P={result['precision']:.4f}, "
              f"R={result['recall']:.4f}, F1={result['f1']:.4f}, "
              f"FP={result['fp']}, FN={result['fn']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUT}", flush=True)

if __name__ == "__main__":
    main()
