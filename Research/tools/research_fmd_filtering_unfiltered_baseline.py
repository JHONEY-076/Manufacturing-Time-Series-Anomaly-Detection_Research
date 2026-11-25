"""Add a random-selection baseline to the F1-threshold FMD filter comparison."""
from pathlib import Path

import pandas as pd
import numpy as np

from research_section43_ppt_aligned import (
    FILTERS, NORMAL_N, SENSITIVITY_SEED, filtered, fit_eval, load_data,
    read_npz, sample_normal,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/section_4_3_unified/ppt_aligned/filtering_strength_with_unfiltered.csv"
POOL = ROOT / "data/research02/generated/diffusion_masked_windows.npz"

def main():
    normal_all, anomaly, xv, yv, xt, yt = load_data()
    normal = sample_normal(normal_all, NORMAL_N, SENSITIVITY_SEED)
    pool = read_npz(POOL)
    random_idx = np.random.default_rng(SENSITIVITY_SEED).choice(len(pool), 750, replace=False)
    rows = []
    random_result = fit_eval(
        "No Filtering (Random Selection)", normal, anomaly, pool[random_idx],
        xv, yv, xt, yt, SENSITIVITY_SEED,
        {"selection_method": "Random", "candidates_passing_filter": None,
         "selected_count": 750},
    )
    rows.append(random_result)
    for strength in FILTERS:
        selected, diag = filtered(pool, anomaly, normal_all, 750, strength)
        result = fit_eval(
            f"FMD {strength}", normal, anomaly, selected, xv, yv, xt, yt,
            SENSITIVITY_SEED,
            {"selection_method": strength, **diag},
        )
        rows.append(result)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows)[["method", "candidates_passing_filter", "selected_count",
                              "precision", "recall", "f1", "auroc", "ap", "fp", "fn"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
