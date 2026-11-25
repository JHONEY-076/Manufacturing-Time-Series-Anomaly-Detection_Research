"""Evaluate masking amount and allocation at the validation-F2 threshold."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from research_section43_supplement_f2 import (  # noqa: E402
    NORMAL_SEED, N_NORMAL, SPLIT, OUT, evaluate, npz, source_rows, disjoint,
)

CONTROLLED = ROOT / "data/research10/controlled_generated"
MASK_POOLS = ROOT / "data/research10/generated"
FILES = [
    ("Mask amount: 46/150", "diffusion_masked_small_windows.npz", "Total masked amount", 46),
    ("Mask amount: 84/150", "diffusion_masked_default_windows.npz", "Total masked amount", 84),
    ("Mask amount: 114/150", "amount_114.npz", "Total masked amount", 114),
    ("Temporal-focused: 94/150", "temporal_heavy_94.npz", "Mask allocation", 94),
    ("Feature-focused: 94/150", "feature_heavy_94.npz", "Mask allocation", 94),
]


def main():
    data, meta = {}, {}
    for key in ["normal_train", "anomaly_train_seed", "validation", "final_test"]:
        data[key] = npz(SPLIT / f"{key}_windows.npz")
        meta[key] = pd.read_csv(SPLIT / f"{key}_metadata.csv")
    tr = source_rows(pd.concat([meta["normal_train"], meta["anomaly_train_seed"]], ignore_index=True))
    vk = disjoint(meta["validation"], tr)
    vr = source_rows(meta["validation"].loc[vk])
    used = {k: tr.get(k, set()) | vr.get(k, set()) for k in tr.keys() | vr.keys()}
    tk = disjoint(meta["final_test"], used)
    normal_all, _ = data["normal_train"]
    anomaly, _ = data["anomaly_train_seed"]
    xv_all, yv_all = data["validation"]
    xt_all, yt_all = data["final_test"]
    xv, yv, xt, yt = xv_all[vk], yv_all[vk], xt_all[tk], yt_all[tk]
    idx = np.random.default_rng(NORMAL_SEED).choice(len(normal_all), N_NORMAL, replace=False)
    normal = normal_all[idx]
    rows = []
    for label, filename, family, masked_cells in FILES:
        pool_dir = MASK_POOLS if filename.startswith("diffusion_masked") else CONTROLLED
        aug, _ = npz(pool_dir / filename)
        row = evaluate(label, normal, anomaly, aug, xv, yv, xt, yt,
                       {"experiment_family": family,
                        "masked_cells": masked_cells})
        rows.append(row)
        print(label, row["validation_f2"], row["validation_recall"], row["f2"], row["recall"], row["fp"], row["fn"], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "masking_f2.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
