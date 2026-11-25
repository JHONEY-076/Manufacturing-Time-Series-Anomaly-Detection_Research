"""Repeat section 4.3 sensitivity experiments on the PPT-era evaluation split.

All operating thresholds are selected by maximum validation F1, matching the
PPT comparison protocol. The pre-generated pools are reused; no generator is
retrained. Test windows are evaluated as stored (2,785 normal, 296 anomaly),
without the later source-row-overlap exclusion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from research05_balanced_aug_experiment import (
    comprehensive_augmentation, frequency_domain, magnitude_warp,
    noise_injection, time_warp,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/augmentation_split"
GEN = ROOT / "data/research02/generated"
MASK_RATIO = ROOT / "data/research10/generated"
MASK_CONTROLLED = ROOT / "data/research10/controlled_generated"
OUT = ROOT / "output/section_4_3_unified/ppt_aligned"

NORMAL_N = 5000
REAL_ANOMALY_N = 247
RANDOM_STATE = 42
SENSITIVITY_SEED = 42
RF_PARAMS = dict(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
)
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
FILTERS = {
    "Loose": (1.50, 0.50, 0.80),
    "Default": (1.25, 0.75, 0.90),
    "Strict": (1.00, 1.00, 0.95),
}


def ensure_window_3d(x):
    x = np.asarray(x)
    if x.ndim == 3:
        return x.astype(np.float32)
    x = np.squeeze(x)
    if x.ndim == 3:
        return x.astype(np.float32)
    raise ValueError(f"Expected 3D windows, got {x.shape}")


def read_npz(path, with_y=False):
    with np.load(path, allow_pickle=True) as d:
        x = ensure_window_3d(d["X" if "X" in d else "x"])
        if with_y:
            return x, d["y"].astype(np.int8)
        return x


def quality_filter_generated_anomalies_with_strength(
    x_generated, x_anomaly_seed, x_normal_train, n_select,
    anomaly_radius_multiplier, normal_radius_multiplier,
    within_anomaly_range_threshold,
):
    """Same distance/range gate and ranking used by research09."""
    gen = ensure_window_3d(x_generated).reshape(len(x_generated), -1).astype(np.float64)
    an = ensure_window_3d(x_anomaly_seed).reshape(len(x_anomaly_seed), -1).astype(np.float64)
    no = ensure_window_3d(x_normal_train).reshape(len(x_normal_train), -1).astype(np.float64)
    ref = np.concatenate([no, an])
    mean, std = ref.mean(0), ref.std(0)
    std[std < 1e-6] = 1.0
    gz, az, nz = (gen - mean) / std, (an - mean) / std, (no - mean) / std
    ac, nc = az.mean(0), nz.mean(0)
    da = np.linalg.norm(gz - ac, axis=1)
    dn = np.linalg.norm(gz - nc, axis=1)
    amean, astd = an.mean(0), an.std(0)
    astd[astd < 1e-6] = 1.0
    within = (np.abs((gen - amean) / astd) <= 3.0).mean(1)
    ar = np.percentile(np.linalg.norm(az - ac, axis=1), 95)
    nr = np.percentile(np.linalg.norm(nz - nc, axis=1), 25)
    eligible = ((da <= ar * anomaly_radius_multiplier)
                & (dn >= nr * normal_radius_multiplier)
                & (within >= within_anomaly_range_threshold))
    ids = np.flatnonzero(eligible)
    fallback = len(ids) < n_select
    if fallback:
        ids = np.arange(len(gen))
    score = -da + 0.5 * dn + 10.0 * within
    chosen = ids[np.argsort(score[ids])[::-1][:min(n_select, len(ids))]]
    return ensure_window_3d(x_generated)[chosen], {
        "candidates_passing_filter": int(eligible.sum()),
        "fallback_used": bool(fallback),
        "selected_count": int(len(chosen)),
    }


def load_data():
    xn_all, _ = read_npz(SPLIT / "normal_train_windows.npz", True)
    xa, _ = read_npz(SPLIT / "anomaly_train_seed_windows.npz", True)
    xv, yv = read_npz(SPLIT / "validation_windows.npz", True)
    xt, yt = read_npz(SPLIT / "final_test_windows.npz", True)
    assert len(xa) == REAL_ANOMALY_N
    assert int((yt == 0).sum()) == 2785 and int((yt == 1).sum()) == 296
    return xn_all, xa, xv, yv, xt, yt


def sample_normal(x_normal_all, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_normal_all), n, replace=False)
    return x_normal_all[idx]


def choose_f1_threshold(y, p):
    rows = []
    for t in THRESHOLDS:
        pred = (p >= t).astype(np.int8)
        rows.append((
            float(f1_score(y, pred, zero_division=0)),
            float(recall_score(y, pred, zero_division=0)),
            float(precision_score(y, pred, zero_division=0)),
            float(t),
        ))
    f1, recall, precision, threshold = max(rows, key=lambda row: row[:3])
    return {"threshold": threshold, "validation_f1": f1,
            "validation_recall": recall, "validation_precision": precision}


def fit_eval(name, normal, anomaly, augmented, xv, yv, xt, yt, model_seed, extras=None):
    normal, anomaly = ensure_window_3d(normal), ensure_window_3d(anomaly)
    blocks = [normal, anomaly]
    n_aug = 0
    if augmented is not None:
        augmented = ensure_window_3d(augmented)
        blocks.append(augmented)
        n_aug = len(augmented)
    xx = np.concatenate(blocks)
    yy = np.r_[np.zeros(len(normal), np.int8), np.ones(len(xx) - len(normal), np.int8)]
    model = RandomForestClassifier(**RF_PARAMS, random_state=model_seed)
    model.fit(xx.reshape(len(xx), -1), yy)
    pv = model.predict_proba(xv.reshape(len(xv), -1))[:, 1]
    pt = model.predict_proba(xt.reshape(len(xt), -1))[:, 1]
    selected = choose_f1_threshold(yv, pv)
    pred = (pt >= selected["threshold"]).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return {
        "method": name,
        "normal_train": len(normal),
        "real_anomaly_train": len(anomaly),
        "added_anomalies": n_aug,
        **selected,
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall": float(recall_score(yt, pred, zero_division=0)),
        "f1": float(f1_score(yt, pred, zero_division=0)),
        "auroc": float(roc_auc_score(yt, pt)),
        "ap": float(average_precision_score(yt, pt)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        **(extras or {}),
    }


def filtered(pool, anomaly, normal_ref, n, strength):
    a, b, c = FILTERS[strength]
    chosen, diag = quality_filter_generated_anomalies_with_strength(
        x_generated=pool,
        x_anomaly_seed=anomaly,
        x_normal_train=normal_ref,
        n_select=n,
        anomaly_radius_multiplier=a,
        normal_radius_multiplier=b,
        within_anomaly_range_threshold=c,
    )
    return chosen, diag


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xn_all, xa, xv, yv, xt, yt = load_data()
    md = read_npz(GEN / "diffusion_masked_windows.npz")

    # Main augmentation comparison on the same PPT-era split and F1 threshold rule.
    # Transformation methods and unfiltered generative methods add 1,000 samples;
    # FMD adds the 750 selected samples from the 1,000-candidate masked pool.
    main_rows = []
    main_normal = sample_normal(xn_all, NORMAL_N, SENSITIVITY_SEED)
    main_rows.append(fit_eval("Original Only", main_normal, xa, None, xv, yv, xt, yt,
                              SENSITIVITY_SEED,
                              {"experiment": "main_comparison", "method_family": "Baseline"}))
    transform_fns = {
        "Time Warping": time_warp,
        "Magnitude Warping": magnitude_warp,
        "Noise Injection": noise_injection,
        "Frequency Domain Transform": frequency_domain,
        "Comprehensive Transform": comprehensive_augmentation,
    }
    for index, (name, function) in enumerate(transform_fns.items()):
        aug = function(xa, 1000, seed=17000 + index)
        main_rows.append(fit_eval(name, main_normal, xa, aug, xv, yv, xt, yt,
                                  SENSITIVITY_SEED,
                                  {"experiment": "main_comparison", "method_family": "Transformation"}))
    generated_files = {
        "GT-GAN": "gtgan_series_windows.npz",
        "Diffusion": "diffusion_series_windows.npz",
        "Masked GT-GAN": "gtgan_masked_windows.npz",
        "Masked Diffusion": "diffusion_masked_windows.npz",
    }
    for name, filename in generated_files.items():
        aug = read_npz(GEN / filename)
        main_rows.append(fit_eval(name, main_normal, xa, aug, xv, yv, xt, yt,
                                  SENSITIVITY_SEED,
                                  {"experiment": "main_comparison", "method_family": "Generative"}))
    fmd_main, fmd_diag = filtered(md, xa, xn_all, 750, "Default")
    main_rows.append(fit_eval("FMD", main_normal, xa, fmd_main, xv, yv, xt, yt,
                              SENSITIVITY_SEED,
                              {"experiment": "main_comparison", "method_family": "Generative",
                               **fmd_diag}))

    # Generated-sample count: FMD-ranked samples, re-thresholded by validation F1.
    count_rows = []
    count_normal = sample_normal(xn_all, NORMAL_N, SENSITIVITY_SEED)
    for n in [0, 50, 100, 200, 500, 750, 1000]:
        aug = None if n == 0 else filtered(md, xa, xn_all, n, "Default")[0]
        count_rows.append(fit_eval(f"FMD {n}", count_normal, xa, aug, xv, yv, xt, yt,
                                   SENSITIVITY_SEED, {"experiment": "generated_count"}))

    # Class composition at 750 added anomaly windows.
    fmd750 = filtered(md, xa, xn_all, 750, "Default")[0]
    md750 = md[np.random.default_rng(42).choice(len(md), 750, replace=False)]
    balance_rows = []
    cases = [
        ("Original Only", None, "Balanced", REAL_ANOMALY_N),
        ("Masked Diffusion", md750, "Balanced", REAL_ANOMALY_N + 750),
        ("FMD", fmd750, "Balanced", REAL_ANOMALY_N + 750),
        ("Original Only", None, "Imbalanced", NORMAL_N),
        ("Masked Diffusion", md750, "Imbalanced", NORMAL_N),
        ("FMD", fmd750, "Imbalanced", NORMAL_N),
    ]
    normal_order = np.random.default_rng(SENSITIVITY_SEED).permutation(len(xn_all))
    for method, aug, composition, n_normal in cases:
        normal = xn_all[normal_order[:n_normal]]
        balance_rows.append(fit_eval(
            method, normal[:n_normal], xa, aug, xv, yv, xt, yt,
            SENSITIVITY_SEED,
            {"experiment": "class_composition", "class_composition": composition,
             "normal_to_anomaly_ratio": n_normal / (len(xa) + (0 if aug is None else len(aug)))},
        ))

    # Mask amount and allocation: use all 1,000 candidates without FMD filtering,
    # matching the original masking ablation protocol.
    pools = [
        ("Mask amount 46/150", MASK_RATIO / "diffusion_masked_small_windows.npz", 46, "Amount"),
        ("Mask amount 84/150", MASK_RATIO / "diffusion_masked_default_windows.npz", 84, "Amount"),
        ("Mask amount 114/150", MASK_CONTROLLED / "amount_114.npz", 114, "Amount"),
        ("Temporal-focused 94/150", MASK_CONTROLLED / "temporal_heavy_94.npz", 94, "Allocation"),
        ("Feature-focused 94/150", MASK_CONTROLLED / "feature_heavy_94.npz", 94, "Allocation"),
    ]
    mask_rows = []
    mask_normal = sample_normal(xn_all, NORMAL_N, SENSITIVITY_SEED)
    for name, path, cells, family in pools:
        pool = read_npz(path)
        aug = pool
        mask_rows.append(fit_eval(name, mask_normal, xa, aug, xv, yv, xt, yt,
                                   SENSITIVITY_SEED,
                                   {"experiment": "masking", "masking_family": family,
                                    "masked_cells": cells,
                                    "candidates_passing_filter": None,
                                    "selected_count": len(aug)}))

    # Filtering strength with the same default-mask candidate pool and 750 selected.
    filtering_rows = []
    filter_normal = sample_normal(xn_all, NORMAL_N, SENSITIVITY_SEED)
    for strength in FILTERS:
        aug, diag = filtered(md, xa, xn_all, 750, strength)
        filtering_rows.append(fit_eval(f"FMD {strength}", filter_normal, xa, aug, xv, yv, xt, yt,
                                       SENSITIVITY_SEED,
                                       {"experiment": "filtering_strength",
                                        "filtering_strength": strength, **diag}))

    groups = {
        "main_comparison": main_rows,
        "generated_count": count_rows,
        "class_composition": balance_rows,
        "masking": mask_rows,
        "filtering_strength": filtering_rows,
    }
    for key, rows in groups.items():
        pd.DataFrame(rows).to_csv(OUT / f"{key}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "protocol": "PPT-aligned sensitivity rerun",
        "threshold_selection": "Maximum validation F1; ties resolved by recall then precision; thresholds 0.05 to 0.95 by 0.01.",
        "normal_train_pool": 5000,
        "real_anomaly_train": 247,
        "final_test": {"normal": 2785, "anomaly": 296},
        "classifier": {**RF_PARAMS, "random_state": SENSITIVITY_SEED},
        "seed_protocol": "Paired sensitivities use normal-sample seed 42 and Random Forest seed 42 within each comparison group.",
        "generators_retrained": False,
        "runtime": {"python": __import__("sys").version.split()[0],
                    "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "filter_parameters": FILTERS,
        **groups,
    }
    (OUT / "ppt_aligned_sensitivity.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: [{"method": r["method"], "threshold": r["threshold"],
                           "precision": r["precision"], "recall": r["recall"],
                           "f1": r["f1"], "fp": r["fp"], "fn": r["fn"],
                           **{q: r[q] for q in ("class_composition", "masked_cells", "filtering_strength") if q in r}}
                          for r in rows] for k, rows in groups.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
