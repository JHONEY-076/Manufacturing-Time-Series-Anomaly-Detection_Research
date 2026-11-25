"""Evaluate the five saved generative methods at 1,000 samples.

Threshold is selected on validation by maximum F2, with recall and then
precision as tie breakers. Evaluation uses the existing source-row-disjoint
split and does not retrain any generator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

SPLIT = ROOT / "data/augmentation_split"
GENERATED = ROOT / "data/research02/generated"
OUT = ROOT / "output/section_4_3_unified/compare_1000_f2"
METHODS = {
    "GT-GAN": "gtgan_series_windows.npz",
    "Diffusion": "diffusion_series_windows.npz",
    "Masked GT-GAN": "gtgan_masked_windows.npz",
    "Masked Diffusion": "diffusion_masked_windows.npz",
}
N_NORMAL = 5000
N_GENERATED = 1000
NORMAL_SEED = 65
MODEL_SEED = 42
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "n_jobs": 4,
}
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def source_rows(frame):
    result = {}
    for row in frame.itertuples():
        result.setdefault(row.source_file, set()).update(
            range(int(row.start_index), int(row.end_index) + 1))
    return result


def disjoint_mask(frame, used):
    return np.array([
        not any(i in used.get(row.source_file, set())
                for i in range(int(row.start_index), int(row.end_index) + 1))
        for row in frame.itertuples()
    ], dtype=bool)


def read_npz(path: Path):
    with np.load(path, allow_pickle=True) as d:
        x = d["X" if "X" in d else "x"].astype(np.float32)
        y = d["y"].astype(np.int8) if "y" in d else None
    return x, y


def choose_threshold(y, p):
    rows = []
    for t in THRESHOLDS:
        pred = (p >= t).astype(np.int8)
        rows.append((
            float(fbeta_score(y, pred, beta=2, zero_division=0)),
            float(recall_score(y, pred, zero_division=0)),
            float(precision_score(y, pred, zero_division=0)),
            float(t),
        ))
    f2, recall, precision, threshold = max(rows)
    return {"threshold": threshold, "validation_f2": f2,
            "validation_recall": recall, "validation_precision": precision}


def evaluate(name, augmented, x_normal, x_anomaly, x_val, y_val, x_test, y_test):
    blocks = [x_normal, x_anomaly]
    if augmented is not None:
        blocks.append(augmented)
    x_train = np.concatenate(blocks, axis=0)
    y_train = np.r_[np.zeros(len(x_normal), dtype=np.int8),
                    np.ones(len(x_train) - len(x_normal), dtype=np.int8)]
    clf = RandomForestClassifier(**RF_PARAMS, random_state=MODEL_SEED)
    clf.fit(x_train.reshape(len(x_train), -1), y_train)
    val_p = clf.predict_proba(x_val.reshape(len(x_val), -1))[:, 1]
    test_p = clf.predict_proba(x_test.reshape(len(x_test), -1))[:, 1]
    selected = choose_threshold(y_val, val_p)
    pred = (test_p >= selected["threshold"]).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    row = {
        "method": name,
        "generated_count": 0 if augmented is None else len(augmented),
        **selected,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f2": float(fbeta_score(y_test, pred, beta=2, zero_division=0)),
        "auroc": float(roc_auc_score(y_test, test_p)),
        "ap": float(average_precision_score(y_test, test_p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "test_f1": float(fbeta_score(y_test, pred, beta=1, zero_division=0)),
    }
    print(name, json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, metadata = {}, {}
    for split in ["normal_train", "anomaly_train_seed", "validation", "final_test"]:
        data[split] = read_npz(SPLIT / f"{split}_windows.npz")
        metadata[split] = pd.read_csv(SPLIT / f"{split}_metadata.csv")
    train_rows = source_rows(pd.concat(
        [metadata["normal_train"], metadata["anomaly_train_seed"]], ignore_index=True))
    val_keep = disjoint_mask(metadata["validation"], train_rows)
    val_rows = source_rows(metadata["validation"].loc[val_keep])
    used = {k: train_rows.get(k, set()) | val_rows.get(k, set())
            for k in train_rows.keys() | val_rows.keys()}
    test_keep = disjoint_mask(metadata["final_test"], used)
    x_normal_all, _ = data["normal_train"]
    x_anomaly, _ = data["anomaly_train_seed"]
    _, y_val_all = data["validation"]
    _, y_test_all = data["final_test"]
    x_val_all, _ = data["validation"]
    x_test_all, _ = data["final_test"]
    x_val, y_val = x_val_all[val_keep], y_val_all[val_keep]
    x_test, y_test = x_test_all[test_keep], y_test_all[test_keep]
    normal_idx = np.random.default_rng(NORMAL_SEED).choice(
        len(x_normal_all), N_NORMAL, replace=False)
    x_normal = x_normal_all[normal_idx]

    augmented = {"Original only": None}
    for method, filename in METHODS.items():
        x, _ = read_npz(GENERATED / filename)
        assert len(x) >= N_GENERATED, (method, len(x))
        augmented[method] = x[:N_GENERATED]
    # All 1,000 saved masked-diffusion candidates pass the default FMD gate.
    # At this count the FMD training set is therefore identical to Masked Diffusion.
    augmented["FMD (1,000 candidates; all pass default gate)"] = augmented["Masked Diffusion"].copy()

    rows = [evaluate(name, augmented[name], x_normal, x_anomaly,
                     x_val, y_val, x_test, y_test) for name in augmented]
    pd.DataFrame(rows).to_csv(OUT / "generative_comparison_1000_f2.csv",
                              index=False, encoding="utf-8-sig")
    summary = {
        "sample_count": N_GENERATED,
        "normal_train_count": N_NORMAL,
        "real_anomaly_train_count": len(x_anomaly),
        "validation_counts": {"n": int(len(y_val)), "normal": int((y_val == 0).sum()), "anomaly": int((y_val == 1).sum())},
        "test_counts": {"n": int(len(y_test)), "normal": int((y_test == 0).sum()), "anomaly": int((y_test == 1).sum())},
        "threshold_selection": "Maximum validation F2; ties resolved by recall, then precision; thresholds 0.05 to 0.95 in 0.01 increments.",
        "fmd_note": "The saved pool contains exactly 1,000 Masked Diffusion candidates and all pass the default FMD gate; selecting 1,000 does not remove candidates, so this FMD row is the same training set as Masked Diffusion.",
        "generator_retraining": False,
        "results": rows,
    }
    (OUT / "generative_comparison_1000_f2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()
