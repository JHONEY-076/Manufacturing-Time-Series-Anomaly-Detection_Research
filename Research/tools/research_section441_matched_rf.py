"""Matched 750-window RF comparison with source-row-disjoint evaluation.

Retains existing training/generated pools and never modifies stored partitions.
The main table removes overlapping evaluation windows using metadata only and
selects operating thresholds by validation F1.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from research02_quality_comparison_table import FILES, statistics, pooled_window_acf
from research05_balanced_aug_experiment import quality_filter_generated_anomalies
from research08_augmentation_saturation import select_threshold_by_validation_f1


SPLIT = ROOT / "data/augmentation_split"
OUT = ROOT / "output/section_4_4_1"
NORMAL_SEED = 65
MODEL_SEED = 42
RF_PARAMS = {
    "n_estimators": 300, "min_samples_leaf": 2,
    "class_weight": "balanced_subsample", "random_state": MODEL_SEED,
    "n_jobs": 4,
}
def source_rows(frame):
    result = {}
    for row in frame.itertuples():
        result.setdefault(row.source_file, set()).update(range(int(row.start_index), int(row.end_index) + 1))
    return result


def disjoint_mask(frame, used):
    return np.array([
        not any(i in used.get(row.source_file, set()) for i in range(int(row.start_index), int(row.end_index) + 1))
        for row in frame.itertuples()
    ], dtype=bool)


def label_counts(y):
    return {"total": int(len(y)), "normal": int((y == 0).sum()), "anomaly": int((y == 1).sum())}


def metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "auprc_ap": float(average_precision_score(y, proba)),
        "auroc": float(roc_auc_score(y, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quality = json.loads((OUT / "generation_quality_comparison.json").read_text(encoding="utf-8"))
    data, meta = {}, {}
    files = [Path(__file__), OUT / "generation_quality_comparison.json"]
    for name in ["normal_train", "anomaly_train_seed", "validation", "final_test"]:
        path = SPLIT / f"{name}_windows.npz"
        with np.load(path, allow_pickle=True) as stored:
            data[name] = (stored["X"].astype(np.float32), stored["y"].astype(np.int8))
            assert stored["features"].tolist() == quality["features"]
        metadata_path = SPLIT / f"{name}_metadata.csv"
        meta[name] = pd.read_csv(metadata_path)
        assert len(meta[name]) == len(data[name][0])
        np.testing.assert_array_equal(meta[name]["label"].to_numpy(), data[name][1])
        files.extend([path, metadata_path])

    train_rows = source_rows(pd.concat([meta["normal_train"], meta["anomaly_train_seed"]], ignore_index=True))
    val_keep = disjoint_mask(meta["validation"], train_rows)
    clean_val_rows = source_rows(meta["validation"].loc[val_keep])
    train_and_val = {key: train_rows.get(key, set()) | clean_val_rows.get(key, set()) for key in train_rows.keys() | clean_val_rows.keys()}
    test_keep = disjoint_mask(meta["final_test"], train_and_val)
    clean_test_rows = source_rows(meta["final_test"].loc[test_keep])
    for left, right in [(train_rows, clean_val_rows), (train_rows, clean_test_rows), (clean_val_rows, clean_test_rows)]:
        assert all(not left[key].intersection(right.get(key, set())) for key in left)

    normal, _ = data["normal_train"]
    anomaly, _ = data["anomaly_train_seed"]
    val, y_val = data["validation"]
    test, y_test = data["final_test"]
    normal_indices = np.random.default_rng(NORMAL_SEED).choice(len(normal), 5000, replace=False)
    normal_selected = normal[normal_indices]
    generation_indices = np.asarray(quality["baseline_sample_indices"], dtype=np.int64)
    pools = {}
    for method, filename in FILES.items():
        path = ROOT / "data/research02/generated" / filename
        with np.load(path, allow_pickle=True) as stored:
            pools[method] = stored["X"].astype(np.float32)
        files.append(path)
    augmented = {"Original only": None}
    augmented.update({method: x[generation_indices] for method, x in pools.items()})
    augmented["FMD"] = quality_filter_generated_anomalies(pools["Masking Diffusion"], anomaly, normal, 750)

    ref_stats = statistics(anomaly)
    ref_acf = pooled_window_acf(anomaly)
    expected_quality = {row["method"]: row for row in quality["results"]}
    for method, values in augmented.items():
        if values is None:
            continue
        expected = expected_quality["FMD (default filter)" if method == "FMD" else method]
        actual = [float(np.abs(a - b).mean()) for a, b in zip(statistics(values), ref_stats)]
        actual.append(float(np.abs(pooled_window_acf(values) - ref_acf).mean()))
        np.testing.assert_allclose(actual, [expected[k] for k in ["mean_abs_diff", "std_abs_diff", "corr_abs_diff", "acf_abs_diff"]], rtol=1e-12, atol=1e-12)

    clean_split = {
        "validation": label_counts(y_val[val_keep]),
        "final_test": label_counts(y_test[test_keep]),
        "validation_removed": meta["validation"].loc[~val_keep].to_dict(orient="records"),
        "final_test_removed": meta["final_test"].loc[~test_keep].to_dict(orient="records"),
    }
    print("Source-row-disjoint evaluation:", {k: clean_split[k] for k in ["validation", "final_test"]}, flush=True)
    rows, legacy_rows, val_predictions, test_predictions = [], [], [], []
    for method, extra in augmented.items():
        start = time.monotonic()
        blocks = [normal_selected, anomaly] + ([] if extra is None else [extra])
        train_x = np.concatenate(blocks)
        train_y = np.concatenate([np.zeros(5000, dtype=np.int8), np.ones(len(train_x) - 5000, dtype=np.int8)])
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(train_x.reshape(len(train_x), -1), train_y)
        val_proba = model.predict_proba(val.reshape(len(val), -1))[:, 1]
        test_proba = model.predict_proba(test.reshape(len(test), -1))[:, 1]
        choice = select_threshold_by_validation_f1(y_val[val_keep], val_proba[val_keep])
        row = {"method": method, "normal_count": 5000, "real_anomaly_count": len(anomaly),
               "generated_count": 0 if extra is None else len(extra), **choice,
               **metrics(y_test[test_keep], test_proba[test_keep], choice["threshold"])}
        rows.append(row)
        old_choice = select_threshold_by_validation_f1(y_val, val_proba)
        legacy_rows.append({"method": method, **old_choice, **metrics(y_test, test_proba, old_choice["threshold"])})
        val_predictions.append(val_proba)
        test_predictions.append(test_proba)
        print(f"{method}: P={row['precision']:.5f}, R={row['recall']:.5f}, F1={row['f1']:.5f}, AP={row['auprc_ap']:.5f}, threshold={row['threshold']:.2f}; {time.monotonic()-start:.1f}s", flush=True)

    baseline = rows[0]["f1"]
    for row in rows:
        row["f1_difference_vs_original"] = row["f1"] - baseline
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Matched fixed-count RF comparison for section 4.4.1; single RF run per method using saved generation pools and validation-F1 threshold selection.",
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "rf_parameters": RF_PARAMS, "normal_sampling_seed": NORMAL_SEED,
        "normal_indices": normal_indices.tolist(),
        "same_augmented_arrays_as_quality_table": True,
        "generator_retrained": False,
        "generated_sampling_seed": quality["baseline_sampling_seed"],
        "fmd_filter_parameters": quality["fmd_filter_parameters"],
        "fmd_filter_reference": "Full original normal training pool and all 247 real anomaly seed windows, matching Table A.",
        "threshold_selection": "Retained validation windows only; maximize F1 over thresholds 0.05..0.95 in steps of 0.01; ties maximize recall then precision.",
        "evaluation_exclusion_rule": "Remove validation windows sharing any (source_file, original row index) with any original training/reference window. Remove final-test windows sharing any such row with training/reference or retained validation windows. No outcomes or model scores used for exclusions.",
        "original_split": {"validation": label_counts(y_val), "final_test": label_counts(y_test)},
        "retained_split": clean_split,
        "results": rows,
        "legacy_split_sensitivity_only": legacy_rows,
        "limitations": [
            "Single fixed RF seed and saved generation pools; no retraining variability or significance estimate.",
            "Source-row disjointness does not establish independence across nearby windows, manufacturing runs, or identical values from different source files.",
            "Historical model and method development used the existing dataset; this is not a newly collected external test.",
            "These results use a different RF seed/sample and evaluation protocol than historical Research08 augmentation curves and must not be spliced into those curves.",
        ],
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    (OUT / "downstream_matched_750_f1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "downstream_matched_750_f1_predictions.npz",
                        methods=np.array(list(augmented)), validation_probabilities=np.stack(val_predictions),
                        final_probabilities=np.stack(test_predictions), validation_labels=y_val, final_labels=y_test,
                        validation_keep=val_keep, final_keep=test_keep)
    lines = ["| 방법 | 생성 수 | Precision | Recall | F1 | F2 | AP | 임계값 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    maxima = {key: max(row[key] for row in rows) for key in ["precision", "recall", "f1", "f2", "auprc_ap"]}
    for row in rows:
        cells = [f"{row[key]:.4f}" if row[key] != maxima[key] else f"**{row[key]:.4f}**" for key in maxima]
        lines.append(f"| {row['method']} | {row['generated_count']} | " + " | ".join(cells) + f" | {row['threshold']:.2f} |")
    (OUT / "downstream_matched_750_f1_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
