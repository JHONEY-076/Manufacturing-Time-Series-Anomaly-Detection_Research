"""Matched count and RF training-composition experiments for section 4.4.3.

Reuses the saved generator and source-row-disjoint evaluation of section 4.4.1.
No source arrays or historical experiment results are modified.
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
from research_section441_matched_rf import source_rows, disjoint_mask, label_counts, metrics, RF_PARAMS
from research08_augmentation_saturation import select_threshold_by_validation_f1
from research09_filtering_strength import quality_filter_generated_anomalies_with_strength, FILTERING_STRENGTHS

OUT = ROOT / "output/section_4_4_3"
COUNTS = [0, 50, 100, 200, 500, 750, 1000]
NORMAL_COUNTS = [997, 1994, 5000, 9970, 19940]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parent_path = ROOT / "output/section_4_4_1/downstream_matched_750.json"
    quality_path = ROOT / "output/section_4_4_1/generation_quality_comparison.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert parent["rf_parameters"] == RF_PARAMS
    files = [Path(__file__), parent_path, quality_path,
             ROOT / "tools/research_section441_matched_rf.py",
             ROOT / "tools/research09_filtering_strength.py",
             ROOT / "tools/research08_augmentation_saturation.py"]
    data, meta = {}, {}
    for name in ["normal_train", "anomaly_train_seed", "validation", "final_test"]:
        path = ROOT / f"data/augmentation_split/{name}_windows.npz"
        with np.load(path, allow_pickle=True) as stored:
            data[name] = (stored["X"].astype(np.float32), stored["y"].astype(np.int8))
            assert stored["features"].tolist() == quality["features"]
        metadata_path = path.with_name(f"{name}_metadata.csv")
        meta[name] = pd.read_csv(metadata_path)
        np.testing.assert_array_equal(meta[name].label.to_numpy(), data[name][1])
        files.extend([path, metadata_path])

    train_rows = source_rows(pd.concat([meta["normal_train"], meta["anomaly_train_seed"]], ignore_index=True))
    val_keep = disjoint_mask(meta["validation"], train_rows)
    val_rows = source_rows(meta["validation"].loc[val_keep])
    used = {k: train_rows.get(k, set()) | val_rows.get(k, set()) for k in train_rows.keys() | val_rows.keys()}
    test_keep = disjoint_mask(meta["final_test"], used)
    test_rows = source_rows(meta["final_test"].loc[test_keep])
    for a, b in [(train_rows, val_rows), (train_rows, test_rows), (val_rows, test_rows)]:
        assert all(not a[k].intersection(b.get(k, set())) for k in a)
    with np.load(ROOT / "output/section_4_4_1/downstream_matched_750_predictions.npz") as old:
        np.testing.assert_array_equal(val_keep, old["validation_keep"])
        np.testing.assert_array_equal(test_keep, old["final_keep"])

    normal = data["normal_train"][0]
    anomaly = data["anomaly_train_seed"][0]
    assert len(anomaly) == 247
    val, y_val = data["validation"]
    test, y_test = data["final_test"]
    val, y_val, test, y_test = val[val_keep], y_val[val_keep], test[test_keep], y_test[test_keep]
    normal_base = np.asarray(parent["normal_indices"], dtype=np.int64)
    normal_remaining = np.setdiff1d(np.arange(len(normal)), normal_base)
    normal_extra = np.random.default_rng(66).choice(normal_remaining, max(NORMAL_COUNTS) - len(normal_base), replace=False)
    normal_order = np.concatenate([normal_base, normal_extra])
    assert len(np.unique(normal_order)) == len(normal_order)
    generated_path = ROOT / "data/research02/generated/diffusion_masked_windows.npz"
    files.append(generated_path)
    with np.load(generated_path, allow_pickle=True) as stored:
        pool = stored["X"].astype(np.float32)
    assert len(pool) == 1000
    md_base = np.asarray(quality["baseline_sample_indices"], dtype=np.int64)
    md_remaining = np.setdiff1d(np.arange(len(pool)), md_base)
    md_order = np.concatenate([md_base, np.random.default_rng(43).permutation(md_remaining)])
    ranked, diagnostics = quality_filter_generated_anomalies_with_strength(
        pool, anomaly, normal, len(pool), **FILTERING_STRENGTHS["default"])
    assert diagnostics["candidate_count"] == len(pool) and not diagnostics["fallback_used"]
    pool_lookup = {row.tobytes(): i for i, row in enumerate(pool)}
    assert len(pool_lookup) == len(pool), "Candidate identities must be unique for index reconstruction."
    fmd_order = np.array([pool_lookup[row.tobytes()] for row in ranked], dtype=np.int64)
    assert len(np.unique(fmd_order)) == len(pool)

    # At the full pool, align row order too: RF bootstrap draws are index-based.
    # At 750, retain exactly the ordered arrays used in section 4.4.1.
    def generated_indices(method, count):
        if count == 0:
            return np.empty(0, dtype=np.int64)
        return md_order[:count] if method == "Masking Diffusion" or count == len(pool) else fmd_order[:count]

    rows, cache, val_predictions, test_predictions = [], {}, [], []
    def evaluate(method, normal_count, count):
        key = (method, normal_count, count)
        if key in cache:
            return cache[key]
        start = time.monotonic()
        indices = generated_indices(method, count)
        blocks = [normal[normal_order[:normal_count]], anomaly]
        if count:
            blocks.append(pool[indices])
        train = np.concatenate(blocks)
        labels = np.concatenate([np.zeros(normal_count, dtype=np.int8), np.ones(len(train) - normal_count, dtype=np.int8)])
        assert len(train) == normal_count + len(anomaly) + count
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(train.reshape(len(train), -1), labels)
        val_proba = model.predict_proba(val.reshape(len(val), -1))[:, 1]
        test_proba = model.predict_proba(test.reshape(len(test), -1))[:, 1]
        choice = select_threshold_by_validation_f1(y_val, val_proba)
        row = {"run_id": len(rows), "method": method, "normal_count": normal_count,
               "real_anomaly_count": len(anomaly), "generated_count": count,
               "total_anomaly_count": len(anomaly) + count,
               "normal_to_anomaly_ratio": normal_count / (len(anomaly) + count),
               "anomaly_fraction": (len(anomaly) + count) / len(train),
               "generated_fraction_within_anomaly": count / (len(anomaly) + count),
               "total_train_count": len(train),
               **choice, **metrics(y_test, test_proba, choice["threshold"])}
        if normal_count == 5000 and count in [0, 750]:
            old_row = next(r for r in parent["results"] if r["method"] == method)
            for field in ["threshold", "precision", "recall", "f1", "auprc_ap", "tn", "fp", "fn", "tp"]:
                np.testing.assert_allclose(row[field], old_row[field], rtol=0, atol=1e-12)
        cache[key] = row
        rows.append(row)
        val_predictions.append(val_proba)
        test_predictions.append(test_proba)
        print(f"{method:18s} N={normal_count:5d} G={count:4d} ratio={row['normal_to_anomaly_ratio']:5.2f}: "
              f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} AP={row['auprc_ap']:.4f} "
              f"val_F1={row['validation_f1']:.4f} ({time.monotonic()-start:.1f}s)", flush=True)
        # Checkpoint each completed run without changing the inputs.
        (OUT / "runs_checkpoint.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return row

    count_rows = [evaluate("Original only", 5000, 0)]
    for count in COUNTS[1:]:
        for method in ["Masking Diffusion", "FMD"]:
            count_rows.append(evaluate(method, 5000, count))
    balance_rows = []
    for normal_count in NORMAL_COUNTS:
        for method in ["Original only", "Masking Diffusion", "FMD"]:
            balance_rows.append(evaluate(method, normal_count, 0 if method == "Original only" else 750))

    md_full = cache[("Masking Diffusion", 5000, 1000)]
    fmd_full = cache[("FMD", 5000, 1000)]
    np.testing.assert_allclose(test_predictions[md_full["run_id"]], test_predictions[fmd_full["run_id"]], rtol=0, atol=1e-14)
    for row in rows:
        assert row["tn"] + row["fp"] == 2785 and row["fn"] + row["tp"] == 295
        np.testing.assert_allclose(row["f1"], 2 * row["tp"] / (2 * row["tp"] + row["fp"] + row["fn"]), rtol=0, atol=1e-12)

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Single-seed matched sensitivity analysis of augmentation count and RF training composition.",
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "rf_parameters": RF_PARAMS, "generator_retrained": False,
        "count_experiment": {"normal_count": 5000, "generated_counts": COUNTS, "results": count_rows},
        "balance_experiment": {"generated_count": 750, "normal_counts": NORMAL_COUNTS, "results": balance_rows},
        "unique_run_count": len(rows), "all_runs": rows,
        "normal_order": normal_order.tolist(), "normal_extension_seed": 66,
        "md_candidate_order": md_order.tolist(), "md_extension_seed": 43,
        "fmd_candidate_order": fmd_order.tolist(), "filter_diagnostics_full_pool": diagnostics,
        "fmd_filter_parameters": FILTERING_STRENGTHS["default"],
        "fmd_reference": "Full original normal training pool and 247 real anomalies, fixed across every RF normal-count condition.",
        "selection_design": "Nested subsets. The original 4.4.1 ordered 750-sample MD/FMD arrays and 5000 normal array are preserved. At 1000, both methods use the entire same pool in the common MD order to prevent bootstrap row-order artifacts.",
        "retained_split": {"validation": label_counts(y_val), "final_test": label_counts(y_test)},
        "evaluation_exclusion_rule": parent["evaluation_exclusion_rule"],
        "threshold_selection": parent["threshold_selection"],
        "limitations": [
            "One RF seed and one saved generation pool; no confidence intervals or generator retraining variation.",
            "At fixed normal count, added anomaly count changes both training size and class composition.",
            "At fixed anomaly count, varying normal count changes both class composition and normal-data coverage; it is not a pure prior intervention.",
            "balanced_subsample reweights classes within each RF bootstrap; reported ratios are raw counts, not aggregate class weights.",
            "All 1000 candidates pass the default gates. Smaller FMD subsets are quality-ranked truncations; at 1000 FMD and MD use identical data.",
            "Full generator/filter references remain fixed even when fewer normal windows train the RF; this is not a total-data-scarcity experiment.",
            "Final-test maxima are descriptive; do not tune count or ratio using them.",
            "Source-row separation does not remove nearby-window or same-record dependence; the test is not new external data.",
        ],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (OUT / "balance_count_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "balance_count_predictions.npz", validation_probabilities=np.stack(val_predictions),
                        final_probabilities=np.stack(test_predictions), validation_labels=y_val, final_labels=y_test,
                        validation_keep=val_keep, final_keep=test_keep)
    print(f"Completed {len(rows)} matched RF runs. Shared 4.4.1 conditions reproduced; full-pool endpoints agree.", flush=True)


if __name__ == "__main__":
    main()
