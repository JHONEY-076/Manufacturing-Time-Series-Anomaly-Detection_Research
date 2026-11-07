"""Run the baseline or main comparison without retraining saved generators."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

import research_section441_matched_rf as matched
from research08_augmentation_saturation import build_augmented_sets, select_threshold_by_validation_f1
from research05_balanced_aug_experiment import ensure_window_3d
from research09_filtering_strength import quality_filter_generated_anomalies_with_strength, FILTERING_STRENGTHS

ROOT = Path(__file__).resolve().parents[1]


def load_splits(split_dir):
    arrays, frames, inputs = {}, {}, []
    for name in ("normal_train", "anomaly_train_seed", "validation", "final_test"):
        path = split_dir / f"{name}_windows.npz"
        with np.load(path, allow_pickle=False) as stored:
            x, y = stored["X"].astype(np.float32), stored["y"].astype(np.int8)
            features = stored["features"].tolist()
        if x.shape != (len(y), 30, 5) or not np.isfinite(x).all():
            raise ValueError(f"Invalid input shape or values: {path}")
        if features != ["RealPower", "GateOnTime", "SetPower", "Speed", "Length"]:
            raise ValueError(f"Unexpected feature order: {path}")
        metadata_path = split_dir / f"{name}_metadata.csv"
        frame = pd.read_csv(metadata_path)
        np.testing.assert_array_equal(frame.label.to_numpy(), y)
        if not ((frame.end_index - frame.start_index) == 29).all():
            raise ValueError(f"Unexpected source intervals: {metadata_path}")
        arrays[name], frames[name] = (x, y), frame
        inputs.extend([path, metadata_path])
    if np.any(arrays["normal_train"][1] != 0) or np.any(arrays["anomaly_train_seed"][1] != 1):
        raise ValueError("normal_train must contain class 0 and anomaly_train_seed class 1")
    train_rows = matched.source_rows(pd.concat([frames["normal_train"], frames["anomaly_train_seed"]], ignore_index=True))
    val_keep = matched.disjoint_mask(frames["validation"], train_rows)
    val_rows = matched.source_rows(frames["validation"].loc[val_keep])
    used_rows = {k: train_rows.get(k, set()) | val_rows.get(k, set()) for k in train_rows.keys() | val_rows.keys()}
    test_keep = matched.disjoint_mask(frames["final_test"], used_rows)
    test_rows = matched.source_rows(frames["final_test"].loc[test_keep])
    for left, right in [(train_rows, val_rows), (train_rows, test_rows), (val_rows, test_rows)]:
        if any(left[k] & right.get(k, set()) for k in left):
            raise ValueError("Evaluation still shares source rows")
    for name, keep in [("validation", val_keep), ("final_test", test_keep)]:
        x, y = arrays[name]
        arrays[name] = x[keep], y[keep]
        if set(np.unique(y[keep])) != {0, 1}:
            raise ValueError(f"Both classes required in retained {name}")
    counts = {name: {"total": len(y), "normal": int((y == 0).sum()), "anomaly": int((y == 1).sum())} for name, (_, y) in arrays.items()}
    counts["excluded"] = {"validation": int((~val_keep).sum()), "final_test": int((~test_keep).sum())}
    return arrays, (val_keep, test_keep), counts, inputs


def run(stage, output, split_dir, trees=300, normal_count=5000, seed=42, jobs=4):
    arrays, (val_keep, test_keep), counts, inputs = load_splits(split_dir)
    normal_all = arrays["normal_train"][0]
    anomaly = arrays["anomaly_train_seed"][0]
    if normal_count > len(normal_all):
        raise ValueError("normal-count exceeds available normal windows")
    normal_indices = np.random.default_rng(65).choice(len(normal_all), normal_count, replace=False)
    normal = normal_all[normal_indices]
    augmented = {"Original only": None}
    if stage == "compare":
        quality_path = ROOT / "output/section_4_4_1/generation_quality_comparison.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        inputs.append(quality_path)
        pools = {}
        for method, filename in matched.FILES.items():
            path = ROOT / "data/research02/generated" / filename
            with np.load(path, allow_pickle=False) as stored:
                pools[method] = stored["X"].astype(np.float32)
            inputs.append(path)
        md = pools["Masking Diffusion"]
        for i, method in enumerate(["Time warping", "Magnitude warping", "Noise injection", "Frequency domain", "Comprehensive"]):
            augmented[method] = build_augmented_sets(method, 750, anomaly, normal_all, md, seed=17000 + i)[1]
        indices = np.asarray(quality["baseline_sample_indices"], dtype=np.int64)
        augmented.update({method: pool[indices] for method, pool in pools.items()})
        augmented["FMD"] = quality_filter_generated_anomalies_with_strength(md, anomaly, normal_all, 750, **FILTERING_STRENGTHS["default"])[0]
    x_val, y_val = arrays["validation"]
    x_test, y_test = arrays["final_test"]
    rows, val_predictions, test_predictions = [], [], []
    for method, extra in augmented.items():
        if extra is not None:
            extra = ensure_window_3d(extra)
        train = np.concatenate([normal, anomaly] + ([] if extra is None else [extra]))
        labels = np.concatenate([np.zeros(len(normal), dtype=np.int8), np.ones(len(train) - len(normal), dtype=np.int8)])
        model = RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=jobs)
        model.fit(train.reshape(len(train), -1), labels)
        val_proba = model.predict_proba(x_val.reshape(len(x_val), -1))[:, 1]
        test_proba = model.predict_proba(x_test.reshape(len(x_test), -1))[:, 1]
        threshold = select_threshold_by_validation_f1(y_val, val_proba)
        row = {"method": method, "generated_count": 0 if extra is None else len(extra), **threshold, **matched.metrics(y_test, test_proba, threshold["threshold"])}
        rows.append(row)
        val_predictions.append(val_proba)
        test_predictions.append(test_proba)
        print(f"{method}: F1={row['f1']:.6f}, AP={row['auprc_ap']:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "main_comparison.csv", index=False)
    np.savez_compressed(output / "predictions.npz", methods=np.array(list(augmented)), validation_labels=y_val, test_labels=y_test, validation_probabilities=np.stack(val_predictions), test_probabilities=np.stack(test_predictions), validation_keep=val_keep, test_keep=test_keep)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "rf_parameters": {"n_estimators": trees, "min_samples_leaf": 2, "class_weight": "balanced_subsample", "random_state": seed, "n_jobs": jobs},
        "normal_sampling_seed": 65, "normal_count": normal_count, "normal_indices": normal_indices.tolist(), "splits": counts,
        "generator_retrained": False, "threshold_selection": "Validation F1; ties higher recall then precision; thresholds 0.05..0.95 step 0.01",
        "limitations": ["Cached generator pools; no generator retraining uncertainty.", "Source-row disjointness does not establish independence of nearby production records."],
        "inputs_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "run.py", *Path(__file__).parent.glob("*.py")]},
    }
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
