"""Persist reproduced baseline, Magnitude warping@1000, and FMD@750 API bundles."""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from fmd_api.common import (
    ARTIFACT_DIR, MODEL_NAMES, RESULT_DIR, ROOT, SPLIT_DIR, classification_metrics,
    load_split, raw_split, scale_raw, sha256, write_json,
)
from research01_randomforest_baseline import BASELINE_PARAMS, select_threshold_by_validation_f1
from research05_balanced_aug_experiment import magnitude_warp, select_threshold_by_validation_f1 as select_magnitude_threshold
from research08_augmentation_saturation import select_threshold_by_validation_f1 as select_fmd_threshold
from research09_filtering_strength import FILTERING_STRENGTHS, quality_filter_generated_anomalies_with_strength


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    normal, _, features = load_split("normal_train")
    anomaly, _, _ = load_split("anomaly_train_seed")
    val, y_val, _ = load_split("validation")
    final, y_final, _ = load_split("final_test")
    train_path = ROOT / "data/raw_data/train/Training_Data.csv"
    frame = pd.read_csv(train_path)
    scaler = StandardScaler().fit(frame[features])
    scaler_spec = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "fit_source": "data/raw_data/train/Training_Data.csv", "fit_rows": len(frame),
                   "operation": "float64 standardization then float32 cast"}
    # The raw training CSV was reordered after window creation. Align by unique timestamps.
    frame["WorkingTime"] = frame.WorkingTime.str.strip()
    if frame.WorkingTime.duplicated().any():
        raise ValueError("Training timestamp alignment is ambiguous")
    meta = pd.read_csv(SPLIT_DIR / "normal_train_metadata.csv")
    timestamps = pd.concat([meta.start_time, meta.end_time.iloc[-29:]], ignore_index=True)
    ordered = frame.set_index("WorkingTime").loc[timestamps, features].to_numpy()
    scaled = scale_raw(ordered, scaler_spec)
    reconstructed = np.lib.stride_tricks.sliding_window_view(scaled, 30, axis=0).transpose(0, 2, 1)
    checks = {"normal_train": float(np.max(np.abs(reconstructed - normal)))}
    for name, expected in [("anomaly_train_seed", anomaly), ("validation", val), ("final_test", final)]:
        raw, _ = raw_split(name, features)
        checks[name] = float(np.max(np.abs(scale_raw(raw, scaler_spec) - expected)))
    if any(error != 0.0 for error in checks.values()):
        raise RuntimeError(f"Preprocessing reconstruction differs: {checks}")
    print("Raw preprocessing exactly matches all stored windows.", flush=True)

    generated_path = ROOT / "data/research02/generated/diffusion_masked_windows.npz"
    with np.load(generated_path, allow_pickle=False) as data:
        if data["features"].tolist() != features:
            raise ValueError("Generated feature order differs")
        generated = data["X"]
    filtered, diagnostics = quality_filter_generated_anomalies_with_strength(
        generated, anomaly, normal, 750, **FILTERING_STRENGTHS["default"],
    )
    np.savez_compressed(ARTIFACT_DIR / "fmd_selected_windows.npz", X=filtered, features=np.asarray(features))
    # Research05: augmentation seed = 42 + 102; RF seed = 42 + 100 * (method_idx + 1).
    magnitude = magnitude_warp(anomaly, n_generate=1000, seed=144)
    np.savez_compressed(ARTIFACT_DIR / "magnitude_warping_windows.npz", X=magnitude, features=np.asarray(features))
    augmented_sets = {"baseline": normal[:0], "magnitude_warping": magnitude, "fmd": filtered}
    seeds = {"baseline": (42, 42), "magnitude_warping": (242, 265), "fmd": (7792, 7815)}
    sources = [train_path, generated_path,
               ROOT / "data/research02/results/research02_generation_summary.json",
               ROOT / "tools/research01_randomforest_baseline.py",
               ROOT / "tools/research05_balanced_aug_experiment.py",
               ROOT / "tools/research08_augmentation_saturation.py",
               ROOT / "tools/research09_filtering_strength.py",
               ROOT / "fmd_api/common.py",
               ROOT / "data/research01/results/randomforest_original_baseline.csv",
               ROOT / "data/research05/results/research05_balanced_augmentation_comparison.csv",
               ROOT / "data/research08/results/research08_augmentation_saturation.csv",
               Path(__file__).resolve()]
    sources += sorted(SPLIT_DIR.glob("*.npz")) + sorted(SPLIT_DIR.glob("*_metadata.csv"))
    sources += sorted((ROOT / "data/raw_data/test").glob("*.csv"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "feature_names": features, "window_size": 30, "stride": 1,
        "preprocessing": scaler_spec, "raw_reconstruction_max_abs_error": checks,
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sources},
        "comparison_scope": "Reproduction of three existing runs on the same evaluation windows. Augmentation counts, normal sample seeds, and RF seeds differ; this is not a paired causal augmentation experiment.",
        "generator_checkpoint_available": False,
        "models": {},
    }
    predictions = pd.DataFrame({"row_index": np.arange(len(final)), "label": y_final})
    for name in MODEL_NAMES:
        # Research08 method_idx=6, count=750: 42 + 7000 + 750 = 7792.
        model_seed, sample_seed = seeds[name]
        indices = np.random.default_rng(sample_seed).choice(len(normal), size=5000, replace=False)
        blocks = [normal[indices], anomaly, augmented_sets[name]]
        x_train = np.concatenate(blocks)
        y_train = np.concatenate([np.zeros(5000, dtype=np.int8), np.ones(len(x_train) - 5000, dtype=np.int8)])
        model = RandomForestClassifier(**BASELINE_PARAMS, random_state=model_seed, n_jobs=-1)
        print(f"Training {name}: {len(x_train)} windows", flush=True)
        model.fit(x_train.reshape(len(x_train), -1), y_train)
        model.n_jobs = 1  # Avoid per-request parallel dispatch overhead; tree weights are unchanged.
        val_scores = model.predict_proba(val.reshape(len(val), -1))[:, 1]
        if name == "baseline":
            threshold = select_threshold_by_validation_f1(y_val, val_scores)[0]
        elif name == "magnitude_warping":
            threshold = select_magnitude_threshold(y_val, val_scores)["threshold"]
        else:
            threshold = select_fmd_threshold(y_val, val_scores)["threshold"]
        final_scores = model.predict_proba(final.reshape(len(final), -1))[:, 1]
        bundle = {"model": model, "threshold": threshold, "feature_names": features,
                  "window_size": 30, "scaler": scaler_spec, "model_name": name}
        path = ARTIFACT_DIR / f"{name}.joblib"
        joblib.dump(bundle, path, compress=3)
        restored = joblib.load(path)["model"].predict_proba(final.reshape(len(final), -1))[:, 1]
        if not np.array_equal(final_scores, restored):
            raise RuntimeError("Reload changed model predictions")
        metrics = classification_metrics(y_final, final_scores, threshold)
        if name == "baseline":
            historical = pd.read_csv(ROOT / "data/research01/results/randomforest_original_baseline.csv").iloc[0]
        elif name == "magnitude_warping":
            table = pd.read_csv(ROOT / "data/research05/results/research05_balanced_augmentation_comparison.csv")
            historical = table[(table.method == "Magnitude warping") & (table.condition == "imbalanced")].iloc[0]
        else:
            table = pd.read_csv(ROOT / "data/research08/results/research08_augmentation_saturation.csv")
            historical = table[(table.method == "Filtered Masking Diffusion") & (table.augmentation_count == 750)].iloc[0]
        historical_diff = {k: float(metrics[k] - historical[k]) for k in ("precision", "recall", "f1", "auroc", "auprc")}
        if any(abs(error) > 1e-12 for error in historical_diff.values()):
            raise RuntimeError(f"Historical reproduction failed for {name}: {historical_diff}")
        info = {
            "artifact_file": path.name, "artifact_sha256": sha256(path), "artifact_bytes": path.stat().st_size,
            "threshold": threshold, "threshold_selection": "validation F1 only, original experiment tie breaking",
            "training": {"normal_windows": 5000, "real_anomaly_windows": len(anomaly),
                         "generated_anomaly_windows": len(augmented_sets[name]),
                         "rf_seed": model_seed, "normal_sample_seed": sample_seed, "rf_params": BASELINE_PARAMS},
            "final_test_metrics": metrics, "historical_metric_difference": historical_diff,
            "reload_max_abs_score_error": float(np.max(np.abs(final_scores - restored))),
        }
        if name == "fmd":
            info["filtering"] = {"strength": "default", **FILTERING_STRENGTHS["default"], **diagnostics,
                                 "selected_windows_sha256": sha256(ARTIFACT_DIR / "fmd_selected_windows.npz")}
        elif name == "magnitude_warping":
            info["augmentation"] = {
                "method": "Magnitude warping", "source_experiment": "Research05 imbalanced",
                "seed": 144, "count": 1000, "sigma": 0.12, "n_knots": 4,
                "interpolation": "linear", "input_space": "existing standardized anomaly seed windows",
                "windows_sha256": sha256(ARTIFACT_DIR / "magnitude_warping_windows.npz"),
            }
        manifest["models"][name] = info
        predictions[f"{name}_score"] = final_scores
        predictions[f"{name}_prediction"] = (final_scores >= threshold).astype(np.int8)
        print(name, metrics, "historical differences", historical_diff, flush=True)
    predictions.to_csv(RESULT_DIR / "offline_predictions.csv", index=False)
    manifest["offline_predictions_sha256"] = sha256(RESULT_DIR / "offline_predictions.csv")
    write_json(ARTIFACT_DIR / "manifest.json", manifest)
    print(f"Saved models and manifest to {ARTIFACT_DIR}", flush=True)


if __name__ == "__main__":
    main()
