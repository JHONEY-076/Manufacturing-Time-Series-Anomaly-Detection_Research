"""Re-run section 4.3 performance experiments on one source-row-disjoint split.

Uses saved generative pools, regenerates transformation-based samples from the
247 anomaly seeds, applies one retained validation/test split, and records all
conditions in a new output directory. Historical results are left untouched.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score

import research_section441_matched_rf as matched
from research05_balanced_aug_experiment import ensure_window_3d, sample_rows
from research08_augmentation_saturation import build_augmented_sets, select_threshold_by_validation_f1
from research09_filtering_strength import FILTERING_STRENGTHS, quality_filter_generated_anomalies_with_strength


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/augmentation_split"
GENERATED = ROOT / "data/research02/generated"
MASK_GENERATED = ROOT / "data/research10/generated"
QUALITY_PATH = ROOT / "output/section_4_4_1/generation_quality_comparison.json"
OUT = ROOT / "output/section_4_3_unified"

COUNT = 750
NORMAL_N = 5000
NORMAL_SEED = 65
MODEL_SEED = 42
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2, "class_weight": "balanced_subsample", "n_jobs": 4}
METHODS = ["Original only", "Time warping", "Magnitude warping", "Noise injection", "Frequency domain", "Comprehensive", "GT-GAN", "Diffusion", "Masking GT-GAN", "Masking Diffusion", "FMD"]


def load_xy(path: Path):
    with np.load(path, allow_pickle=True) as data:
        key_x = "X" if "X" in data else "x"
        key_y = "y" if "y" in data else None
        x = data[key_x].astype(np.float32)
        y = data[key_y].astype(np.int8) if key_y else None
        features = data["features"].tolist() if "features" in data else None
    return x, y, features


def label_counts(y):
    return {"n": int(len(y)), "normal": int((y == 0).sum()), "anomaly": int((y == 1).sum())}


def calc_metrics(y, probabilities, threshold):
    prediction = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "f2": float(fbeta_score(y, prediction, beta=2, zero_division=0)),
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc_ap": float(average_precision_score(y, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_eval(x_normal, x_anomaly, x_augmented, x_val, y_val, x_test, y_test, model_seed):
    blocks = [x_normal, x_anomaly]
    if x_augmented is not None and len(x_augmented):
        blocks.append(ensure_window_3d(x_augmented))
    x_train = np.concatenate(blocks, axis=0)
    y_train = np.concatenate([np.zeros(len(x_normal), dtype=np.int8), np.ones(len(x_train) - len(x_normal), dtype=np.int8)])
    model = RandomForestClassifier(**RF_PARAMS, random_state=model_seed)
    model.fit(x_train.reshape(len(x_train), -1), y_train)
    val_proba = model.predict_proba(x_val.reshape(len(x_val), -1))[:, 1]
    test_proba = model.predict_proba(x_test.reshape(len(x_test), -1))[:, 1]
    threshold = select_threshold_by_validation_f1(y_val, val_proba)
    return {**threshold, **calc_metrics(y_test, test_proba, threshold["threshold"])}, val_proba, test_proba


def save_figures(out_dir, comparison_rows, balance_rows, filter_rows, mask_rows, stability_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "Original only": "Original",
        "Time warping": "Time warp",
        "Magnitude warping": "Magnitude",
        "Noise injection": "Noise",
        "Frequency domain": "Frequency",
        "Comprehensive": "Comprehensive",
        "GT-GAN": "GT-GAN",
        "Diffusion": "Diffusion",
        "Masking GT-GAN": "Masked GT-GAN",
        "Masking Diffusion": "Masked Diffusion",
        "FMD": "FMD",
    }
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("auprc_ap", "AUPRC")]
    names = [labels[row["method"]] for row in comparison_rows]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [row[key] for row in comparison_rows]
        bars = ax.bar(range(len(values)), values, color=["#718096" if i == 0 else ("#c53030" if row["method"] == "FMD" else "#3182ce") for i, row in enumerate(comparison_rows)])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    fig.suptitle("Anomaly detection performance at 750 augmented windows")
    fig.savefig(out_dir / "main_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected = ["Original only", "Magnitude warping", "Noise injection", "Masking Diffusion", "FMD"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, metric_key, title in [(axes[0], "f1", "F1-score"), (axes[1], "auprc_ap", "AUPRC")]:
        x = np.arange(len(selected)); width = 0.36
        for offset, condition, color, label in [(-width/2, "balanced_1_to_1", "#4c78a8", "Balanced training"), (width/2, "imbalanced_5000_normal", "#f58518", "Imbalanced training")]:
            table = {row["method"]: row for row in balance_rows if row["condition"] == condition}
            vals = [table[m][metric_key] for m in selected]
            ax.bar(x + offset, vals, width, color=color, label=label)
        ax.set_title(title); ax.set_ylim(0, 1.05); ax.set_xticks(x, [labels[m] for m in selected], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Effect of training class composition")
    fig.savefig(out_dir / "class_balance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, rows, x_key, title in [(axes[0], filter_rows, "strength", "Filtering strength"), (axes[1], mask_rows, "mask_ratio", "Masking ratio")]:
        names = [row[x_key] for row in rows]; x = np.arange(len(names)); width = 0.24
        for offset, key, color in [(-width, "precision", "#4c78a8"), (0, "recall", "#54a24b"), (width, "f1", "#e45756")]:
            ax.bar(x + offset, [row[key] for row in rows], width, label=key.capitalize(), color=color)
        ax.set_title(title); ax.set_ylim(0, 1.05); ax.set_xticks(x, names); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("FMD parameter sensitivity at 750 generated windows")
    fig.savefig(out_dir / "fmd_parameter_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    methods = stability_summary["method"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for ax, mean_col, std_col, title in [(axes[0], "f1_mean", "f1_std", "F1-score"), (axes[1], "auprc_ap_mean", "auprc_ap_std", "AUPRC")]:
        ax.bar(range(len(methods)), stability_summary[mean_col], yerr=stability_summary[std_col], capsize=4, color="#3182ce")
        ax.set_title(title); ax.set_ylim(0, 1.05); ax.set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.suptitle("Thirty paired repetitions (mean ± standard deviation)")
    fig.savefig(out_dir / "stability_30_runs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))
    data, metadata, input_files = {}, {}, [Path(__file__), QUALITY_PATH]
    for name in ["normal_train", "anomaly_train_seed", "validation", "final_test"]:
        path = SPLIT / f"{name}_windows.npz"
        x, y, features = load_xy(path)
        metadata_path = SPLIT / f"{name}_metadata.csv"
        frame = pd.read_csv(metadata_path)
        if y is not None:
            np.testing.assert_array_equal(frame["label"].to_numpy(), y)
        if features is not None:
            assert features == quality["features"]
        data[name] = (x, y)
        metadata[name] = frame
        input_files.extend([path, metadata_path])

    train_rows = matched.source_rows(pd.concat([metadata["normal_train"], metadata["anomaly_train_seed"]], ignore_index=True))
    val_keep = matched.disjoint_mask(metadata["validation"], train_rows)
    retained_val_rows = matched.source_rows(metadata["validation"].loc[val_keep])
    used_rows = {key: train_rows.get(key, set()) | retained_val_rows.get(key, set()) for key in train_rows.keys() | retained_val_rows.keys()}
    test_keep = matched.disjoint_mask(metadata["final_test"], used_rows)
    test_rows = matched.source_rows(metadata["final_test"].loc[test_keep])
    for left, right in [(train_rows, retained_val_rows), (train_rows, test_rows), (retained_val_rows, test_rows)]:
        assert all(not left[key].intersection(right.get(key, set())) for key in left)

    x_normal_all, _ = data["normal_train"]
    x_anomaly, _ = data["anomaly_train_seed"]
    assert len(x_anomaly) == 247
    x_val_all, y_val_all = data["validation"]
    x_test_all, y_test_all = data["final_test"]
    x_val, y_val = x_val_all[val_keep], y_val_all[val_keep]
    x_test, y_test = x_test_all[test_keep], y_test_all[test_keep]
    normal_indices = np.random.default_rng(NORMAL_SEED).choice(len(x_normal_all), NORMAL_N, replace=False)
    x_normal = x_normal_all[normal_indices]

    with np.load(GENERATED / "diffusion_masked_windows.npz", allow_pickle=True) as pool_data:
        md_pool = pool_data["X"].astype(np.float32)
    sample_indices = np.asarray(quality["baseline_sample_indices"], dtype=np.int64)
    generation_pools = {}
    for method, filename in matched.FILES.items():
        path = GENERATED / filename
        with np.load(path, allow_pickle=True) as pool_data:
            generation_pools[method] = pool_data["X"].astype(np.float32)
        input_files.append(path)
    x_fmd = quality_filter_generated_anomalies_with_strength(md_pool, x_anomaly, x_normal_all, COUNT, **FILTERING_STRENGTHS["default"])[0]

    augmented = {"Original only": None}
    transform_names = ["Time warping", "Magnitude warping", "Noise injection", "Frequency domain", "Comprehensive"]
    for idx, method in enumerate(transform_names):
        _, values = build_augmented_sets(method, COUNT, x_anomaly, x_normal_all, md_pool, seed=17000 + idx)
        augmented[method] = values
    augmented.update({method: pool[sample_indices] for method, pool in generation_pools.items()})
    augmented["FMD"] = x_fmd

    comparison_rows = []
    predictions = {}
    for idx, method in enumerate(METHODS):
        metrics, val_proba, test_proba = fit_eval(x_normal, x_anomaly, augmented[method], x_val, y_val, x_test, y_test, MODEL_SEED)
        comparison_rows.append({"method": method, "generated_count": 0 if augmented[method] is None else len(augmented[method]), **metrics})
        predictions[method] = {"validation": val_proba, "test": test_proba}
        print(f"main {method}: F1={metrics['f1']:.4f}, AP={metrics['auprc_ap']:.4f}", flush=True)

    # Compare matched methods under balanced and imbalanced training counts.
    # The same nested normal sample, anomaly seed, augmentation arrays, validation,
    # test set, and RF seed are used; only the number of normal training windows
    # changes (997 vs. 5000).
    balance_rows = []
    for condition in ["balanced_1_to_1", "imbalanced_5000_normal"]:
        for method in METHODS:
            generated_count = 0 if augmented[method] is None else len(augmented[method])
            normal_count = len(x_anomaly) + generated_count if condition == "balanced_1_to_1" else NORMAL_N
            condition_normal = x_normal_all[normal_indices[:normal_count]] if normal_count <= NORMAL_N else sample_rows(x_normal_all, normal_count, NORMAL_SEED)
            row, _, _ = fit_eval(condition_normal, x_anomaly, augmented[method], x_val, y_val, x_test, y_test, MODEL_SEED)
            balance_rows.append({"condition": condition, "normal_count": normal_count,
                                 "normal_to_anomaly_ratio": normal_count / (len(x_anomaly) + generated_count),
                                 "method": method, "generated_count": generated_count, **row})
        print(f"class balance {condition} complete", flush=True)

    filter_rows = []
    for idx, strength in enumerate(["loose", "default", "strict"]):
        selected, diagnostics = quality_filter_generated_anomalies_with_strength(md_pool, x_anomaly, x_normal_all, COUNT, **FILTERING_STRENGTHS[strength])
        row, _, _ = fit_eval(x_normal, x_anomaly, selected, x_val, y_val, x_test, y_test, MODEL_SEED)
        filter_rows.append({"strength": strength, "generated_count": len(selected), **row, **diagnostics})
        print(f"filter {strength}: F1={row['f1']:.4f}, R={row['recall']:.4f}, FN={row['fn']}, FP={row['fp']}", flush=True)

    mask_rows = []
    for mask_ratio in ["small", "default", "large"]:
        path = MASK_GENERATED / f"diffusion_masked_{mask_ratio}_windows.npz"
        with np.load(path, allow_pickle=True) as pool_data:
            pool = pool_data["x" if "x" in pool_data else "X"].astype(np.float32)
        input_files.append(path)
        selected, diagnostics = quality_filter_generated_anomalies_with_strength(pool, x_anomaly, x_normal_all, COUNT, **FILTERING_STRENGTHS["loose"])
        row, _, _ = fit_eval(x_normal, x_anomaly, selected, x_val, y_val, x_test, y_test, MODEL_SEED)
        mask_rows.append({"mask_ratio": mask_ratio, "generated_count": len(selected), **row, **diagnostics})
        print(f"mask {mask_ratio}: F1={row['f1']:.4f}, R={row['recall']:.4f}, FN={row['fn']}, FP={row['fp']}", flush=True)

    # Thirty paired runs quantify variability from normal subsampling, augmentation
    # randomness for magnitude warping, and RF seeds. Each run uses one shared
    # normal subset and RF seed across the four compared methods.
    stability_rows = []
    for run in range(30):
        run_normal = sample_rows(x_normal_all, NORMAL_N, seed=91000 + run)
        run_mag = build_augmented_sets("Magnitude warping", COUNT, x_anomaly, x_normal_all, md_pool, seed=92000 + run)[1]
        rng = np.random.default_rng(93000 + run)
        md_indices = rng.choice(len(md_pool), COUNT, replace=False)
        run_md = md_pool[md_indices]
        for method, extra in [("Original only", None), ("Magnitude warping", run_mag), ("Masking Diffusion", run_md), ("FMD", x_fmd)]:
            metrics, _, _ = fit_eval(run_normal, x_anomaly, extra, x_val, y_val, x_test, y_test, 94000 + run)
            stability_rows.append({"run": run, "method": method, **metrics})
        if (run + 1) % 5 == 0:
            print(f"stability {run + 1}/30 complete", flush=True)

    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Unified section 4.3 performance rerun on a source-row-disjoint validation/test split.",
        "rf_parameters": {**RF_PARAMS, "main_comparison_seed": MODEL_SEED},
        "normal_train_n": NORMAL_N,
        "real_anomaly_seed_n": len(x_anomaly),
        "clean_validation": label_counts(y_val),
        "clean_final_test": label_counts(y_test),
        "split_exclusion": "Source-row-disjoint across training/reference, retained validation, and retained final-test windows using source_file and original row intervals.",
        "threshold_selection": "Validation F1 maximized over 0.05 to 0.95 in 0.01 steps; ties resolved by higher recall and then precision.",
        "main_comparison": comparison_rows,
        "class_balance_comparison": balance_rows,
        "filtering_strength": filter_rows,
        "masking_ratio": mask_rows,
        "stability_runs": stability_rows,
        "stability_design": "30 paired repetitions; one shared normal subsample and RF seed per repetition across methods; magnitude warping and MD pool samples vary by repetition; FMD uses fixed quality-ranked subset.",
        "limitations": ["Single existing manufacturing dataset and saved generator pools; no external test set or generator retraining.", "Source-row disjointness removes overlapping original rows but does not guarantee independence across nearby windows or production records.", "Stability standard deviations are descriptive across 30 paired training/model runs; they are not confidence intervals over independent production runs."],
    }
    (OUT / "section_4_3_unified_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(OUT / "main_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(balance_rows).to_csv(OUT / "class_balance_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filter_rows).to_csv(OUT / "filtering_strength.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mask_rows).to_csv(OUT / "masking_ratio.csv", index=False, encoding="utf-8-sig")
    stability = pd.DataFrame(stability_rows)
    summary = stability.groupby("method").agg({"precision": ["mean", "std"], "recall": ["mean", "std"], "f1": ["mean", "std"], "f2": ["mean", "std"], "auprc_ap": ["mean", "std"], "auroc": ["mean", "std"]})
    summary.to_csv(OUT / "stability_summary.csv", encoding="utf-8-sig")
    flat_summary = summary.copy()
    flat_summary.columns = [f"{metric}_{stat}" for metric, stat in flat_summary.columns]
    flat_summary = flat_summary.reset_index()
    save_figures(OUT, comparison_rows, balance_rows, filter_rows, mask_rows, flat_summary)
    np.savez_compressed(OUT / "main_predictions.npz", methods=np.array(METHODS), validation_labels=y_val, test_labels=y_test,
                        validation_keep=val_keep, test_keep=test_keep,
                        validation_probabilities=np.stack([predictions[m]["validation"] for m in METHODS]),
                        test_probabilities=np.stack([predictions[m]["test"] for m in METHODS]))
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files}
    (OUT / "source_sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved unified section 4.3 results to", OUT, flush=True)


if __name__ == "__main__":
    main()
