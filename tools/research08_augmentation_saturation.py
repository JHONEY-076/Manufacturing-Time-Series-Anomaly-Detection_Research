from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from research05_balanced_aug_experiment import (
    comprehensive_augmentation,
    ensure_window_3d,
    flatten_windows,
    frequency_domain,
    load_npz_x,
    load_npz_xy,
    load_research02_generation_summary,
    magnitude_warp,
    noise_injection,
    quality_filter_generated_anomalies,
    sample_rows,
    score_filter_generated_anomalies,
    time_warp,
)


RANDOM_STATE = 42
NORMAL_TRAIN_SIZE = 5000
AUGMENTATION_COUNTS = [50, 100, 200, 500, 750, 1000]
THRESHOLDS = np.linspace(0.05, 0.95, 91)
FBETA_BETA = 2.0
PLATEAU_FRACTION = 0.95


def select_threshold_by_validation_f1(y_true: np.ndarray, anomaly_proba: np.ndarray) -> dict:
    best = None
    for threshold in THRESHOLDS:
        pred = (anomaly_proba >= threshold).astype(np.int8)
        candidate = {
            "threshold": float(threshold),
            "validation_precision": precision_score(y_true, pred, zero_division=0),
            "validation_recall": recall_score(y_true, pred, zero_division=0),
            "validation_f1": f1_score(y_true, pred, zero_division=0),
            "validation_f2": fbeta_score(y_true, pred, beta=FBETA_BETA, zero_division=0),
        }
        if best is None or (
            candidate["validation_f1"],
            candidate["validation_recall"],
            candidate["validation_precision"],
        ) > (
            best["validation_f1"],
            best["validation_recall"],
            best["validation_precision"],
        ):
            best = candidate
    if best is None:
        raise RuntimeError("No threshold candidates were evaluated.")
    return best


def evaluate_augmented_model(
    method: str,
    augmentation_family: str,
    augmentation_count: int,
    x_normal_train: np.ndarray,
    x_anomaly_seed: np.ndarray,
    x_augmented: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_final: np.ndarray,
    y_final: np.ndarray,
    seed: int,
) -> dict:
    x_normal_sample = sample_rows(x_normal_train, NORMAL_TRAIN_SIZE, seed=seed + 23)
    x_anomaly_train = np.concatenate([x_anomaly_seed, x_augmented], axis=0)
    x_train = np.concatenate([x_normal_sample, x_anomaly_train], axis=0)
    y_train = np.concatenate(
        [
            np.zeros(len(x_normal_sample), dtype=np.int8),
            np.ones(len(x_anomaly_train), dtype=np.int8),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(flatten_windows(x_train), y_train)

    val_proba = model.predict_proba(flatten_windows(x_val))[:, 1]
    threshold_result = select_threshold_by_validation_f1(y_val, val_proba)
    threshold = threshold_result["threshold"]

    final_proba = model.predict_proba(flatten_windows(x_final))[:, 1]
    final_pred = (final_proba >= threshold).astype(np.int8)

    return {
        "method": method,
        "augmentation_family": augmentation_family,
        "augmentation_count": int(augmentation_count),
        "normal_train_used": int(len(x_normal_sample)),
        "real_anomaly_seed_used": int(len(x_anomaly_seed)),
        "augmented_anomaly_used": int(len(x_augmented)),
        "total_anomaly_train_used": int(len(x_anomaly_train)),
        "normal_to_anomaly_ratio": float(len(x_normal_sample) / len(x_anomaly_train)),
        "threshold": threshold,
        "validation_precision": float(threshold_result["validation_precision"]),
        "validation_recall": float(threshold_result["validation_recall"]),
        "validation_f1": float(threshold_result["validation_f1"]),
        "validation_f2": float(threshold_result["validation_f2"]),
        "precision": precision_score(y_final, final_pred, zero_division=0),
        "recall": recall_score(y_final, final_pred, zero_division=0),
        "f1": f1_score(y_final, final_pred, zero_division=0),
        "f2": fbeta_score(y_final, final_pred, beta=FBETA_BETA, zero_division=0),
        "auroc": roc_auc_score(y_final, final_proba),
        "auprc": average_precision_score(y_final, final_proba),
        "false_negative": int(((y_final == 1) & (final_pred == 0)).sum()),
        "false_positive": int(((y_final == 0) & (final_pred == 1)).sum()),
    }


def build_augmented_sets(
    method: str,
    count: int,
    x_anomaly_seed: np.ndarray,
    x_normal_train: np.ndarray,
    masking_diffusion_x: np.ndarray,
    seed: int,
) -> tuple[str, np.ndarray]:
    if method == "Time warping":
        return "traditional", time_warp(x_anomaly_seed, count, seed)
    if method == "Magnitude warping":
        return "traditional", magnitude_warp(x_anomaly_seed, count, seed)
    if method == "Noise injection":
        return "traditional", noise_injection(x_anomaly_seed, count, seed)
    if method == "Frequency domain":
        return "traditional", frequency_domain(x_anomaly_seed, count, seed)
    if method == "Comprehensive":
        return "traditional", comprehensive_augmentation(x_anomaly_seed, count, seed)
    if method == "Masking Diffusion":
        return "generative", sample_rows(masking_diffusion_x, count, seed)
    if method == "Filtered Masking Diffusion":
        return (
            "generative_filtered",
            quality_filter_generated_anomalies(
                x_generated=masking_diffusion_x,
                x_anomaly_seed=x_anomaly_seed,
                x_normal_train=x_normal_train,
                n_select=count,
            ),
        )
    if method == "ScoreFiltered Masking Diffusion":
        return (
            "generative_score_filtered",
            score_filter_generated_anomalies(
                x_generated=masking_diffusion_x,
                x_anomaly_seed=x_anomaly_seed,
                x_normal_train=x_normal_train,
                n_select=count,
                seed=seed,
            ),
        )
    raise ValueError(f"Unknown method: {method}")


def make_plateau_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["f1", "f2", "recall", "auprc"]
    for method, method_df in results_df.groupby("method"):
        method_df = method_df.sort_values("augmentation_count")
        row = {
            "method": method,
            "augmentation_family": method_df["augmentation_family"].iloc[0],
        }
        for metric in metrics:
            max_score = float(method_df[metric].max())
            final_1000 = method_df.loc[method_df["augmentation_count"] == 1000, metric]
            final_1000_score = float(final_1000.iloc[0]) if not final_1000.empty else np.nan
            plateau_threshold = max_score * PLATEAU_FRACTION
            plateau_candidates = method_df[method_df[metric] >= plateau_threshold]
            row[f"{metric}_max"] = max_score
            row[f"{metric}_at_50"] = float(
                method_df.loc[method_df["augmentation_count"] == 50, metric].iloc[0]
            )
            row[f"{metric}_at_1000"] = final_1000_score
            row[f"{metric}_count_to_95pct_max"] = int(
                plateau_candidates["augmentation_count"].iloc[0]
            )
            row[f"{metric}_50_to_1000_gain"] = float(final_1000_score - row[f"{metric}_at_50"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["augmentation_family", "method"])


def save_plots(results_df: pd.DataFrame, plateau_df: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plot_df = results_df.melt(
        id_vars=["method", "augmentation_family", "augmentation_count"],
        value_vars=["f1", "f2", "recall", "auprc"],
        var_name="metric",
        value_name="score",
    )

    for metric in ["f1", "f2", "recall", "auprc"]:
        metric_df = plot_df[plot_df["metric"] == metric]
        plt.figure(figsize=(11, 6))
        sns.lineplot(
            data=metric_df,
            x="augmentation_count",
            y="score",
            hue="method",
            style="augmentation_family",
            marker="o",
        )
        plt.ylim(0, 1.05)
        plt.title(f"{metric.upper()} Saturation by Augmentation Count")
        plt.xlabel("Augmented anomaly windows used for training")
        plt.ylabel("Final test score")
        plt.tight_layout()
        plt.savefig(figure_dir / f"research08_{metric}_saturation.png", dpi=200)
        plt.close()

    plateau_plot_df = plateau_df.melt(
        id_vars=["method", "augmentation_family"],
        value_vars=["f1_count_to_95pct_max", "f2_count_to_95pct_max", "auprc_count_to_95pct_max"],
        var_name="metric",
        value_name="count_to_95pct_max",
    )
    plateau_plot_df["metric"] = plateau_plot_df["metric"].str.replace(
        "_count_to_95pct_max", "", regex=False
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=plateau_plot_df,
        x="method",
        y="count_to_95pct_max",
        hue="metric",
    )
    plt.xticks(rotation=35, ha="right")
    plt.title("Augmentation Count Needed to Reach 95% of Each Method's Maximum")
    plt.xlabel("")
    plt.ylabel("Augmented anomaly windows")
    plt.tight_layout()
    plt.savefig(figure_dir / "research08_count_to_95pct_max.png", dpi=200)
    plt.close()


def main() -> None:
    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    generated_dir = root / "data" / "research02" / "generated"
    result_dir = root / "data" / "research08" / "results"
    figure_dir = root / "data" / "research08" / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    generation_summary = load_research02_generation_summary(root)
    x_normal_train, _ = load_npz_xy(split_dir / "normal_train_windows.npz")
    x_anomaly_seed, _ = load_npz_xy(split_dir / "anomaly_train_seed_windows.npz")
    x_val, y_val = load_npz_xy(split_dir / "validation_windows.npz")
    x_final, y_final = load_npz_xy(split_dir / "final_test_windows.npz")
    masking_diffusion_x = ensure_window_3d(load_npz_x(generated_dir / "diffusion_masked_windows.npz"))

    methods = [
        "Time warping",
        "Magnitude warping",
        "Noise injection",
        "Frequency domain",
        "Comprehensive",
        "Masking Diffusion",
        "Filtered Masking Diffusion",
        "ScoreFiltered Masking Diffusion",
    ]

    rows = []
    for method_idx, method in enumerate(methods):
        for count in AUGMENTATION_COUNTS:
            seed = RANDOM_STATE + 1000 * (method_idx + 1) + count
            family, augmented_x = build_augmented_sets(
                method=method,
                count=count,
                x_anomaly_seed=x_anomaly_seed,
                x_normal_train=x_normal_train,
                masking_diffusion_x=masking_diffusion_x,
                seed=seed,
            )
            rows.append(
                evaluate_augmented_model(
                    method=method,
                    augmentation_family=family,
                    augmentation_count=count,
                    x_normal_train=x_normal_train,
                    x_anomaly_seed=x_anomaly_seed,
                    x_augmented=augmented_x,
                    x_val=x_val,
                    y_val=y_val,
                    x_final=x_final,
                    y_final=y_final,
                    seed=seed,
                )
            )

    results_df = pd.DataFrame(rows).sort_values(["method", "augmentation_count"])
    plateau_df = make_plateau_summary(results_df)

    best_by_count = results_df.loc[results_df.groupby("augmentation_count")["f2"].idxmax()]
    best_by_family_count = results_df.loc[
        results_df.groupby(["augmentation_family", "augmentation_count"])["f2"].idxmax()
    ].sort_values(["augmentation_count", "augmentation_family"])

    results_df.to_csv(result_dir / "research08_augmentation_saturation.csv", index=False)
    plateau_df.to_csv(result_dir / "research08_plateau_summary.csv", index=False)
    best_by_count.to_csv(result_dir / "research08_best_f2_by_count.csv", index=False)
    best_by_family_count.to_csv(result_dir / "research08_best_f2_by_family_count.csv", index=False)

    save_plots(results_df, plateau_df, figure_dir)

    summary = {
        "research_question": (
            "Do generative augmentations saturate earlier than traditional augmentations, "
            "or do traditional augmentations require more augmented samples to become effective?"
        ),
        "model": "RandomForestClassifier",
        "normal_train_used_each_method": NORMAL_TRAIN_SIZE,
        "real_anomaly_seed_used_each_method": int(len(x_anomaly_seed)),
        "augmentation_counts": AUGMENTATION_COUNTS,
        "threshold_selection": "Best F1 on real validation_windows only.",
        "plateau_definition": f"Smallest augmentation count reaching {PLATEAU_FRACTION:.0%} of each method's maximum score.",
        "methods": methods,
        "research02_generation_summary": generation_summary,
        "best_f2_by_count": best_by_count.to_dict(orient="records"),
        "plateau_summary": plateau_df.to_dict(orient="records"),
    }
    with open(result_dir / "research08_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Research08 augmentation saturation results")
    print(results_df)
    print()
    print("Plateau summary")
    print(plateau_df)
    print()
    print("Best F2 by count")
    print(best_by_count)


if __name__ == "__main__":
    main()
