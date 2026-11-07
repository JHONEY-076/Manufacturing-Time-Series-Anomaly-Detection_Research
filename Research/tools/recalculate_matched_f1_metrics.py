"""Recalculate matched RF metrics using validation-set F1 thresholds.

This reuses saved validation/test probabilities; it does not retrain models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "section_4_4_1"
THRESHOLDS = np.linspace(0.05, 0.95, 91)


def choose_validation_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> dict:
    best = None
    for threshold in THRESHOLDS:
        pred = (proba >= threshold).astype(np.int8)
        candidate = {
            "threshold": float(threshold),
            "validation_precision": float(precision_score(y_true, pred, zero_division=0)),
            "validation_recall": float(recall_score(y_true, pred, zero_division=0)),
            "validation_f1": float(f1_score(y_true, pred, zero_division=0)),
            "validation_f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
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
        raise RuntimeError("No validation thresholds were evaluated")
    return best


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "ap": float(average_precision_score(y_true, proba)),
        "auroc": float(roc_auc_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    predictions_path = OUT / "downstream_matched_750_predictions.npz"
    previous_json_path = OUT / "downstream_matched_750.json"
    with np.load(predictions_path) as saved:
        methods = saved["methods"].astype(str).tolist()
        val_probas = saved["validation_probabilities"]
        test_probas = saved["final_probabilities"]
        val_y = saved["validation_labels"]
        test_y = saved["final_labels"]
        val_keep = saved["validation_keep"].astype(bool)
        test_keep = saved["final_keep"].astype(bool)

    val_y, test_y = val_y[val_keep], test_y[test_keep]
    previous = json.loads(previous_json_path.read_text(encoding="utf-8"))
    previous_rows = {row["method"]: row for row in previous["results"]}
    rows = []
    for idx, method in enumerate(methods):
        operating_point = choose_validation_f1_threshold(val_y, val_probas[idx, val_keep])
        final_metrics = evaluate(test_y, test_probas[idx, test_keep], operating_point["threshold"])
        rows.append({
            "method": method,
            "generated_count": previous_rows[method]["generated_count"],
            **operating_point,
            **final_metrics,
        })

    payload = dict(previous)
    payload.update({
        "purpose": "Matched fixed-count RF comparison recalculated with validation-F1 operating thresholds.",
        "threshold_selection": (
            "Retained validation windows only; maximize F1 over thresholds 0.05..0.95 "
            "in steps of 0.01; ties maximize recall, then precision."
        ),
        "recalculated_from_saved_predictions": True,
        "models_retrained": False,
        "source_predictions": str(predictions_path.relative_to(ROOT)),
        "results": rows,
    })
    json_path = OUT / "downstream_matched_750_f1.json"
    table_path = OUT / "downstream_matched_750_f1_table.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    headers = ["방법", "생성 수", "Precision", "Recall", "F1", "F2", "AP", "임계값", "FN", "FP"]
    metric_keys = ["precision", "recall", "f1", "f2", "ap"]
    maxima = {key: max(row[key] for row in rows) for key in metric_keys}
    lines = ["| " + " | ".join(headers) + " |", "|" + "---:|" * len(headers)]
    for row in rows:
        cells = [row["method"], str(row["generated_count"])]
        for key in metric_keys:
            cell = f"{row[key]:.4f}"
            if row[key] == maxima[key]:
                cell = f"**{cell}**"
            cells.append(cell)
        cells.extend([f"{row['threshold']:.2f}", str(row["fn"]), str(row["fp"])])
        lines.append("| " + " | ".join(cells) + " |")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Threshold selection: validation F1; ties by recall then precision")
    print("Final evaluation split:", len(test_y), "windows; normal", int((test_y == 0).sum()), "anomaly", int((test_y == 1).sum()))
    print("\n".join(lines))
    print("Saved:", json_path)
    print("Saved:", table_path)


if __name__ == "__main__":
    main()
