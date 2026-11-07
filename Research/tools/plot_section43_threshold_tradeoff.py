"""Plot validation-selected F1/F2 threshold trade-offs for section 4.3."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "output/section_4_3_unified/main_predictions.npz"
OUT = ROOT / "output/section_4_3_unified"
METHOD = "Magnitude warping"


def main():
    data = np.load(PREDICTIONS)
    idx = list(data["methods"]).index(METHOD)
    y_val = data["validation_labels"].astype(np.int8)
    y_test = data["test_labels"].astype(np.int8)
    p_val = data["validation_probabilities"][idx]
    p_test = data["test_probabilities"][idx]
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

    curves = {key: [] for key in ("precision", "recall", "f1", "f2")}
    for threshold in thresholds:
        predicted = (p_val >= threshold).astype(np.int8)
        curves["precision"].append(precision_score(y_val, predicted, zero_division=0))
        curves["recall"].append(recall_score(y_val, predicted, zero_division=0))
        curves["f1"].append(f1_score(y_val, predicted, zero_division=0))
        curves["f2"].append(fbeta_score(y_val, predicted, beta=2, zero_division=0))

    def select(key):
        values = curves[key]
        best = max(range(len(thresholds)), key=lambda j: (values[j], curves["recall"][j], curves["precision"][j]))
        return float(thresholds[best])

    selected = {"Best F1": select("f1"), "Best F2": select("f2")}
    errors = {}
    for label, threshold in selected.items():
        predicted = (p_test >= threshold).astype(np.int8)
        errors[label] = {
            "threshold": threshold,
            "false_negative": int(((y_test == 1) & (predicted == 0)).sum()),
            "false_positive": int(((y_test == 0) & (predicted == 1)).sum()),
            "test_precision": float(precision_score(y_test, predicted, zero_division=0)),
            "test_recall": float(recall_score(y_test, predicted, zero_division=0)),
            "test_f1": float(f1_score(y_test, predicted, zero_division=0)),
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    colors = {"precision": "#2563eb", "recall": "#dc2626", "f1": "#111827", "f2": "#7c3aed"}
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1-score", "f2": "F2-score"}
    for key in curves:
        ax.plot(thresholds, curves[key], lw=2.2, color=colors[key], label=labels[key])
    for label, threshold in selected.items():
        color = colors["f1"] if label == "Best F1" else colors["f2"]
        ax.axvline(threshold, color=color, ls="--", alpha=0.65)
        y_val_marker = curves["f1" if label == "Best F1" else "f2"][int(round((threshold - 0.05) / 0.01))]
        ax.scatter([threshold], [y_val_marker], s=55, color=color, zorder=5)
        ax.annotate(f"{label}\nth={threshold:.2f}", (threshold, y_val_marker), xytext=(7, -31 if label == "Best F1" else 8), textcoords="offset points", color=color, fontsize=10)
    ax.set_xlim(0.68, 0.92)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Validation score")
    ax.set_title("Validation metrics by threshold\n(Magnitude warping)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    ax = axes[1]
    names = list(selected)
    x = np.arange(len(names))
    width = 0.34
    fn = [errors[n]["false_negative"] for n in names]
    fp = [errors[n]["false_positive"] for n in names]
    b1 = ax.bar(x - width / 2, fn, width, label="False negative", color="#ea580c")
    b2 = ax.bar(x + width / 2, fp, width, label="False positive", color="#0891b2")
    ax.bar_label(b1, padding=3)
    ax.bar_label(b2, padding=3)
    ax.set_xticks(x, [f"{n}\n(th={selected[n]:.2f})" for n in names])
    ax.set_ylabel("Count on final test set")
    ax.set_title("Test errors at validation-selected thresholds")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / "threshold_tradeoff_magnitude_warping.png", dpi=220, bbox_inches="tight")
    (OUT / "threshold_tradeoff_magnitude_warping.json").write_text(
        json.dumps({"method": METHOD, "thresholds_selected_on": "validation", "error_counts_on": "final_test", "results": errors}, indent=2),
        encoding="utf-8",
    )
    for name, values in errors.items():
        print(name, values)


if __name__ == "__main__":
    main()
