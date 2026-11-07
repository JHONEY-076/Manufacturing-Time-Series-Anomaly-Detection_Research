"""Recompute confusion-matrix metrics in the public comparison using stdlib only."""
import csv
import math
from pathlib import Path


def verify():
    path = Path(__file__).resolve().parents[2] / "docs/results/main_comparison.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Comparison CSV has no result rows")
    for row in rows:
        tn, fp, fn, tp = (int(row[key]) for key in ("tn", "fp", "fn", "tp"))
        if min(tn, fp, fn, tp) < 0 or tn + fp + fn + tp != 3080:
            raise ValueError(f"Unexpected evaluation counts: {row['method']}")
        values = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "f2": 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0.0,
        }
        for key, expected in values.items():
            if not math.isclose(float(row[key]), expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Metric mismatch: {row['method']} / {key}")
    print(f"Verified Precision, Recall, F1 and F2 for {len(rows)} result rows.")
    print("This verifies aggregate arithmetic; it does not rerun models or verify AP/AUROC.")


if __name__ == "__main__":
    verify()
