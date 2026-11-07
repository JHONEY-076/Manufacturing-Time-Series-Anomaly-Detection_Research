from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "research11"
RESULT_DIR = ROOT / "data" / "research11" / "results"
SPLIT_DIR = ROOT / "data" / "augmentation_split"
MODEL_NAMES = ("baseline", "magnitude_warping", "fmd")
MODEL_LABELS = {
    "baseline": "Baseline RF",
    "magnitude_warping": "Magnitude warping 1000 RF",
    "fmd": "FMD 750 RF",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_split(name: str):
    with np.load(SPLIT_DIR / f"{name}_windows.npz", allow_pickle=False) as data:
        return data["X"], data["y"], data["features"].tolist()


def scale_raw(raw: np.ndarray, scaler: dict) -> np.ndarray:
    # Match StandardScaler on float64 sensor values followed by float32 storage.
    return ((np.asarray(raw, dtype=np.float64) - np.asarray(scaler["mean"]))
            / np.asarray(scaler["scale"])).astype(np.float32)


def raw_split(name: str, features: list[str], window_size: int = 30):
    meta = pd.read_csv(SPLIT_DIR / f"{name}_metadata.csv")
    raw = np.empty((len(meta), window_size, len(features)), dtype=np.float64)
    for source, group in meta.groupby("source_file", sort=False):
        frame = pd.read_csv(ROOT / "data" / "raw_data" / "test" / source)
        values = frame[features].to_numpy(dtype=np.float64)
        starts = group.start_index.to_numpy()
        indices = starts[:, None] + np.arange(window_size)
        if not np.all(group.end_index.to_numpy() == starts + window_size - 1):
            raise ValueError("Unexpected window metadata")
        times = pd.to_datetime(frame.WorkingTime.str.strip())
        if not np.array_equal(times.iloc[starts].to_numpy(), pd.to_datetime(group.start_time).to_numpy()):
            raise ValueError(f"Raw start timestamps differ: {source}")
        if not np.array_equal(times.iloc[indices[:, -1]].to_numpy(), pd.to_datetime(group.end_time).to_numpy()):
            raise ValueError(f"Raw end timestamps differ: {source}")
        raw[group.index] = values[indices]
    return raw, meta


def classification_metrics(y, scores, threshold):
    predicted = (np.asarray(scores) >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
