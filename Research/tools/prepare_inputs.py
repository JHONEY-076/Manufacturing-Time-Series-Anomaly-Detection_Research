"""Rebuild standardized windows from raw sensor CSVs and recorded split membership."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["RealPower", "GateOnTime", "SetPower", "Speed", "Length"]


def prepare(output):
    split_out = output / "augmentation_split"
    split_out.mkdir()
    raw = ROOT / "data/raw_data"
    train = pd.read_csv(raw / "train/Training_Data.csv")
    # The historical normal-window indices refer to chronological training rows.
    train = train.assign(_time=pd.to_datetime(train.WorkingTime.str.strip())).sort_values("_time").drop(columns="_time").reset_index(drop=True)
    scaler = StandardScaler().fit(train[FEATURES].to_numpy(dtype=np.float64))
    cache = {"Training_Data.csv": train}
    counts = []
    for name in ["normal_train", "anomaly_train_seed", "validation", "final_test"]:
        meta = pd.read_csv(ROOT / "data/augmentation_split" / f"{name}_metadata.csv")
        x = np.empty((len(meta), 30, len(FEATURES)), dtype=np.float32)
        for source, group in meta.groupby("source_file", sort=False):
            if source not in cache:
                path = raw / ("train" if source == "Training_Data.csv" else "test") / source
                frame = pd.read_csv(path)
                frame.columns = frame.columns.str.strip()
                cache[source] = frame
            frame = cache[source]
            starts, ends = group.start_index.to_numpy(), group.end_index.to_numpy()
            if not np.array_equal(ends, starts + 29):
                raise ValueError("Unexpected window length")
            times = pd.to_datetime(frame.WorkingTime.str.strip())
            np.testing.assert_array_equal(times.iloc[starts].to_numpy(), pd.to_datetime(group.start_time).to_numpy())
            np.testing.assert_array_equal(times.iloc[ends].to_numpy(), pd.to_datetime(group.end_time).to_numpy())
            values = scaler.transform(frame[FEATURES].to_numpy(dtype=np.float64)).astype(np.float32)
            indices = starts[:, None] + np.arange(30)
            x[group.index] = values[indices]
            labels = frame.label.to_numpy() if "label" in frame else np.zeros(len(frame), dtype=np.int8)
            np.testing.assert_array_equal(labels[indices].max(axis=1), group.label.to_numpy())
        y = meta.label.to_numpy(dtype=np.int8)
        np.savez_compressed(split_out / f"{name}_windows.npz", X=x, y=y, features=np.array(FEATURES))
        meta.to_csv(split_out / f"{name}_metadata.csv", index=False)
        with np.load(ROOT / "data/augmentation_split" / f"{name}_windows.npz", allow_pickle=False) as reference:
            np.testing.assert_array_equal(x, reference["X"])
            np.testing.assert_array_equal(y, reference["y"])
        counts.append({"split": name, "windows": len(y), "normal": int((y == 0).sum()), "anomaly": int((y == 1).sum())})
        print(f"Rebuilt {name}: {x.shape}; matches bundled snapshot", flush=True)
    pd.DataFrame(counts).to_csv(split_out / "split_summary.csv", index=False)
    (output / "scaler.json").write_text(json.dumps({"features": FEATURES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "fit_source": "data/raw_data/train/Training_Data.csv", "split_membership": "Bundled metadata; historical membership retained, overlap excluded by comparison runner."}, indent=2) + "\n", encoding="utf-8")
