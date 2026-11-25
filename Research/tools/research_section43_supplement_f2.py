"""Re-evaluate count, balance, and FMD filtering experiments at max validation F2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/augmentation_split"
GEN = ROOT / "data/research02/generated"
OUT = ROOT / "output/section_4_3_unified/compare_1000_f2"
QUALITY = ROOT / "output/section_4_4_1/generation_quality_comparison.json"
N_NORMAL, N_REAL = 5000, 247
NORMAL_SEED, MODEL_SEED = 65, 42
PARAMS = dict(n_estimators=300, min_samples_leaf=2,
              class_weight="balanced_subsample", n_jobs=4)
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
STRENGTHS = {
    "Loose": (1.50, 0.50, 0.80),
    "Default": (1.25, 0.75, 0.90),
    "Strict": (1.00, 1.00, 0.95),
}


def npz(path):
    with np.load(path, allow_pickle=True) as d:
        return d["X" if "X" in d else "x"].astype(np.float32), d.get("y")


def source_rows(frame):
    result = {}
    for row in frame.itertuples():
        result.setdefault(row.source_file, set()).update(
            range(int(row.start_index), int(row.end_index) + 1))
    return result


def disjoint(frame, used):
    return np.array([
        not any(i in used.get(r.source_file, set())
                for i in range(int(r.start_index), int(r.end_index) + 1))
        for r in frame.itertuples()], dtype=bool)


def threshold_f2(y, p):
    candidates = []
    for t in THRESHOLDS:
        pred = (p >= t).astype(np.int8)
        candidates.append((float(fbeta_score(y, pred, beta=2, zero_division=0)),
                           float(recall_score(y, pred, zero_division=0)),
                           float(precision_score(y, pred, zero_division=0)), float(t)))
    f2, rec, prec, t = max(candidates)
    return t, f2, rec, prec


def score_pool(pool, anomaly, normal_ref):
    gen = pool.reshape(len(pool), -1).astype(np.float64)
    an = anomaly.reshape(len(anomaly), -1).astype(np.float64)
    no = normal_ref.reshape(len(normal_ref), -1).astype(np.float64)
    ref = np.concatenate([no, an])
    mean, std = ref.mean(0), ref.std(0)
    std[std < 1e-6] = 1
    gz, az, nz = (gen-mean)/std, (an-mean)/std, (no-mean)/std
    da = np.linalg.norm(gz-az.mean(0), axis=1)
    dn = np.linalg.norm(gz-nz.mean(0), axis=1)
    am, ast = an.mean(0), an.std(0)
    ast[ast < 1e-6] = 1
    within = (np.abs((gen-am)/ast) <= 3.0).mean(1)
    ar = np.percentile(np.linalg.norm(az-az.mean(0), axis=1), 95)
    nr = np.percentile(np.linalg.norm(nz-nz.mean(0), axis=1), 25)
    return da, dn, within, ar, nr


def select(pool, score_parts, n, strength="Default", weights=(1.0, 0.5, 10.0)):
    da, dn, within, ar, nr = score_parts
    ma, mn, mr = STRENGTHS[strength]
    eligible = (da <= ar*ma) & (dn >= nr*mn) & (within >= mr)
    ids = np.flatnonzero(eligible)
    fallback = len(ids) < n
    if fallback:
        ids = np.arange(len(pool))
    wa, wn, wr = weights
    q = -wa*da + wn*dn + wr*within
    chosen = ids[np.argsort(q[ids])[::-1][:n]]
    return pool[chosen], int(eligible.sum()), fallback


def evaluate(name, normal, anomaly, aug, xv, yv, xt, yt, extra=None):
    blocks = [normal, anomaly] + ([aug] if aug is not None else [])
    xx = np.concatenate(blocks)
    yy = np.r_[np.zeros(len(normal), np.int8), np.ones(len(xx)-len(normal), np.int8)]
    clf = RandomForestClassifier(**PARAMS, random_state=MODEL_SEED)
    clf.fit(xx.reshape(len(xx), -1), yy)
    pv = clf.predict_proba(xv.reshape(len(xv), -1))[:, 1]
    pt = clf.predict_proba(xt.reshape(len(xt), -1))[:, 1]
    t, val_f2, val_recall, val_precision = threshold_f2(yv, pv)
    pred = (pt >= t).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return {"method": name, "generated_count": len(aug) if aug is not None else 0,
            "threshold": t, "validation_f2": val_f2,
            "validation_recall": val_recall, "validation_precision": val_precision,
            "precision": float(precision_score(yt, pred, zero_division=0)),
            "recall": float(recall_score(yt, pred, zero_division=0)),
            "f1": float(fbeta_score(yt, pred, beta=1, zero_division=0)),
            "f2": float(fbeta_score(yt, pred, beta=2, zero_division=0)),
            "auroc": float(roc_auc_score(yt, pt)),
            "ap": float(average_precision_score(yt, pt)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), **(extra or {})}


def main():
    data, meta = {}, {}
    for key in ["normal_train", "anomaly_train_seed", "validation", "final_test"]:
        data[key] = npz(SPLIT/f"{key}_windows.npz")
        meta[key] = pd.read_csv(SPLIT/f"{key}_metadata.csv")
    tr = source_rows(pd.concat([meta["normal_train"], meta["anomaly_train_seed"]], ignore_index=True))
    vk = disjoint(meta["validation"], tr)
    vr = source_rows(meta["validation"].loc[vk])
    used = {k: tr.get(k, set()) | vr.get(k, set()) for k in tr.keys() | vr.keys()}
    tk = disjoint(meta["final_test"], used)
    normal_all, _ = data["normal_train"]
    anomaly, _ = data["anomaly_train_seed"]
    xv_all, yv_all = data["validation"]
    xt_all, yt_all = data["final_test"]
    xv, yv, xt, yt = xv_all[vk], yv_all[vk], xt_all[tk], yt_all[tk]
    ni = np.random.default_rng(NORMAL_SEED).choice(len(normal_all), N_NORMAL, replace=False)
    normal = normal_all[ni]
    md, _ = npz(GEN/"diffusion_masked_windows.npz")
    parts = score_pool(md, anomaly, normal_all)
    sample_indices = np.asarray(json.loads(QUALITY.read_text(encoding="utf-8"))["baseline_sample_indices"], dtype=int)

    counts = []
    for n in [0, 50, 100, 200, 500, 750, 1000]:
        aug = None if n == 0 else select(md, parts, n, "Default")[0]
        row = evaluate(f"FMD count {n}", normal, anomaly, aug, xv, yv, xt, yt,
                       {"target_count": n})
        counts.append(row)
        print("count", n, row["f2"], row["recall"], flush=True)

    balances = []
    for label in ["Balanced", "Imbalanced"]:
        for method, aug in [("Original only", None),
                            ("Masked Diffusion", md[sample_indices]),
                            ("FMD", select(md, parts, 750, "Default")[0])]:
            nnormal = (N_REAL + (0 if aug is None else len(aug))) if label == "Balanced" else N_NORMAL
            nm = normal_all[ni[:nnormal]]
            row = evaluate(method, nm, anomaly, aug, xv, yv, xt, yt,
                           {"training_condition": label, "normal_count": nnormal})
            balances.append(row)
            print("balance", label, method, row["f2"], row["recall"], flush=True)

    filtering = []
    random750 = md[sample_indices]
    filtering.append(evaluate("No filtering (random 750)", normal, anomaly, random750,
                              xv, yv, xt, yt, {"setting": "No filtering"}))
    for s in STRENGTHS:
        aug, candidates, fallback = select(md, parts, 750, s)
        filtering.append(evaluate(f"FMD {s}", normal, anomaly, aug, xv, yv, xt, yt,
                                  {"setting": s, "candidate_count": candidates,
                                   "fallback_used": fallback}))
    for label, weights in [
        ("Baseline", (1, .5, 10)),
        ("Anomaly distance emphasized", (2, .5, 10)),
        ("Normal distance emphasized", (1, 1, 10)),
        ("Anomaly range emphasized", (1, .5, 20)),
    ]:
        aug, candidates, fallback = select(md, parts, 750, "Default", weights)
        filtering.append(evaluate(f"FMD weights {label}", normal, anomaly, aug,
                                  xv, yv, xt, yt, {"setting": label,
                                  "weights": str(weights), "candidate_count": candidates,
                                  "fallback_used": fallback}))

    for filename, rows in [("count_f2.csv", counts), ("balance_f2.csv", balances),
                           ("filtering_f2.csv", filtering)]:
        pd.DataFrame(rows).to_csv(OUT/filename, index=False, encoding="utf-8-sig")
    meta_out = {"threshold_rule": "Maximum validation F2; ties resolved by recall, then precision; grid 0.05-0.95 by 0.01.",
                "test_n": int(len(yt)), "test_normal": int((yt == 0).sum()),
                "test_anomaly": int((yt == 1).sum()),
                "count_results": counts, "balance_results": balances,
                "filtering_results": filtering}
    (OUT/"supplement_f2.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved supplemental F2 results", flush=True)


if __name__ == "__main__":
    main()
