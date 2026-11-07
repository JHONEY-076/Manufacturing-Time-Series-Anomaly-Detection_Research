"""Run a real localhost Uvicorn server and replay original evaluation welding windows."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import numpy as np
import pandas as pd

from fmd_api.common import (
    ARTIFACT_DIR, MODEL_LABELS, MODEL_NAMES, RESULT_DIR, ROOT, classification_metrics,
    load_split, raw_split, scale_raw, sha256, write_json,
)


def statistics(values):
    data = np.asarray(values, dtype=float)
    return {"count": len(data), "mean": float(data.mean()), "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)), "p99": float(np.percentile(data, 99)),
            "max": float(data.max())}


def interval_audit():
    from fmd_api.common import SPLIT_DIR
    tables = {name: pd.read_csv(SPLIT_DIR / f"{name}_metadata.csv")
              for name in ("anomaly_train_seed", "validation", "final_test")}
    def samples(table):
        return {(row.source_file, i) for row in table.itertuples()
                for i in range(row.start_index, row.end_index + 1)}
    sets = {name: samples(table) for name, table in tables.items()}
    return {f"{a}_vs_{b}": len(sets[a] & sets[b])
            for a, b in [("anomaly_train_seed", "validation"), ("anomaly_train_seed", "final_test"),
                         ("validation", "final_test")]}


def make_report(summary, manifest):
    lines = [
        "# Research11: Baseline · Magnitude warping · FMD의 FastAPI 비교", "",
        f"실행 시각(UTC): {summary['created_utc']}", "",
        "Research01 baseline, Research05 Magnitude warping 1,000개(imbalanced), "
        "Research08 Filtered Masking Diffusion 750개 증강 RandomForest를 "
        "재학습해 저장하고, Uvicorn/FastAPI를 별도 프로세스로 실행해 실제 localhost HTTP 요청으로 검증했다. "
        "이 실험의 범위는 저장된 원본 제조 데이터의 윈도 단위 재생이다.", "",
        "## 데이터와 재현성", "",
        "- 원본: `data/raw_data/train/Training_Data.csv`와 `data/raw_data/test/`의 용접 센서 CSV.",
        "- 입력 순서: RealPower, GateOnTime, SetPower, Speed, Length. 요청당 원시 센서값 30 × 5개.",
        "- train CSV만으로 StandardScaler를 복원했으며, 기존 모든 split의 정규화 윈도와 최대 오차 0을 확인했다.",
        "- raw train CSV의 행 순서가 기존 윈도 생성 당시와 달라, 학습 데이터 검증에는 고유 타임스탬프를 사용했다.",
        "- 세 모델의 기존 결과 재현, 저장 후 재로딩, 원시 입력 API와 기존 정규화 입력의 예측 일치를 확인했다.",
        "- 정상 학습 5,000개·실제 이상 seed 247개·RF 300 trees는 공통이다. "
        "증강 수는 baseline 0개, Magnitude warping 1,000개, FMD 750개다. 각 기존 실험의 seed를 유지했다.",
        f"- 최종평가 {summary['replay']['requests']}개 윈도(정상 2,785 / 이상 296). Threshold는 validation에서만 선택했다.",
        f"- seed/validation/final_test 간 같은 파일의 원시 행 겹침: `{summary['raw_row_overlap']}`.",
        "- 기존 split에서 seed–validation 136개 행, validation–final_test 1개 행의 겹침이 발견되었다. "
        "따라서 split의 완전한 독립성은 확보되지 않았으며, 이 결과를 누수 없는 새로운 일반화 평가로 표현하면 안 된다.",
        "- 데이터·코드·모델 SHA-256 및 학습 설정은 `artifacts/research11/manifest.json`에 기록했다.", "",
        "## 실제 HTTP 추론 결과", "",
        "| 모델 | Precision | Recall | F1 | AUPRC | FN | FP | API·오프라인 판정 일치 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in MODEL_NAMES:
        metrics = summary["models"][name]["metrics"]
        label = MODEL_LABELS[name]
        lines.append(f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
                     f"{metrics['auprc']:.4f} | {metrics['fn']} | {metrics['fp']} | 100% |")
    replay = summary["replay"]
    paced = summary["paced_load"]
    lines.extend([
        "", "![세 모델 성능 및 오류 수](three_model_comparison.png)", "",
        f"`/compare`는 한 요청에서 세 모델을 순차 실행한다. 총 {replay['requests']}개 요청을 "
        f"원본 윈도의 종료 시각 순으로 가속 재생했으며 오류는 {replay['errors']}건이었다. "
        "원래의 수초~수시간 간격으로 대기한 실험은 아니다.", "",
        "| 측정 항목 | 평균(ms) | p95(ms) | p99(ms) | 최대(ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for label, stats in [("HTTP 왕복(세 모델 포함)", replay["http_round_trip_ms"])] + [
        (f"{MODEL_LABELS[name]} 정규화+추론", summary["models"][name]["inference_ms"]) for name in MODEL_NAMES
    ]:
        lines.append(f"| {label} | {stats['mean']:.3f} | {stats['p95']:.3f} | {stats['p99']:.3f} | {stats['max']:.3f} |")
    lines.extend(["", "각 모델의 `/predict/{model_name}` endpoint도 별도로 측정했다. "
                  "동일한 대표 윈도를 사용하고 모델 호출 순서를 교대했다. 아래는 해당 모델만 실행한 HTTP 왕복시간이다.", "",
                  "| 모델 | 요청 수 | 평균(ms) | p95(ms) | p99(ms) |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in MODEL_NAMES:
        stats = summary["single_model_http"][name]["http_round_trip_ms"]
        lines.append(f"| {MODEL_LABELS[name]} | {stats['count']} | {stats['mean']:.3f} | {stats['p95']:.3f} | {stats['p99']:.3f} |")
    durations = summary["window_span_seconds"]
    filtering = manifest["models"]["fmd"]["filtering"]
    lines.extend([
        "", f"별도 부하 검증: 초당 {paced['target_requests_per_second']:g}개 요청을 "
        f"{paced['requests']}회 예약해 전송했다. 예약시점부터 응답까지 p95는 "
        f"{paced['scheduled_to_response_ms']['p95']:.3f} ms, "
        f"{paced['deadline_ms']:.1f} ms 주기 초과는 {paced['deadline_misses']}건, "
        f"요청 오류는 {paced['errors']}건이었다. 이 주기는 검증용으로 지정했으며 설비의 확정된 SLA가 아니다.", "",
        f"동시 요청 검증: 클라이언트 4개로 {summary['concurrency']['requests']}회 요청했고 "
        f"오류 {summary['concurrency']['errors']}건, 예측 불일치 {summary['concurrency']['prediction_mismatches']}건이었다.", "",
        f"입력 계약 검증: {len(summary['contract_checks'])}개 확인 통과 "
        "(개별 모델 endpoint, 잘못된 행·열 개수, feature 순서, 비유한 값, 알 수 없는 모델, 추가 필드 등).", "",
        "## 해석 범위", "",
        "1. 실제 HTTP API 적용과 이 PC의 시험 부하에서의 응답 지연을 입증한다. 실제 공장 설비 연결, "
        "외부 네트워크 지연, 장기 연속 운전, 미지 분포·다른 설비에 대한 성능은 검증하지 않았다.",
        f"2. 윈도 30개를 모으는 시간은 별도다. 최종평가 윈도의 시간 길이는 중앙값 {durations['p50']:.3f}초, "
        f"최소 {durations['min']:.3f}초, 최대 {durations['max']:.3f}초다. "
        "API 왕복시간은 이미 모인 윈도에 대한 응답시간이며, 이상 발생부터 경보까지의 지연이 아니다. "
        "원본 데이터의 긴 시간 공백을 기존 연구와 동일하게 유지했다.",
        "3. 기존 세 실험을 재현했으므로 증강 수, normal sample seed와 RF seed가 다르다. "
        "이 API 실험만으로 성능 차이를 오직 증강 효과에 귀속하지 않는다. 최종평가 결과는 새 독립 시험이 아니라 기존 split의 재현이다.",
        f"4. 기존 FMD 필터를 그대로 사용했다. gate 통과 후보 {filtering['candidate_count']}개, "
        f"fallback_used={filtering['fallback_used']}. "
        + ("이번 실행에서는 1,000개 후보가 모두 gate를 통과했고, quality score 순위로 750개를 선택했다."
           if not filtering['fallback_used'] and filtering['candidate_count'] == 1000
           else "후보가 목표보다 적을 때는 기존 구현에 따라 전체 후보의 quality score 순위로 선택하므로, 모든 선택 샘플의 gate 통과를 보장하지 않는다."),
        "5. 저장한 세 joblib은 이상탐지 RandomForest다. Diffusion 생성기 checkpoint는 원래 환경에 없었으며 "
        "이번에도 생성기를 재학습하지 않았다. 기존 생성 데이터와 필터로 선택한 750개 샘플을 보존했다.",
        "6. 지표는 서로 겹치는 윈도 단위다. 독립 제품·불량 사건별 탐지율이나 통계적 유의성을 의미하지 않는다.", "",
        "## 논문에 사용할 수 있는 문장", "",
        "> 원본 데이터 기반 baseline, Magnitude warping 및 FMD 증강 데이터를 활용해 학습한 "
        "세 이상탐지 모델을 저장하고, "
        "FastAPI 기반 추론 시스템에 통합하였다. 저장된 실제 용접 공정 데이터를 시간 순서에 따라 "
        "윈도 단위로 재생하여 HTTP 요청을 통한 추론을 수행하였으며, 오프라인 결과와의 일치성과 "
        "응답 지연을 측정함으로써 실시간 추론을 위한 API 적용 가능성을 검증하였다.", "",
        "## 실행 및 증빙", "",
        "```powershell", ".\\.venv-api\\Scripts\\python.exe tools/research11_export_api_models.py",
        ".\\.venv-api\\Scripts\\python.exe tools/research11_verify_api.py",
        ".\\.venv-api\\Scripts\\python.exe -m uvicorn fmd_api.app:app --host 127.0.0.1 --port 8000",
        "```", "", "- API 문서: 서버 실행 중 http://127.0.0.1:8000/docs",
        "- `api_verification_summary.json`: 머신 판독 가능한 종합 결과와 실행 환경.",
        "- `api_predictions.csv`: 원본 파일·시각·정답·세 모델 예측·HTTP 지연 기록.",
        "- `three_model_comparison.csv`, `.png`, `.svg`: 모델별 성능·응답시간 비교표와 그림.",
        "- `single_model_requests.csv`: 각 모델 endpoint의 개별 HTTP 측정·예측 검증 기록.",
        "- `paced_requests.csv`, `concurrent_requests.csv`: 부하 검증 요청별 기록.",
        "- `example_request.json`, `example_response.json`: 실제 입력과 응답 예시.",
        "- `server.log`: Uvicorn 실행·요청 로그. `evidence_sha256.json`: 결과 파일 해시.", "",
        "서버 시작 시 모델을 한 번 로드하는 구조는 "
        "[FastAPI 공식 Lifespan 문서](https://fastapi.tiangolo.com/advanced/events/)를 따른다.",
        "측정값은 위 로컬 실험의 결과이며, 공식 문서는 성능 수치의 근거가 아니다.", "",
        f"실행 환경: {summary['environment']}", "",
    ])
    return "\n".join(lines)


def save_comparison(summary, manifest):
    rows = []
    for name in MODEL_NAMES:
        result = summary["models"][name]
        standalone = summary["single_model_http"][name]["http_round_trip_ms"]
        rows.append({"model": name, "label": MODEL_LABELS[name],
                     "augmentation_count": manifest["models"][name]["training"]["generated_anomaly_windows"],
                     "threshold": manifest["models"][name]["threshold"], **result["metrics"],
                     "prediction_mismatches": result["prediction_mismatches"],
                     "inference_mean_ms": result["inference_ms"]["mean"],
                     "inference_p95_ms": result["inference_ms"]["p95"],
                     "single_http_mean_ms": standalone["mean"], "single_http_p95_ms": standalone["p95"]})
    table = pd.DataFrame(rows)
    table.to_csv(RESULT_DIR / "three_model_comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#64748b", "#e09f3e", "#147d92"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    x = np.arange(3)
    for index, metric in enumerate(("precision", "recall", "f1")):
        bars = axes[0].bar(x + (index - 1) * 0.25, table[metric], width=0.24,
                           label=metric.capitalize(), color=colors[index])
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set(xticks=x, xticklabels=["Baseline", "Magnitude\nwarping 1000", "FMD 750"],
                ylim=(0, 1.12), ylabel="Score", title="Detection performance")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    for index, metric in enumerate(("fn", "fp")):
        bars = axes[1].bar(x + (index - 0.5) * 0.32, table[metric], width=0.3,
                           label={"fn": "False negatives", "fp": "False positives"}[metric],
                           color=["#bb4430", "#4c78a8"][index])
        axes[1].bar_label(bars, padding=3)
    axes[1].set(xticks=x, xticklabels=["Baseline", "Magnitude\nwarping 1000", "FMD 750"],
                ylim=(0, max(table.fn.max(), table.fp.max()) * 1.25), ylabel="Windows", title="Error counts")
    axes[1].legend(loc="upper right", fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same 3,081 raw welding windows replayed through FastAPI")
    fig.savefig(RESULT_DIR / "three_model_comparison.png", dpi=220)
    fig.savefig(RESULT_DIR / "three_model_comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--paced-requests", type=int, default=200)
    parser.add_argument("--single-requests", type=int, default=100, help="Requests per individual model endpoint")
    args = parser.parse_args()
    if not np.isfinite(args.rate) or args.rate <= 0 or args.paced_requests < 1 or args.single_requests < 1:
        parser.error("rate, paced-requests, and single-requests must be positive and finite")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_DIR / "run_status.json", {"status": "running", "models": list(MODEL_NAMES),
                                               "started_utc": datetime.now(timezone.utc).isoformat()})
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Source changed since model export: {relative}")
    final, y, features = load_split("final_test")
    raw, metadata = raw_split("final_test", features)
    if not np.array_equal(scale_raw(raw, manifest["preprocessing"]), final):
        raise RuntimeError("Raw replay does not match final-test windows")
    if sha256(RESULT_DIR / "offline_predictions.csv") != manifest["offline_predictions_sha256"]:
        raise RuntimeError("Offline predictions changed since model export")
    offline = pd.read_csv(RESULT_DIR / "offline_predictions.csv", float_precision="round_trip")
    order = pd.to_datetime(metadata.end_time).sort_values(kind="stable").index.to_numpy()
    payloads = [{"feature_names": features, "window": raw[i].tolist()} for i in range(len(raw))]
    overlap = interval_audit()
    if any(overlap.values()):
        print(f"Existing split overlap detected; recording as a research limitation: {overlap}", flush=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = RESULT_DIR / "server.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "uvicorn", "fmd_api.app:app", "--host", "127.0.0.1", "--port", str(port)]
    rows, paced_rows, concurrent_rows, single_rows, contract = [], [], [], [], []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=url, timeout=30.0, trust_env=False) as client:
                ready_deadline = perf_counter() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited; inspect {log_path}")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    if perf_counter() > ready_deadline:
                        raise RuntimeError("Server startup timeout")
                    sleep(0.1)
                print(f"Real HTTP server ready: {url}", flush=True)
                response = client.get("/models")
                response.raise_for_status()
                if set(response.json()["models"]) != set(MODEL_NAMES):
                    raise RuntimeError("API does not expose exactly the expected three models")
                for name in MODEL_NAMES:
                    if response.json()["models"][name]["artifact_sha256"] != manifest["models"][name]["artifact_sha256"]:
                        raise RuntimeError("Server model does not match exported artifact")
                example_index = int(np.flatnonzero(y == 1)[0])
                sample = payloads[example_index]
                def check(label, path, payload, expected_status):
                    response = client.post(path, json=payload)
                    contract.append({"check": label, "status": response.status_code, "expected": expected_status})
                    if response.status_code != expected_status:
                        raise RuntimeError(f"Contract check failed: {label}: {response.text}")
                    return response
                for name in MODEL_NAMES:
                    response = check(f"predict_{name}", f"/predict/{name}", sample, 200)
                    if response.json()["prediction"] != int(offline.loc[example_index, f"{name}_prediction"]):
                        raise RuntimeError(f"Individual endpoint prediction differs: {name}")
                response = check("compare_model_set", "/compare", sample, 200)
                if set(response.json()["predictions"]) != set(MODEL_NAMES):
                    raise RuntimeError("Comparison is missing a model")
                check("unknown_model", "/predict/missing", sample, 404)
                check("missing_row", "/compare", {**sample, "window": sample["window"][:-1]}, 422)
                check("extra_row", "/compare", {**sample, "window": sample["window"] + sample["window"][:1]}, 422)
                check("missing_feature", "/compare", {**sample, "window": [row[:-1] for row in sample["window"]]}, 422)
                check("feature_order", "/compare", {**sample, "feature_names": features[::-1]}, 422)
                for label, bad_value in [("null", None), ("nan_string", "NaN"), ("overflow", 1e308)]:
                    values = [row.copy() for row in sample["window"]]
                    values[0][0] = bad_value
                    check(label, "/compare", {**sample, "window": values}, 422)
                check("label_not_an_input", "/compare", {**sample, "label": 1}, 422)
                for _ in range(20):
                    client.post("/compare", json=sample).raise_for_status()
                write_json(RESULT_DIR / "example_request.json", sample)
                replay_start = perf_counter()
                for sequence, index in enumerate(order):
                    request_start = perf_counter()
                    response = client.post("/compare", json=payloads[index])
                    elapsed = (perf_counter() - request_start) * 1000
                    row = {"sequence": sequence, "row_index": int(index),
                           **metadata.iloc[index].to_dict(), "http_status": response.status_code,
                           "http_round_trip_ms": elapsed, "response_utc": datetime.now(timezone.utc).isoformat()}
                    response.raise_for_status()
                    data = response.json()
                    if set(data["predictions"]) != set(MODEL_NAMES):
                        raise RuntimeError("Unexpected comparison response model set")
                    for name in MODEL_NAMES:
                        prediction = data["predictions"][name]
                        for key in ("anomaly_score", "threshold", "prediction", "inference_ms"):
                            row[f"{name}_{key}"] = prediction[key]
                    rows.append(row)
                    if index == example_index:
                        write_json(RESULT_DIR / "example_response.json", data)
                    if (sequence + 1) % 500 == 0:
                        print(f"Replayed {sequence + 1}/{len(order)} raw windows", flush=True)
                replay_seconds = perf_counter() - replay_start
                print(f"Replay complete in {replay_seconds:.2f}s; testing {args.rate:g} requests/s", flush=True)
                paced_start = perf_counter()
                for sequence in range(args.paced_requests):
                    scheduled = paced_start + sequence / args.rate
                    sleep(max(0, scheduled - perf_counter()))
                    start = perf_counter()
                    response = client.post("/compare", json=payloads[order[sequence % len(order)]])
                    end = perf_counter()
                    response.raise_for_status()
                    paced_rows.append({"sequence": sequence, "status": response.status_code,
                                       "scheduled_offset_ms": sequence / args.rate * 1000,
                                       "dispatch_lag_ms": (start - scheduled) * 1000,
                                       "http_round_trip_ms": (end - start) * 1000,
                                       "scheduled_to_response_ms": (end - scheduled) * 1000})
                print("Paced load complete; measuring each model endpoint", flush=True)
                single_indices = order[np.linspace(0, len(order) - 1, args.single_requests, dtype=int)]
                for sequence, index in enumerate(single_indices):
                    offset = sequence % len(MODEL_NAMES)
                    names = MODEL_NAMES[offset:] + MODEL_NAMES[:offset]
                    for name in names:
                        start = perf_counter()
                        response = client.post(f"/predict/{name}", json=payloads[index])
                        elapsed = (perf_counter() - start) * 1000
                        response.raise_for_status()
                        prediction = response.json()
                        single_rows.append({"sequence": sequence, "row_index": int(index), "model": name,
                                            "status": response.status_code, "http_round_trip_ms": elapsed,
                                            "prediction": prediction["prediction"],
                                            "anomaly_score": prediction["anomaly_score"],
                                            "prediction_mismatch": int(prediction["prediction"] != int(offline.loc[index, f"{name}_prediction"])),
                                            "abs_score_error": abs(prediction["anomaly_score"] - offline.loc[index, f"{name}_score"])})
            def concurrent_request(sequence):
                index = int(order[round(sequence * (len(order) - 1) / 39)])
                with httpx.Client(base_url=url, timeout=30.0, trust_env=False) as connection:
                    start = perf_counter()
                    response = connection.post("/compare", json=payloads[index])
                    elapsed = (perf_counter() - start) * 1000
                    response.raise_for_status()
                    mismatches = sum(response.json()["predictions"][name]["prediction"] !=
                                     int(offline.loc[index, f"{name}_prediction"]) for name in MODEL_NAMES)
                    return {"sequence": sequence, "status": response.status_code,
                            "http_round_trip_ms": elapsed, "prediction_mismatches": mismatches}
            with ThreadPoolExecutor(max_workers=4) as pool:
                concurrent_rows = list(pool.map(concurrent_request, range(40)))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    table = pd.DataFrame(rows)
    table.to_csv(RESULT_DIR / "api_predictions.csv", index=False)
    pd.DataFrame(paced_rows).to_csv(RESULT_DIR / "paced_requests.csv", index=False)
    pd.DataFrame(concurrent_rows).to_csv(RESULT_DIR / "concurrent_requests.csv", index=False)
    single_table = pd.DataFrame(single_rows)
    single_table.to_csv(RESULT_DIR / "single_model_requests.csv", index=False)
    aligned = table.set_index("row_index").loc[np.arange(len(y))]
    spans = (pd.to_datetime(metadata.end_time) - pd.to_datetime(metadata.start_time)).dt.total_seconds()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "verification_passed": True,
        "verification_scope": "artifact reproduction and HTTP prediction equivalence; latency is reported separately",
        "model_names": list(MODEL_NAMES),
        "mode": "raw CSV window replay over real localhost HTTP, accelerated chronological replay",
        "live_factory_connection": False, "raw_row_overlap": overlap,
        "split_independence_verified": not any(overlap.values()),
        "artifact_manifest_sha256": sha256(ARTIFACT_DIR / "manifest.json"),
        "environment": {"platform": platform.platform(), "processor": platform.processor(),
                        "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
                        "versions": {name: importlib.metadata.version(name) for name in
                                     ["fastapi", "pydantic", "uvicorn", "httpx", "numpy", "pandas", "scikit-learn", "joblib"]},
                        "server_workers": 1, "model_n_jobs": 1, "url": url},
        "window_span_seconds": {**statistics(spans), "min": float(spans.min())},
        "contract_checks": contract,
        "replay": {"requests": len(rows), "errors": sum(r["http_status"] != 200 for r in rows),
                   "warmup_requests_excluded": 20, "elapsed_seconds": replay_seconds,
                   "achieved_requests_per_second": len(rows) / replay_seconds,
                   "http_round_trip_ms": statistics(table.http_round_trip_ms)},
        "paced_load": {"requests": len(paced_rows), "target_requests_per_second": args.rate,
                       "deadline_ms": 1000 / args.rate,
                       "deadline_misses": sum(r["scheduled_to_response_ms"] > 1000 / args.rate for r in paced_rows),
                       "errors": sum(r["status"] != 200 for r in paced_rows),
                       "scheduled_to_response_ms": statistics([r["scheduled_to_response_ms"] for r in paced_rows]),
                       "http_round_trip_ms": statistics([r["http_round_trip_ms"] for r in paced_rows])},
        "concurrency": {"clients": 4, "requests": len(concurrent_rows),
                        "errors": sum(r["status"] != 200 for r in concurrent_rows),
                        "prediction_mismatches": sum(r["prediction_mismatches"] for r in concurrent_rows),
                        "http_round_trip_ms": statistics([r["http_round_trip_ms"] for r in concurrent_rows])},
        "models": {},
        "single_model_http": {},
    }
    for name in MODEL_NAMES:
        scores = aligned[f"{name}_anomaly_score"].to_numpy()
        mismatches = int(np.sum(aligned[f"{name}_prediction"].to_numpy() != offline[f"{name}_prediction"].to_numpy()))
        max_error = float(np.max(np.abs(scores - offline[f"{name}_score"].to_numpy())))
        summary["models"][name] = {
            "metrics": classification_metrics(y, scores, manifest["models"][name]["threshold"]),
            "prediction_mismatches": mismatches, "max_abs_score_difference": max_error,
            "inference_ms": statistics(table[f"{name}_inference_ms"]),
        }
        if mismatches or max_error > 1e-12:
            summary["verification_passed"] = False
        requests = single_table[single_table.model == name]
        summary["single_model_http"][name] = {
            "requests": len(requests), "prediction_mismatches": int(requests.prediction_mismatch.sum()),
            "max_abs_score_difference": float(requests.abs_score_error.max()),
            "http_round_trip_ms": statistics(requests.http_round_trip_ms),
        }
        if requests.prediction_mismatch.sum() or requests.abs_score_error.max() > 1e-12:
            summary["verification_passed"] = False
    summary["verification_passed"] &= summary["concurrency"]["prediction_mismatches"] == 0
    write_json(RESULT_DIR / "api_verification_summary.json", summary)
    if not summary["verification_passed"]:
        raise RuntimeError("API/offline equivalence failed; inspect summary")
    save_comparison(summary, manifest)
    (RESULT_DIR / "api_verification_report.md").write_text(make_report(summary, manifest), encoding="utf-8")
    write_json(RESULT_DIR / "run_status.json", {"status": "complete", "models": list(MODEL_NAMES),
                                               "completed_utc": summary["created_utc"],
                                               "artifact_manifest_sha256": summary["artifact_manifest_sha256"]})
    evidence = [p for p in RESULT_DIR.iterdir() if p.is_file() and p.name != "evidence_sha256.json"]
    evidence += [p for p in ARTIFACT_DIR.iterdir() if p.is_file()]
    evidence += list((ROOT / "fmd_api").glob("*.py")) + [Path(__file__).resolve(), ROOT / "fmd_api/README.md", ROOT / "requirements-api.txt"]
    write_json(RESULT_DIR / "evidence_sha256.json", {
        "created_utc": summary["created_utc"],
        "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in evidence},
    })
    print(json.dumps({"passed": summary["verification_passed"], "replay": summary["replay"],
                      "paced_load": summary["paced_load"], "models": summary["models"]}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(RESULT_DIR / "run_status.json", {"status": "failed", "models": list(MODEL_NAMES),
                                                   "error": str(error), "failed_utc": datetime.now(timezone.utc).isoformat()})
        raise
