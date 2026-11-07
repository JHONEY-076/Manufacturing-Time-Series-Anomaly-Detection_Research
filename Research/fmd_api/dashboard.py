"""Browser views and read-only access to local research examples and evidence."""
from functools import lru_cache
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd

from .common import RESULT_DIR, raw_split

STATIC_DIR = Path(__file__).parent / "static"
DOWNLOADS = {
    "report": ("api_verification_report.md", "text/markdown"),
    "comparison": ("three_model_comparison.csv", "text/csv"),
    "figure": ("three_model_comparison.png", "image/png"),
    "summary": ("api_verification_summary.json", "application/json"),
    "example": ("example_request.json", "application/json"),
}


@lru_cache(maxsize=1)
def examples(features: tuple):
    raw, meta = raw_split("final_test", list(features))
    order = pd.to_datetime(meta.end_time).sort_values(kind="stable").index.to_numpy()
    positions = {int(index): position for position, index in enumerate(order)}
    return raw, meta, order, positions


def install_dashboard(app: FastAPI):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="dashboard-assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/guide", include_in_schema=False)
    def guide():
        return FileResponse(STATIC_DIR / "guide.html", headers={"Cache-Control": "no-cache"})

    @app.get("/dashboard/catalog", tags=["Browser dashboard"])
    def catalog(request: Request):
        _, meta, order, _ = examples(tuple(request.app.state.manifest["feature_names"]))
        selected = []
        for label in (0, 1):
            indices = meta.index[meta.label == label].to_numpy()
            positions = np.linspace(0, len(indices) - 1, min(4, len(indices)), dtype=int)
            for index in indices[positions]:
                row = meta.loc[index]
                selected.append({"id": int(index), "label": int(label), "source_file": row.source_file,
                                 "end_time": row.end_time})
        return {"total_windows": len(meta), "samples": selected, "chronological_first_id": int(order[0])}

    @app.get("/dashboard/sample/{row_index}", tags=["Browser dashboard"])
    def sample(row_index: int, request: Request):
        features = request.app.state.manifest["feature_names"]
        raw, meta, order, positions = examples(tuple(features))
        if not 0 <= row_index < len(meta):
            raise HTTPException(status_code=404, detail="평가 윈도 번호가 범위를 벗어났습니다.")
        row = meta.iloc[row_index]
        return {
            "id": row_index,
            "metadata": {"source_file": row.source_file, "start_time": row.start_time,
                         "end_time": row.end_time, "label": int(row.label),
                         "start_index": int(row.start_index), "end_index": int(row.end_index)},
            "request": {"feature_names": features, "window": raw[row_index].tolist()},
            "next_ids": [int(index) for index in order[positions[row_index]:positions[row_index] + 10]],
        }

    @app.get("/dashboard/results", tags=["Browser dashboard"])
    def results(request: Request):
        path = RESULT_DIR / "api_verification_summary.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="저장된 검증 결과가 없습니다. 검증 스크립트를 먼저 실행하세요.")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("artifact_manifest_sha256") != request.app.state.manifest_sha256:
            raise HTTPException(status_code=409, detail="현재 모델과 검증 기록의 버전이 다릅니다. 다시 검증해주세요.")
        return result

    @app.get("/dashboard/download/{name}", tags=["Browser dashboard"])
    def download(name: str):
        if name not in DOWNLOADS:
            raise HTTPException(status_code=404, detail="Unknown download")
        filename, media_type = DOWNLOADS[name]
        path = RESULT_DIR / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="검증 결과 파일이 없습니다.")
        return FileResponse(path, media_type=media_type, filename=filename)
