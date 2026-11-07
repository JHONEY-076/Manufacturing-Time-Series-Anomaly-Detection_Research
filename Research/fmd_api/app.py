from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .common import ARTIFACT_DIR, MODEL_NAMES, scale_raw, sha256
from .dashboard import install_dashboard

SensorRow = Annotated[list[FiniteFloat], Field(min_length=5, max_length=5)]


class WindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_names: Annotated[list[str], Field(min_length=5, max_length=5)]
    window: Annotated[list[SensorRow], Field(min_length=30, max_length=30)]


class Prediction(BaseModel):
    model: str
    artifact_sha256: str
    anomaly_score: float
    threshold: float
    prediction: int
    label: str
    inference_ms: float


class Comparison(BaseModel):
    predictions: dict[str, Prediction]
    processing_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = Path(os.environ.get("FMD_ARTIFACT_DIR", str(ARTIFACT_DIR)))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    bundles = {}
    for name in MODEL_NAMES:
        info = manifest["models"][name]
        path = directory / info["artifact_file"]
        if sha256(path) != info["artifact_sha256"]:
            raise RuntimeError(f"Artifact integrity check failed: {name}")
        # These are locally exported, trusted artifacts; no upload/deserialization endpoint.
        bundle = joblib.load(path)
        if bundle["feature_names"] != manifest["feature_names"] or bundle["window_size"] != 30:
            raise RuntimeError("Model schema differs from the API input contract")
        bundle["model"].n_jobs = 1
        bundles[name] = bundle
    app.state.bundles = bundles
    app.state.manifest = manifest
    app.state.manifest_sha256 = sha256(directory / "manifest.json")
    yield
    app.state.bundles.clear()


app = FastAPI(title="Baseline / Magnitude Warping / FMD Anomaly Detection", version="1.2.0", lifespan=lifespan)
install_dashboard(app)


@app.get("/health")
def health():
    return {"status": "ok", "models": list(app.state.bundles)}


@app.get("/models")
def models():
    manifest = app.state.manifest
    return {
        "feature_names": manifest["feature_names"], "window_size": 30,
        "input_units": "raw sensor values, chronological rows",
        "models": {name: {
            "threshold": app.state.bundles[name]["threshold"],
            "artifact_sha256": manifest["models"][name]["artifact_sha256"],
            "training": manifest["models"][name]["training"],
        } for name in MODEL_NAMES},
    }


def predict_one(name: str, request: WindowRequest) -> Prediction:
    if name not in app.state.bundles:
        raise HTTPException(status_code=404, detail={"available_models": list(MODEL_NAMES)})
    bundle = app.state.bundles[name]
    if request.feature_names != bundle["feature_names"]:
        raise HTTPException(status_code=422, detail={"expected_feature_names": bundle["feature_names"]})
    start = perf_counter()
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = scale_raw(np.asarray(request.window), bundle["scaler"])
    if not np.isfinite(scaled).all():
        raise HTTPException(status_code=422, detail="Sensor values exceed the supported numeric range")
    score = float(bundle["model"].predict_proba(scaled.reshape(1, -1))[0, 1])
    prediction = int(score >= bundle["threshold"])
    return Prediction(
        model=name, artifact_sha256=app.state.manifest["models"][name]["artifact_sha256"],
        anomaly_score=score, threshold=bundle["threshold"], prediction=prediction,
        label="anomaly" if prediction else "normal", inference_ms=(perf_counter() - start) * 1000,
    )


@app.post("/predict/{model_name}", response_model=Prediction)
def predict(model_name: str, request: WindowRequest):
    return predict_one(model_name, request)


@app.post("/compare", response_model=Comparison)
def compare(request: WindowRequest):
    start = perf_counter()
    predictions = {name: predict_one(name, request) for name in MODEL_NAMES}
    return Comparison(predictions=predictions, processing_ms=(perf_counter() - start) * 1000)
