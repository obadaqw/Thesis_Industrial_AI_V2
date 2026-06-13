"""
platform_server.py — REST API background server (Decision #14).

Endpoints:
  GET  /health   — system liveness check
  POST /predict  — single-cycle quality inference
  GET  /oee      — latest digital twin OEE snapshot
  POST /rca      — trigger counterfactual RCA on a sensor vector

Run standalone:
  python src/platform_server.py

Or start as a background daemon from app.py:
  from platform_server import start_background
  start_background()
"""
import os
import sys
import threading
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

# Lazy imports so the module can be imported without fastapi installed
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

_server_started = False
_oee_snapshot   = {"oee": None, "updated": None}   # shared mutable state


def _build_app():
    app = FastAPI(
        title="Digital Shadow API",
        description="Injection Molding Quality Control — REST interface",
        version="2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # ── Load model artifacts once ────────────────────────────────────────
    try:
        _model         = joblib.load(os.path.join(MODELS_DIR, "current_model.pkl"))
        _scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        _feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        _ready = True
    except Exception:
        _model = _scaler = _feature_names = None
        _ready = False

    CLASS_NAMES = {0: "Waste", 1: "Acceptable", 2: "Target", 3: "Inefficient"}

    # ── Schemas ──────────────────────────────────────────────────────────
    class SensorVector(BaseModel):
        features: list[float]   # 13 values in physical units

    class OEEUpdate(BaseModel):
        oee: float
        availability: float
        performance: float
        quality: float
        total_cycles: int
        good_cycles: int

    # ── Endpoints ────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return {
            "status":    "online",
            "model_ready": _ready,
            "timestamp": datetime.now().isoformat(),
            "version":   "2.0",
        }

    @app.post("/predict")
    def predict(body: SensorVector):
        if not _ready:
            raise HTTPException(503, "Model not loaded")
        if len(body.features) != len(_feature_names):
            raise HTTPException(
                400,
                f"Expected {len(_feature_names)} features, got {len(body.features)}"
            )
        row_real = pd.DataFrame([body.features], columns=_feature_names)
        row_scaled = _scaler.transform(row_real)
        pred  = int(_model.predict(row_scaled)[0])
        proba = _model.predict_proba(row_scaled)[0].tolist()
        return {
            "prediction":   pred,
            "quality_label": CLASS_NAMES[pred],
            "probabilities": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(proba)},
            "confidence":   round(float(max(proba)), 4),
        }

    @app.get("/oee")
    def oee():
        if _oee_snapshot["oee"] is None:
            raise HTTPException(503, "No OEE data yet — start the Digital Twin")
        return _oee_snapshot

    @app.post("/oee/update")
    def oee_update(body: OEEUpdate):
        _oee_snapshot.update(body.dict())
        _oee_snapshot["updated"] = datetime.now().isoformat()
        return {"status": "ok"}

    @app.post("/rca")
    def rca(body: SensorVector):
        if not _ready:
            raise HTTPException(503, "Model not loaded")
        if len(body.features) != len(_feature_names):
            raise HTTPException(400, "Feature count mismatch")
        try:
            sys.path.insert(0, os.path.join(BASE_DIR, "src"))
            from counterfactual_rca import CounterfactualRCA
            engine = CounterfactualRCA()
            row_real = pd.DataFrame([body.features], columns=_feature_names)
            result   = engine.analyze(row_real)
            return result
        except Exception as e:
            raise HTTPException(500, str(e))

    return app


def start_background(port: int = 8502) -> bool:
    """Start the FastAPI server in a daemon thread. Returns True if started."""
    global _server_started
    if not _FASTAPI_AVAILABLE:
        print("⚠️  fastapi/uvicorn not installed — API server disabled")
        return False
    if _server_started:
        return True

    app = _build_app()

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True, name="platform-api")
    t.start()
    _server_started = True
    print(f"🌐 Platform API started on http://localhost:{port}  (docs: /docs)")
    return True


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(_build_app(), host="0.0.0.0", port=8502, reload=False)
