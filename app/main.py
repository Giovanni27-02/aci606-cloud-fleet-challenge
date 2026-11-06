from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.db import TelemetryRecord, get_db, init_db

# 1. Inicializar tablas en la base de datos
init_db()

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="1.0.0",
    description="Servicio Cloud persistente con inferencia remota para flotas de robots.",
)

# 2. Cargar artefacto del modelo
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
else:
    model = None
    MODEL_LOADED = False

# 3. Métricas operacionales en memoria del proceso
START_TIME = time.time()
METRICS = {
    "requests_total": 0,
    "status_2xx": 0,
    "status_4xx": 0,
    "status_5xx": 0,
    "predictions_total": 0,
    "inference_latencies_ms": [],
}


@app.middleware("http")
async def track_metrics_middleware(request: Request, call_next):
    METRICS["requests_total"] += 1
    try:
        response: Response = await call_next(request)
        if 200 <= response.status_code < 300:
            METRICS["status_2xx"] += 1
        elif 400 <= response.status_code < 500:
            METRICS["status_4xx"] += 1
        elif response.status_code >= 500:
            METRICS["status_5xx"] += 1
        return response
    except Exception:
        METRICS["status_5xx"] += 1
        raise


Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):
    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    battery: float = Field(ge=0, le=100)
    motor_temp: float
    vibration: float = Field(ge=0)
    motor_current: float = Field(ge=0)
    obstacle_distance: float = Field(ge=0)


class PredictionRequest(Telemetry):
    pass


class PredictionResponse(BaseModel):
    robot_id: str
    predicted_status: Status
    confidence: float
    model_version: str
    inference_ms: float


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    total_records = db.query(TelemetryRecord).count()
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "model_loaded": MODEL_LOADED,
        "total_records_db": total_records,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry, db: Session = Depends(get_db)) -> dict:
    record = TelemetryRecord(
        robot_id=sample.robot_id,
        timestamp=sample.timestamp,
        x=sample.x,
        y=sample.y,
        speed=sample.speed,
        battery=sample.battery,
        motor_temp=sample.motor_temp,
        vibration=sample.vibration,
        motor_current=sample.motor_current,
        obstacle_distance=sample.obstacle_distance,
    )
    db.add(record)
    db.commit()
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "persistence": "POSTGRESQL",
    }


@app.get("/robots")
def list_robots(db: Session = Depends(get_db)) -> list[dict]:
    # Obtener el id del registro más reciente para cada robot
    subquery = (
        db.query(
            TelemetryRecord.robot_id,
            func.max(TelemetryRecord.timestamp).label("max_ts"),
        )
        .group_by(TelemetryRecord.robot_id)
        .subquery()
    )

    records = (
        db.query(TelemetryRecord)
        .join(
            subquery,
            (TelemetryRecord.robot_id == subquery.c.robot_id)
            & (TelemetryRecord.timestamp == subquery.c.max_ts),
        )
        .all()
    )

    return [
        {
            "robot_id": r.robot_id,
            "timestamp": r.timestamp,
            "x": r.x,
            "y": r.y,
            "speed": r.speed,
            "battery": r.battery,
            "motor_temp": r.motor_temp,
            "vibration": r.vibration,
            "motor_current": r.motor_current,
            "obstacle_distance": r.obstacle_distance,
        }
        for r in records
    ]


@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str, db: Session = Depends(get_db)) -> list[dict]:
    records = (
        db.query(TelemetryRecord)
        .filter(TelemetryRecord.robot_id == robot_id)
        .order_by(TelemetryRecord.timestamp.asc())
        .all()
    )
    if not records:
        raise HTTPException(status_code=404, detail="Robot not found")

    return [
        {
            "robot_id": r.robot_id,
            "timestamp": r.timestamp,
            "x": r.x,
            "y": r.y,
            "speed": r.speed,
            "battery": r.battery,
            "motor_temp": r.motor_temp,
            "vibration": r.vibration,
            "motor_current": r.motor_current,
            "obstacle_distance": r.obstacle_distance,
        }
        for r in records
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> dict:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de predicción no disponible o no inicializado.",
        )

    features_df = pd.DataFrame(
        [
            {
                "speed": data.speed,
                "battery": data.battery,
                "motor_temp": data.motor_temp,
                "vibration": data.vibration,
                "motor_current": data.motor_current,
                "obstacle_distance": data.obstacle_distance,
            }
        ]
    )

    t0 = time.perf_counter()
    predicted_class = str(model.predict(features_df)[0])
    probabilities = model.predict_proba(features_df)[0]
    t1 = time.perf_counter()

    inference_ms = round((t1 - t0) * 1000, 2)
    confidence = float(np.max(probabilities))

    METRICS["predictions_total"] += 1
    METRICS["inference_latencies_ms"].append(inference_ms)

    return {
        "robot_id": data.robot_id,
        "predicted_status": predicted_class,
        "confidence": round(confidence, 4),
        "model_version": "v1",
        "inference_ms": inference_ms,
    }


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)) -> dict:
    uptime_seconds = int(time.time() - START_TIME)
    latencies = METRICS["inference_latencies_ms"]

    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    robots_count = (
        db.query(func.count(func.distinct(TelemetryRecord.robot_id))).scalar() or 0
    )
    total_telemetry = db.query(func.count(TelemetryRecord.id)).scalar() or 0

    return {
        "uptime_seconds": uptime_seconds,
        "requests_total": METRICS["requests_total"],
        "status_codes": {
            "2xx": METRICS["status_2xx"],
            "4xx": METRICS["status_4xx"],
            "5xx": METRICS["status_5xx"],
        },
        "predictions": {
            "total": METRICS["predictions_total"],
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_mean_ms": round(mean_lat, 2),
        },
        "database": {
            "persistence": "POSTGRESQL",
            "robots_tracked": robots_count,
            "total_samples": total_telemetry,
        },
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")