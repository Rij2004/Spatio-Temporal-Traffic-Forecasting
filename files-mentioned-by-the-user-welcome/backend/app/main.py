from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .inference import TrafficForecastingService
from .schemas import PredictionRequest, PredictionResponse, Sensor


app = FastAPI(
    title="Spatio-Temporal Traffic Forecasting API",
    description="FastAPI wrapper around a PyTorch GNN + LSTM traffic forecasting checkpoint.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = TrafficForecastingService()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Spatio-Temporal Traffic Forecasting API",
        "predict": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_family": service.loaded_model.family,
        "model_path": str(service.model_path),
        "num_nodes": service.num_nodes,
        "device": str(service.device),
        "mean": service.mean,
        "std": service.std,
    }


@app.get("/sensors", response_model=list[Sensor])
def sensors() -> list[dict[str, object]]:
    return service.sensors


@app.get("/sample")
def sample(sensor_id: int = Query(0, ge=0)) -> dict[str, object]:
    try:
        return service.sample(sensor_id=sensor_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> dict[str, object]:
    try:
        return service.predict(
            traffic_sequence=payload.traffic_sequence,
            sensor_id=payload.sensor_id,
            start_time_step=payload.start_time_step,
            return_all_sensors=payload.return_all_sensors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

