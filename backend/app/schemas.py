from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    traffic_sequence: list[Any] = Field(
        ...,
        description="Either 12 traffic values for one sensor or a 12 x N matrix.",
    )
    sensor_id: int = Field(0, ge=0, description="Sensor/node to visualize and return.")
    start_time_step: int | None = Field(
        None,
        ge=0,
        le=287,
        description="Optional 5-minute slot in the day. Defaults to current local time.",
    )
    return_all_sensors: bool = Field(False, description="Include every sensor forecast in the response.")


class PredictionResponse(BaseModel):
    sensor_id: int
    past_traffic: list[float]
    predicted_traffic: list[float]
    predicted_by_sensor: dict[str, list[float]] | None = None
    time_step_minutes: int
    model_family: str
    simulated_missing_nodes: bool
    input_shape: list[int]
    message: str


class Sensor(BaseModel):
    id: int
    name: str
    latitude: float
    longitude: float

