from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config
from .model import LoadedModel, load_trained_model


class TrafficForecastingService:
    def __init__(
        self,
        model_path: Path | None = None,
        adjacency_path: Path | None = None,
        normalization_path: Path | None = None,
        sensors_path: Path | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self._resolve_model_path(model_path or config.MODEL_PATH)
        self.adjacency_path = adjacency_path or config.ADJACENCY_PATH
        self.normalization_path = normalization_path or config.NORMALIZATION_PATH
        self.sensors_path = sensors_path or config.SENSORS_PATH

        self.adjacency = self._load_adjacency(self.adjacency_path)
        self.num_nodes = int(self.adjacency.shape[0])
        self.normalization = self._load_normalization(self.normalization_path)
        self.mean = float(self.normalization["mean"])
        self.std = max(float(self.normalization["std"]), 1e-6)
        self.sensors = self._load_sensors(self.sensors_path)

        self.loaded_model: LoadedModel = load_trained_model(
            self.model_path,
            device=self.device,
            num_nodes=self.num_nodes,
        )
        self.model = self.loaded_model.model
        self.adj_tensor = torch.tensor(self.adjacency, dtype=torch.float32, device=self.device)

    @staticmethod
    def _resolve_model_path(path: Path) -> Path:
        if path.exists():
            return path
        if config.DOWNLOADS_MODEL_PATH.exists():
            return config.DOWNLOADS_MODEL_PATH
        return path

    @staticmethod
    def _load_adjacency(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Adjacency matrix not found at {path}. Run backend/scripts/generate_assets.py."
            )

        adj = np.load(path).astype(np.float32)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {adj.shape}.")
        return adj

    @staticmethod
    def _load_normalization(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Normalization file not found at {path}. Run backend/scripts/generate_assets.py."
            )

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if "mean" not in payload or "std" not in payload:
            raise ValueError("Normalization JSON must contain 'mean' and 'std'.")
        return payload

    @staticmethod
    def _load_sensors(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return [
                {"id": idx, "name": f"Sensor {idx}", "latitude": 0.0, "longitude": 0.0}
                for idx in range(config.DEFAULT_NUM_NODES)
            ]

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def predict(
        self,
        traffic_sequence: list[Any],
        sensor_id: int = 0,
        start_time_step: int | None = None,
        return_all_sensors: bool = False,
    ) -> dict[str, Any]:
        sensor_id = int(sensor_id)
        if not 0 <= sensor_id < self.num_nodes:
            raise ValueError(f"sensor_id must be between 0 and {self.num_nodes - 1}.")

        matrix, simulated_missing_nodes = self._coerce_sequence(traffic_sequence, sensor_id)
        if matrix.shape[0] != config.SEQ_LEN:
            raise ValueError(f"traffic_sequence must contain exactly {config.SEQ_LEN} time steps.")

        start_slot = self._current_time_slot() if start_time_step is None else int(start_time_step)
        time_window = self._time_features(start_slot, config.SEQ_LEN)
        normalized_speed = (matrix - self.mean) / self.std

        pred_norm = self._rollout(normalized_speed, time_window, start_slot, config.HORIZON)
        predicted = np.clip(pred_norm * self.std + self.mean, 0.0, 100.0)
        selected_prediction = self._round_list(predicted[:, sensor_id])
        selected_past = self._round_list(matrix[:, sensor_id])

        by_sensor = None
        if return_all_sensors:
            by_sensor = {
                str(idx): self._round_list(predicted[:, idx])
                for idx in range(self.num_nodes)
            }

        return {
            "sensor_id": sensor_id,
            "past_traffic": selected_past,
            "predicted_traffic": selected_prediction,
            "predicted_by_sensor": by_sensor,
            "time_step_minutes": config.TIME_STEP_MINUTES,
            "model_family": self.loaded_model.family,
            "simulated_missing_nodes": simulated_missing_nodes,
            "input_shape": list(matrix.shape),
            "message": "Prediction generated from the loaded PyTorch GNN + LSTM checkpoint.",
        }

    def sample(self, sensor_id: int = 0) -> dict[str, Any]:
        sensor_id = int(np.clip(sensor_id, 0, self.num_nodes - 1))
        if config.SAMPLE_PATH.exists():
            with config.SAMPLE_PATH.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            sequence = payload.get("traffic_sequence", [])
        else:
            sequence = self._synthetic_sensor_series(sensor_id).tolist()

        matrix, _ = self._coerce_sequence(sequence, sensor_id)
        return {
            "sensor_id": sensor_id,
            "traffic_sequence": self._round_list(matrix[:, sensor_id]),
            "matrix_shape": list(matrix.shape),
        }

    def _rollout(
        self,
        speed_window: np.ndarray,
        time_window: np.ndarray,
        start_slot: int,
        horizon: int,
    ) -> np.ndarray:
        window = speed_window.astype(np.float32).copy()
        time_features = time_window.astype(np.float32).copy()
        predictions = []

        with torch.no_grad():
            for step in range(horizon):
                model_input = np.stack([window, time_features], axis=-1)[np.newaxis, ...]
                x_tensor = torch.tensor(model_input, dtype=torch.float32, device=self.device)
                raw_output = self.model(x_tensor, self.adj_tensor)
                output = raw_output[0] if isinstance(raw_output, tuple) else raw_output
                next_norm = output[:, -1, :].detach().cpu().numpy()[0].astype(np.float32)
                next_norm = np.nan_to_num(next_norm, nan=0.0, posinf=4.0, neginf=-4.0)
                next_norm = np.clip(next_norm, -5.0, 5.0)
                predictions.append(next_norm)

                next_time = self._time_features(start_slot + config.SEQ_LEN + step, 1)[0]
                window = np.vstack([window[1:], next_norm])
                time_features = np.vstack([time_features[1:], next_time])

        return np.stack(predictions, axis=0)

    def _coerce_sequence(self, traffic_sequence: list[Any], sensor_id: int) -> tuple[np.ndarray, bool]:
        arr = np.array(traffic_sequence, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size != config.SEQ_LEN:
                raise ValueError(f"Single-sensor input must have {config.SEQ_LEN} values.")
            return self._simulate_full_network(arr, sensor_id), True

        if arr.ndim != 2:
            raise ValueError("traffic_sequence must be either a 12-value list or a 12 x N matrix.")

        if arr.shape == (config.SEQ_LEN, self.num_nodes):
            return np.nan_to_num(arr, nan=self.mean), False

        if arr.shape == (self.num_nodes, config.SEQ_LEN):
            return np.nan_to_num(arr.T, nan=self.mean), False

        if arr.shape == (config.SEQ_LEN, 1):
            return self._simulate_full_network(arr[:, 0], sensor_id), True

        raise ValueError(
            f"Expected shape ({config.SEQ_LEN}, {self.num_nodes}) or a {config.SEQ_LEN}-value sensor series; "
            f"got {arr.shape}."
        )

    def _simulate_full_network(self, sensor_series: np.ndarray, sensor_id: int) -> np.ndarray:
        base = np.nan_to_num(sensor_series.astype(np.float32), nan=self.mean)
        node_ids = np.arange(self.num_nodes, dtype=np.float32)
        distance = np.minimum(np.abs(node_ids - sensor_id), self.num_nodes - np.abs(node_ids - sensor_id))
        scale = 1.0 + 0.035 * np.sin(node_ids * 0.41)
        offset = 2.8 * np.cos(node_ids * 0.17) * np.exp(-distance / 95.0)
        full = base[:, None] * scale[None, :] + offset[None, :]

        neighbor_mix = self.adjacency / np.maximum(self.adjacency.sum(axis=1, keepdims=True), 1e-6)
        full = 0.82 * full + 0.18 * (full @ neighbor_mix.T)
        full[:, sensor_id] = base
        return np.clip(full, 0.0, 100.0).astype(np.float32)

    def _synthetic_sensor_series(self, sensor_id: int) -> np.ndarray:
        steps = np.arange(config.SEQ_LEN, dtype=np.float32)
        rush = 7.5 * np.sin((steps / (config.SEQ_LEN - 1)) * np.pi)
        local_offset = 2.0 * np.sin(sensor_id * 0.23)
        trend = np.linspace(-2.0, 3.0, config.SEQ_LEN, dtype=np.float32)
        return np.clip(self.mean + local_offset + rush + trend, 5.0, 95.0)

    def _time_features(self, start_slot: int, length: int) -> np.ndarray:
        slots_per_day = (24 * 60) // config.TIME_STEP_MINUTES
        slots = (start_slot + np.arange(length, dtype=np.float32)) % slots_per_day
        values = (slots * config.TIME_STEP_MINUTES) / (24 * 60)
        return np.tile(values[:, None], (1, self.num_nodes)).astype(np.float32)

    @staticmethod
    def _current_time_slot() -> int:
        now = datetime.now()
        minutes = now.hour * 60 + now.minute
        return minutes // config.TIME_STEP_MINUTES

    @staticmethod
    def _round_list(values: np.ndarray) -> list[float]:
        return [round(float(value), 3) for value in values.tolist()]

