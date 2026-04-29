from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
NUM_NODES = 207
SEQ_LEN = 12


def build_adjacency(num_nodes: int = NUM_NODES) -> np.ndarray:
    node_ids = np.arange(num_nodes, dtype=np.float32)
    ring_distance = np.abs(node_ids[:, None] - node_ids[None, :])
    ring_distance = np.minimum(ring_distance, num_nodes - ring_distance)
    sigma = 8.0
    adjacency = np.exp(-(ring_distance**2) / (sigma**2)).astype(np.float32)
    adjacency[adjacency < 0.08] = 0.0
    adjacency += np.eye(num_nodes, dtype=np.float32)
    adjacency = (adjacency + adjacency.T) / 2.0
    return adjacency.astype(np.float32)


def build_sensors(num_nodes: int = NUM_NODES) -> list[dict[str, float | int | str]]:
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    return [
        {
            "id": int(idx),
            "name": f"Sensor {idx:03d}",
            "latitude": round(float(34.05 + 0.18 * np.sin(angle) + 0.015 * np.sin(idx / 7)), 6),
            "longitude": round(float(-118.25 + 0.24 * np.cos(angle) + 0.012 * np.cos(idx / 9)), 6),
        }
        for idx, angle in enumerate(angles)
    ]


def build_sample() -> list[float]:
    steps = np.arange(SEQ_LEN, dtype=np.float32)
    morning_rise = 8.0 * np.sin((steps / (SEQ_LEN - 1)) * np.pi)
    slow_drift = np.linspace(-2.5, 3.5, SEQ_LEN, dtype=np.float32)
    ripple = 1.4 * np.sin(steps * 0.9)
    sample = 52.0 + morning_rise + slow_drift + ripple
    return [round(float(value), 3) for value in sample]


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    np.save(ASSETS / "adjacency.npy", build_adjacency())

    normalization = {
        "mean": 52.0,
        "std": 12.0,
        "num_nodes": NUM_NODES,
        "time_step_minutes": 5,
        "source": "Local demo normalization for the provided checkpoint. Replace with training mean/std if available.",
    }
    (ASSETS / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    (ASSETS / "sensors.json").write_text(json.dumps(build_sensors(), indent=2), encoding="utf-8")
    (ASSETS / "sample_traffic.json").write_text(
        json.dumps({"sensor_id": 0, "traffic_sequence": build_sample()}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

