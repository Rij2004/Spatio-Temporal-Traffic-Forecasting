from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def main() -> None:
    client = TestClient(app)
    health = client.get("/health")
    health.raise_for_status()
    sample = client.get("/sample?sensor_id=0")
    sample.raise_for_status()
    response = client.post(
        "/predict",
        json={
            "sensor_id": 0,
            "traffic_sequence": sample.json()["traffic_sequence"],
        },
    )
    response.raise_for_status()
    payload = response.json()
    assert len(payload["predicted_traffic"]) == 12
    print("Smoke test passed")
    print("Model:", payload["model_family"])
    print("Predicted:", payload["predicted_traffic"])


if __name__ == "__main__":
    main()

