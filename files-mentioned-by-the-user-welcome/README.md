# Spatio-Temporal Traffic Forecasting Web App

This project wraps the supplied PyTorch traffic forecasting checkpoint in a FastAPI backend and provides a browser UI for manual/sample traffic input, sensor selection, prediction, and line-chart visualization.

## Folder Structure

```text
backend/
  app/
    config.py
    inference.py
    main.py
    model.py
    schemas.py
  assets/
    adjacency.npy
    normalization.json
    sample_traffic.json
    sensors.json
  models/
    traffic_model.pth
  scripts/
    generate_assets.py
    smoke_test.py
  requirements.txt
frontend/
  app.js
  index.html
  styles.css
README.md
```

## Backend

Create assets if you ever need to regenerate them:

```powershell
cd backend
python scripts/generate_assets.py
```

Install dependencies and run the API:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Useful endpoints:

```text
GET  http://127.0.0.1:8000/health
GET  http://127.0.0.1:8000/sensors
GET  http://127.0.0.1:8000/sample?sensor_id=0
POST http://127.0.0.1:8000/predict
```

Example prediction body:

```json
{
  "sensor_id": 0,
  "traffic_sequence": [49.5, 52.1, 54.8, 58.2, 61.3, 62.5, 63.1, 61.2, 59.0, 56.8, 55.2, 53.7],
  "return_all_sensors": false
}
```

## Frontend

From a second terminal:

```powershell
cd frontend
python -m http.server 5173
```

Open:

```text
http://127.0.0.1:5173
```

The UI calls the backend with `fetch`, shows loading/errors, and visualizes past vs predicted traffic using Chart.js.

## Model Integration Notes

The backend loads `backend/models/traffic_model.pth` and uses the checkpoint keys to build the matching GNN + LSTM module. It also loads `backend/assets/adjacency.npy` plus `backend/assets/normalization.json`. If the original training mean/std or adjacency matrix are available later, replace those two asset files without changing the API or frontend.
