from pathlib import Path
import os


BACKEND_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BACKEND_DIR / "assets"
MODELS_DIR = BACKEND_DIR / "models"

DEFAULT_MODEL_PATH = MODELS_DIR / "traffic_model.pth"
DOWNLOADS_MODEL_PATH = Path.home() / "Downloads" / "traffic_model.pth"

MODEL_PATH = Path(os.getenv("TRAFFIC_MODEL_PATH", DEFAULT_MODEL_PATH))
ADJACENCY_PATH = Path(os.getenv("TRAFFIC_ADJACENCY_PATH", ASSETS_DIR / "adjacency.npy"))
NORMALIZATION_PATH = Path(os.getenv("TRAFFIC_NORMALIZATION_PATH", ASSETS_DIR / "normalization.json"))
SENSORS_PATH = Path(os.getenv("TRAFFIC_SENSORS_PATH", ASSETS_DIR / "sensors.json"))
SAMPLE_PATH = Path(os.getenv("TRAFFIC_SAMPLE_PATH", ASSETS_DIR / "sample_traffic.json"))

SEQ_LEN = int(os.getenv("TRAFFIC_SEQ_LEN", "12"))
HORIZON = int(os.getenv("TRAFFIC_HORIZON", "12"))
DEFAULT_NUM_NODES = int(os.getenv("TRAFFIC_NUM_NODES", "207"))
TIME_STEP_MINUTES = int(os.getenv("TRAFFIC_TIME_STEP_MINUTES", "5"))

