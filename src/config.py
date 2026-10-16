"""Central configuration for the project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "stock_data.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TICKERS = ["AMZN", "DPZ", "BTC", "NFLX"]

WINDOW_SIZE = 60        # days of history the model looks at (same as the reference model.py)
TEST_RATIO = 0.15       # last 15% of the timeline -> test set
VAL_RATIO = 0.15        # the 15% before that -> validation set
RANDOM_STATE = 42

LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 32

for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
