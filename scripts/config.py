from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CV_DIR = RESULTS_DIR / "cross-validation"
MODEL_DIR = RESULTS_DIR / "selected-model"
STRATEGY_DIR = RESULTS_DIR / "strategy"
BONUS_DIR = RESULTS_DIR / "bonus"

STOCKS_FILE = DATA_DIR / "all_stocks_5yr.csv"
INDEX_FILE = DATA_DIR / "HistoricalData.csv"
FEATURES_FILE = DATA_DIR / "sp500_features.parquet"

TEST_START = "2017-01-01"
RANDOM_STATE = 42
MIN_TRAIN_DAYS = 505
N_SPLITS = 10


def ensure_directories() -> None:
    for path in (DATA_DIR, CV_DIR, MODEL_DIR, STRATEGY_DIR, BONUS_DIR):
        path.mkdir(parents=True, exist_ok=True)
