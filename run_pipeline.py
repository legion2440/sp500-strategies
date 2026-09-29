from scripts.backtest import main as backtest
from scripts.config import ensure_directories
from scripts.create_signal import main as create_signal
from scripts.features_engineering import main as features
from scripts.gridsearch import main as gridsearch
from scripts.model_selection import main as model_selection


def main() -> None:
    ensure_directories()
    features()
    gridsearch()
    model_selection()
    create_signal()
    backtest()


if __name__ == "__main__":
    main()
