from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from scripts.config import BONUS_DIR, FEATURES_FILE, MODEL_DIR, ensure_directories
from scripts.metrics import strategy_metrics
from scripts.strategy import long_only, stock_picking


def _daily_return(weights: pd.Series, returns: pd.Series) -> pd.Series:
    frame = pd.concat([weights.rename("w"), returns.rename("r")], axis=1).dropna()
    return (frame["w"] * frame["r"]).groupby(level="date").sum().sort_index()


def _turnover(weights: pd.Series) -> pd.Series:
    frame = weights.rename("w").unstack("ticker").fillna(0.0).sort_index()
    return frame.diff().abs().sum(axis=1).fillna(frame.abs().sum(axis=1))


def main() -> None:
    ensure_directories()
    dataset = pd.read_parquet(FEATURES_FILE)
    signal = pd.read_csv(MODEL_DIR / "ml_signal.csv", parse_dates=["date"]).set_index(["date", "ticker"])
    frame = signal.join(dataset[["future_return", "target"]], how="inner").dropna(subset=["future_return"])
    s = frame["probability_up"]

    rows = []
    for threshold, cost_bps in itertools.product((0.50, 0.52, 0.55, 0.60), (0, 5, 10, 20)):
        weights = long_only(s, threshold)
        gross = _daily_return(weights, frame["future_return"])
        turnover = _turnover(weights).reindex(gross.index).fillna(0.0)
        net = gross - turnover * (cost_bps / 10000.0)
        rows.append({
            "family": "long_only",
            "parameter": threshold,
            "cost_bps": cost_bps,
            **strategy_metrics(net),
            "mean_turnover": float(turnover.mean()),
        })

    for k, cost_bps in itertools.product((5, 10, 20, 50), (0, 5, 10, 20)):
        weights = stock_picking(s, k)
        gross = _daily_return(weights, frame["future_return"])
        turnover = _turnover(weights).reindex(gross.index).fillna(0.0)
        net = gross - turnover * (cost_bps / 10000.0)
        rows.append({
            "family": "stock_picking",
            "parameter": k,
            "cost_bps": cost_bps,
            **strategy_metrics(net),
            "mean_turnover": float(turnover.mean()),
        })

    pd.DataFrame(rows).to_csv(BONUS_DIR / "robustness.csv", index=False)

    labeled = frame.dropna(subset=["target"]).copy()
    y = labeled["target"].astype(int)
    p = labeled["probability_up"].clip(1e-6, 1 - 1e-6)
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    pd.DataFrame({"mean_predicted_probability": prob_pred, "fraction_positive": prob_true}).to_csv(
        BONUS_DIR / "calibration.csv", index=False
    )
    (BONUS_DIR / "calibration_summary.txt").write_text(
        f"Brier score: {brier_score_loss(y, p):.8f}\n", encoding="utf-8"
    )
    print(f"Saved robustness and calibration artifacts to {BONUS_DIR}")


if __name__ == "__main__":
    main()
