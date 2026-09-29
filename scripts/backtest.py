from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from scripts.config import FEATURES_FILE, INDEX_FILE, MODEL_DIR, STRATEGY_DIR, TEST_START, ensure_directories
from scripts.data import load_index
from scripts.metrics import strategy_metrics
from scripts.strategy import long_only, long_short, probability_weighted, stock_picking


def _load_signal() -> pd.DataFrame:
    signal = pd.read_csv(MODEL_DIR / "ml_signal.csv", parse_dates=["date"])
    signal = signal.set_index(["date", "ticker"]).sort_index()
    return signal


def _benchmark_returns() -> pd.Series:
    index_df = load_index(INDEX_FILE).set_index("date")
    future_return = index_df["close"].shift(-2) / index_df["close"].shift(-1) - 1.0
    return future_return.rename("sp500_return")


def _daily_strategy_return(weights: pd.Series, future_return: pd.Series) -> pd.Series:
    aligned = pd.concat(
        [weights.rename("weight"), future_return.rename("future_return")], axis=1, join="inner"
    ).dropna()
    aligned["contribution"] = aligned["weight"] * aligned["future_return"]
    return aligned["contribution"].groupby(level="date").sum().sort_index()


def main() -> None:
    ensure_directories()
    dataset = pd.read_parquet(FEATURES_FILE)
    signal_df = _load_signal()
    merged = signal_df.join(dataset[["future_return"]], how="inner").dropna(subset=["future_return"])

    signal = merged["probability_up"]
    strategies = {
        "long_only_0.50": long_only(signal, 0.50),
        "long_short_0.55_0.45": long_short(signal, 0.55, 0.45),
        "probability_weighted": probability_weighted(signal),
        "stock_picking_10": stock_picking(signal, 10),
    }

    daily = pd.DataFrame(
        {name: _daily_strategy_return(weights, merged["future_return"]) for name, weights in strategies.items()}
    ).sort_index()
    benchmark = _benchmark_returns()
    daily = daily.join(benchmark, how="left")
    daily.to_csv(STRATEGY_DIR / "daily_returns.csv")

    rows = []
    boundary = pd.Timestamp(TEST_START)
    for name in daily.columns:
        for scope, values in (
            ("train", daily.loc[daily.index < boundary, name]),
            ("test", daily.loc[daily.index >= boundary, name]),
        ):
            rows.append({"strategy": name, "scope": scope, **strategy_metrics(values)})
    results = pd.DataFrame(rows)
    results.to_csv(STRATEGY_DIR / "results.csv", index=False)

    primary = "stock_picking_10"
    primary_equity = (1.0 + daily[primary].fillna(0.0)).cumprod()
    benchmark_equity = (1.0 + daily["sp500_return"].fillna(0.0)).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(primary_equity.index, primary_equity.values, label=primary)
    ax.plot(benchmark_equity.index, benchmark_equity.values, label="S&P 500")
    ax.axvline(boundary, linestyle="--", label="Train / test split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity from $1")
    ax.set_title("Strategy vs S&P 500")
    ax.legend()
    fig.tight_layout()
    fig.savefig(STRATEGY_DIR / "strategy.png", dpi=160)
    plt.close(fig)

    report = {
        "primary_strategy": primary,
        "capital_rule": "sum(abs(weights)) = 1 per trading day",
        "pnl_formula": "weight(D,i) * return(D+1,D+2,i)",
        "test_start": TEST_START,
    }
    (STRATEGY_DIR / "backtest_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
