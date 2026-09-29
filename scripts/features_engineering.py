from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from scripts.config import FEATURES_FILE, TEST_START, ensure_directories
from scripts.data import load_stocks

FEATURE_COLUMNS = [
    "return_1d", "return_2d", "return_5d", "return_10d", "return_20d",
    "log_return_1d", "high_low_range", "open_close_return", "gap_return",
    "sma_5_ratio", "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "ema_5_ratio", "ema_10_ratio", "ema_20_ratio",
    "rsi_14", "macd", "macd_signal", "macd_hist", "roc_5", "roc_10",
    "volatility_5", "volatility_10", "volatility_20",
    "atr_14_ratio", "bb_width", "bb_percent",
    "volume_change", "relative_volume_5", "relative_volume_20",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    close, high, low, open_, volume = g["close"], g["high"], g["low"], g["open"], g["volume"]

    for n in (1, 2, 5, 10, 20):
        g[f"return_{n}d"] = close.pct_change(n, fill_method=None)

    g["log_return_1d"] = np.log(close).diff()
    g["high_low_range"] = _safe_ratio(high - low, close)
    g["open_close_return"] = _safe_ratio(close - open_, open_)
    g["gap_return"] = _safe_ratio(open_, close.shift(1)) - 1.0

    for n in (5, 10, 20, 50):
        sma = SMAIndicator(close, window=n, fillna=False).sma_indicator()
        g[f"sma_{n}_ratio"] = _safe_ratio(close, sma) - 1.0

    for n in (5, 10, 20):
        ema = EMAIndicator(close, window=n, fillna=False).ema_indicator()
        g[f"ema_{n}_ratio"] = _safe_ratio(close, ema) - 1.0

    g["rsi_14"] = RSIIndicator(close, window=14, fillna=False).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    g["macd"] = _safe_ratio(macd.macd(), close)
    g["macd_signal"] = _safe_ratio(macd.macd_signal(), close)
    g["macd_hist"] = _safe_ratio(macd.macd_diff(), close)
    g["roc_5"] = ROCIndicator(close, window=5, fillna=False).roc() / 100.0
    g["roc_10"] = ROCIndicator(close, window=10, fillna=False).roc() / 100.0

    daily = close.pct_change(fill_method=None)
    for n in (5, 10, 20):
        g[f"volatility_{n}"] = daily.rolling(n).std()

    atr = AverageTrueRange(high, low, close, window=14, fillna=False).average_true_range()
    g["atr_14_ratio"] = _safe_ratio(atr, close)
    bb = BollingerBands(close, window=20, window_dev=2, fillna=False)
    g["bb_width"] = bb.bollinger_wband() / 100.0
    g["bb_percent"] = bb.bollinger_pband()

    g["volume_change"] = volume.pct_change(fill_method=None)
    g["relative_volume_5"] = _safe_ratio(volume, volume.rolling(5).mean())
    g["relative_volume_20"] = _safe_ratio(volume, volume.rolling(20).mean())

    future_return = close.shift(-2) / close.shift(-1) - 1.0
    g["future_return"] = future_return
    g["target"] = np.where(future_return.notna(), (future_return > 0).astype(int), np.nan)
    return g


def build_feature_dataset(stocks: pd.DataFrame) -> pd.DataFrame:
    frames = [_ticker_features(group) for _, group in stocks.groupby("ticker", sort=False)]
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["date", "ticker"]).set_index(["date", "ticker"])
    return df


def split_train_test(df: pd.DataFrame, test_start: str = TEST_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = df.index.get_level_values("date")
    test_start_ts = pd.Timestamp(test_start)
    train = df.loc[dates < test_start_ts].copy()
    test = df.loc[dates >= test_start_ts].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split is empty; check input date range and TEST_START")
    if train.index.get_level_values("date").max() >= test.index.get_level_values("date").min():
        raise AssertionError("Temporal train/test boundary is invalid")
    return train, test


def main() -> None:
    ensure_directories()
    stocks = load_stocks()
    dataset = build_feature_dataset(stocks)
    dataset.to_parquet(FEATURES_FILE)
    train, test = split_train_test(dataset)
    print(f"Saved {len(dataset):,} rows to {FEATURES_FILE}")
    print(f"Train: {train.index.get_level_values('date').min().date()} -> {train.index.get_level_values('date').max().date()}")
    print(f"Test:  {test.index.get_level_values('date').min().date()} -> {test.index.get_level_values('date').max().date()}")
    print(f"Features: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
