from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_daily_weights(weights: pd.Series) -> pd.Series:
    gross = weights.abs().groupby(level="date").transform("sum")
    return weights.where(gross.eq(0), weights / gross.replace(0, np.nan)).fillna(0.0)


def long_only(signal: pd.Series, threshold: float = 0.5) -> pd.Series:
    weights = (signal > threshold).astype(float)
    return normalize_daily_weights(weights)


def long_short(signal: pd.Series, upper: float = 0.55, lower: float = 0.45) -> pd.Series:
    raw = pd.Series(0.0, index=signal.index)
    raw.loc[signal > upper] = 1.0
    raw.loc[signal < lower] = -1.0
    return normalize_daily_weights(raw)


def probability_weighted(signal: pd.Series) -> pd.Series:
    raw = signal.clip(lower=0.0)
    return normalize_daily_weights(raw)


def stock_picking(signal: pd.Series, k: int = 10) -> pd.Series:
    if k < 1:
        raise ValueError("k must be >= 1")

    def per_day(day_signal: pd.Series) -> pd.Series:
        n = len(day_signal)
        if n < 2:
            return pd.Series(0.0, index=day_signal.index)
        kk = min(k, n // 2)
        result = pd.Series(0.0, index=day_signal.index)
        result.loc[day_signal.nlargest(kk).index] = 1.0
        result.loc[day_signal.nsmallest(kk).index] = -1.0
        return result

    raw = signal.groupby(level="date", group_keys=False).apply(per_day)
    return normalize_daily_weights(raw)
