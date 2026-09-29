from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    equity = 1.0 + cumulative_pnl
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def annualized_return(daily_pnl: pd.Series) -> float:
    clean = daily_pnl.dropna()
    return float(clean.mean() * TRADING_DAYS) if not clean.empty else float("nan")


def annualized_volatility(daily_pnl: pd.Series) -> float:
    return float(daily_pnl.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_pnl: pd.Series) -> float:
    clean = daily_pnl.dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / std * np.sqrt(TRADING_DAYS)) if std and np.isfinite(std) else float("nan")


def sortino_ratio(daily_pnl: pd.Series) -> float:
    clean = daily_pnl.dropna()
    downside = clean[clean < 0].std(ddof=1)
    return float(clean.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and np.isfinite(downside) else float("nan")


def strategy_metrics(daily_pnl: pd.Series) -> dict[str, float]:
    clean = daily_pnl.dropna()
    cumulative = clean.cumsum()
    mdd = max_drawdown(cumulative) if not cumulative.empty else float("nan")
    ann = annualized_return(clean)
    return {
        "pnl": float(clean.sum()) if not clean.empty else float("nan"),
        "annualized_return": ann,
        "annualized_volatility": annualized_volatility(clean),
        "sharpe": sharpe_ratio(clean),
        "sortino": sortino_ratio(clean),
        "max_drawdown": mdd,
        "calmar": float(ann / abs(mdd)) if np.isfinite(ann) and mdd < 0 else float("nan"),
        "win_rate": float((clean > 0).mean()) if not clean.empty else float("nan"),
    }
