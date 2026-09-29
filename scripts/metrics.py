from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def annualized_return(daily_returns: pd.Series) -> float:
    clean = daily_returns.dropna()
    if clean.empty:
        return float("nan")
    total = float((1.0 + clean).prod())
    years = len(clean) / TRADING_DAYS
    return total ** (1.0 / years) - 1.0 if total > 0 and years > 0 else float("nan")


def annualized_volatility(daily_returns: pd.Series) -> float:
    return float(daily_returns.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series) -> float:
    clean = daily_returns.dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / std * np.sqrt(TRADING_DAYS)) if std and np.isfinite(std) else float("nan")


def sortino_ratio(daily_returns: pd.Series) -> float:
    clean = daily_returns.dropna()
    downside = clean[clean < 0].std(ddof=1)
    return float(clean.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and np.isfinite(downside) else float("nan")


def strategy_metrics(daily_returns: pd.Series) -> dict[str, float]:
    clean = daily_returns.dropna()
    equity = (1.0 + clean).cumprod()
    mdd = max_drawdown(equity) if not equity.empty else float("nan")
    ann = annualized_return(clean)
    return {
        "pnl": float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan"),
        "annualized_return": ann,
        "annualized_volatility": annualized_volatility(clean),
        "sharpe": sharpe_ratio(clean),
        "sortino": sortino_ratio(clean),
        "max_drawdown": mdd,
        "calmar": float(ann / abs(mdd)) if np.isfinite(ann) and mdd < 0 else float("nan"),
        "win_rate": float((clean > 0).mean()) if not clean.empty else float("nan"),
    }
