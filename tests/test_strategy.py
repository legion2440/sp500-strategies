import numpy as np
import pandas as pd

from scripts.strategy import long_only, long_short, stock_picking


def _signal():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2016-01-04", "2016-01-05"]), ["A", "B", "C", "D"]],
        names=["date", "ticker"],
    )
    return pd.Series([0.9, 0.7, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2], index=index)


def test_long_only_normalizes_daily_capital():
    weights = long_only(_signal(), 0.5)
    gross = weights.abs().groupby(level="date").sum()
    assert np.allclose(gross.values, 1.0)


def test_long_short_normalizes_daily_gross_exposure():
    weights = long_short(_signal(), upper=0.65, lower=0.35)
    gross = weights.abs().groupby(level="date").sum()
    assert np.allclose(gross.values, 1.0)


def test_stock_picking_is_market_neutral():
    weights = stock_picking(_signal(), k=1)
    net = weights.groupby(level="date").sum()
    gross = weights.abs().groupby(level="date").sum()
    assert np.allclose(net.values, 0.0)
    assert np.allclose(gross.values, 1.0)
