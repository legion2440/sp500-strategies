import pandas as pd

from scripts.features_engineering import build_feature_dataset


def _stocks(days=90):
    dates = pd.date_range("2016-01-01", periods=days, freq="B")
    rows = []
    for ticker, base in (("AAA", 100.0), ("BBB", 200.0)):
        for i, date in enumerate(dates):
            close = base + i
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + i,
            })
    return pd.DataFrame(rows)


def test_target_is_computed_within_ticker():
    df = build_feature_dataset(_stocks())
    aaa = df.xs("AAA", level="ticker")
    expected = aaa["close"].shift(-2) / aaa["close"].shift(-1) - 1.0
    pd.testing.assert_series_equal(aaa["future_return"], expected, check_names=False)


def test_future_price_change_does_not_modify_past_features():
    source = _stocks()
    before = build_feature_dataset(source)
    changed = source.copy()
    mask = (changed["ticker"] == "AAA") & (changed["date"] == changed["date"].max())
    changed.loc[mask, ["open", "high", "low", "close"]] *= 10
    after = build_feature_dataset(changed)

    cutoff = changed["date"].max() - pd.Timedelta(days=5)
    feature_cols = [c for c in before.columns if c not in {"future_return", "target"}]
    pd.testing.assert_frame_equal(
        before.loc[(slice(None, cutoff), "AAA"), feature_cols],
        after.loc[(slice(None, cutoff), "AAA"), feature_cols],
    )
