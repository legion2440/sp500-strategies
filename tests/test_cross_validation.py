import pandas as pd

from scripts.cross_validation import assert_temporal_folds, blocking_date_splits, expanding_date_splits


def test_expanding_cv_has_ten_temporal_folds():
    dates = pd.date_range("2013-01-01", periods=1100, freq="B")
    folds = expanding_date_splits(dates, n_splits=10, min_train_days=505)
    assert len(folds) == 10
    assert all(len(f.train_dates) >= 505 for f in folds)
    assert_temporal_folds(folds, "2018-01-01")


def test_blocking_cv_has_ten_temporal_folds():
    dates = pd.date_range("2013-01-01", periods=1100, freq="B")
    folds = blocking_date_splits(dates, n_splits=10, min_train_days=505)
    assert len(folds) == 10
    assert all(len(f.train_dates) >= 505 for f in folds)
    assert all(f.train_dates.max() < f.validation_dates.min() for f in folds)
    assert_temporal_folds(folds, "2018-01-01")
