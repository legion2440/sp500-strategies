from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import CV_DIR, MIN_TRAIN_DAYS, N_SPLITS, ensure_directories


@dataclass(frozen=True)
class DateFold:
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex


def unique_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.MultiIndex):
        values = df.index.get_level_values("date")
    else:
        values = df.index
    return pd.DatetimeIndex(values.unique()).sort_values()


def expanding_date_splits(
    dates: pd.DatetimeIndex,
    n_splits: int = N_SPLITS,
    min_train_days: int = MIN_TRAIN_DAYS,
) -> list[DateFold]:
    dates = pd.DatetimeIndex(dates).sort_values()
    if len(dates) <= min_train_days + n_splits:
        raise ValueError("Not enough unique dates for requested expanding CV")

    remaining = len(dates) - min_train_days
    val_size = max(1, remaining // n_splits)
    folds: list[DateFold] = []
    for i in range(n_splits):
        train_end = min_train_days + i * val_size
        val_end = len(dates) if i == n_splits - 1 else min(train_end + val_size, len(dates))
        if val_end <= train_end:
            continue
        folds.append(DateFold(dates[:train_end], dates[train_end:val_end]))

    if len(folds) < n_splits:
        raise ValueError(f"Generated only {len(folds)} folds, expected {n_splits}")
    return folds


def blocking_date_splits(
    dates: pd.DatetimeIndex,
    n_splits: int = N_SPLITS,
    min_train_days: int = MIN_TRAIN_DAYS,
) -> list[DateFold]:
    dates = pd.DatetimeIndex(dates).sort_values()
    block_size = min_train_days + max(1, (len(dates) - n_splits * min_train_days) // n_splits)
    if block_size <= min_train_days or len(dates) < n_splits * (min_train_days + 1):
        block_size = len(dates) // n_splits
    if block_size <= min_train_days:
        raise ValueError("Not enough unique dates for requested blocking CV")

    folds: list[DateFold] = []
    for i in range(n_splits):
        start = i * block_size
        end = len(dates) if i == n_splits - 1 else min((i + 1) * block_size, len(dates))
        block = dates[start:end]
        if len(block) <= min_train_days:
            continue
        folds.append(DateFold(block[:min_train_days], block[min_train_days:]))

    if len(folds) < n_splits:
        raise ValueError(
            "Blocking CV cannot satisfy both >=10 folds and >2y train history with this date range"
        )
    return folds


def fold_indices(df: pd.DataFrame, folds: list[DateFold]):
    dates = df.index.get_level_values("date")
    for fold in folds:
        train_mask = dates.isin(fold.train_dates)
        val_mask = dates.isin(fold.validation_dates)
        yield np.flatnonzero(train_mask), np.flatnonzero(val_mask)


def assert_temporal_folds(folds: list[DateFold], test_start: str) -> None:
    boundary = pd.Timestamp(test_start)
    for i, fold in enumerate(folds, start=1):
        if fold.train_dates.intersection(fold.validation_dates).size:
            raise AssertionError(f"Fold {i}: train and validation dates overlap")
        if fold.train_dates.max() >= fold.validation_dates.min():
            raise AssertionError(f"Fold {i}: validation is not strictly after train")
        if fold.validation_dates.max() >= boundary:
            raise AssertionError(f"Fold {i}: validation overlaps the test period")


def plot_folds(folds: list[DateFold], path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, fold in enumerate(folds):
        ax.scatter(fold.train_dates, np.full(len(fold.train_dates), i), marker="s", s=5, label="Train" if i == 0 else None)
        ax.scatter(fold.validation_dates, np.full(len(fold.validation_dates), i), marker="s", s=5, label="Validation" if i == 0 else None)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_cv_plots(train_df: pd.DataFrame) -> None:
    ensure_directories()
    dates = unique_dates(train_df)
    expanding = expanding_date_splits(dates)
    blocking = blocking_date_splits(dates)
    plot_folds(expanding, CV_DIR / "timeseries_cv.png", "Expanding Time Series Cross-Validation")
    plot_folds(blocking, CV_DIR / "blocking_cv.png", "Blocking Time Series Cross-Validation")
