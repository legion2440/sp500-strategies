from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from scripts.config import FEATURES_FILE, MODEL_DIR, TEST_START, ensure_directories
from scripts.cross_validation import assert_temporal_folds, expanding_date_splits, fold_indices, unique_dates
from scripts.features_engineering import FEATURE_COLUMNS, split_train_test


def main() -> None:
    ensure_directories()
    dataset = pd.read_parquet(FEATURES_FILE)
    train, test = split_train_test(dataset)
    train = train.loc[train["target"].notna()].copy()
    test = test.loc[test["target"].notna()].copy()

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"].astype(int)
    selected = joblib.load(MODEL_DIR / "selected_model.pkl")

    folds = expanding_date_splits(unique_dates(train))
    assert_temporal_folds(folds, TEST_START)
    oof = pd.Series(np.nan, index=train.index, dtype=float, name="probability_up")

    for train_idx, val_idx in fold_indices(train, folds):
        estimator = clone(selected)
        estimator.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof.iloc[val_idx] = estimator.predict_proba(X_train.iloc[val_idx])[:, 1]

    final_model = clone(selected)
    final_model.fit(X_train, y_train)
    joblib.dump(final_model, MODEL_DIR / "selected_model.pkl")

    test_probability = final_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    oof_frame = oof.dropna().to_frame()
    oof_frame["scope"] = "train_oof"
    test_frame = pd.DataFrame(
        {"probability_up": test_probability, "scope": "test"},
        index=test.index,
    )

    signal = pd.concat([oof_frame, test_frame]).sort_index()
    signal.to_csv(MODEL_DIR / "ml_signal.csv")
    print(f"OOF signal rows: {len(oof_frame):,}")
    print(f"Test signal rows: {len(test_frame):,}")
    print(f"Saved {MODEL_DIR / 'ml_signal.csv'}")


if __name__ == "__main__":
    main()
