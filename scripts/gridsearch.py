from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from scripts.config import FEATURES_FILE, MODEL_DIR, TEST_START, ensure_directories
from scripts.cross_validation import assert_temporal_folds, expanding_date_splits, fold_indices, unique_dates
from scripts.features_engineering import FEATURE_COLUMNS, split_train_test
from scripts.modeling import candidate_pipelines


def load_train_matrix() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    dataset = pd.read_parquet(FEATURES_FILE)
    train, _ = split_train_test(dataset)
    train = train.loc[train["target"].notna()].copy()
    X = train[FEATURE_COLUMNS]
    y = train["target"].astype(int)
    return X, y, train


def main() -> None:
    ensure_directories()
    X, y, train = load_train_matrix()

    folds = expanding_date_splits(unique_dates(train))
    assert_temporal_folds(folds, TEST_START)
    cv = list(fold_indices(train, folds))

    all_rows: list[pd.DataFrame] = []
    best_name = None
    best_score = float("-inf")
    best_estimator = None
    best_params = None

    for name, (pipeline, grid) in candidate_pipelines().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X, y)
        rows = pd.DataFrame(search.cv_results_)
        rows.insert(0, "candidate", name)
        all_rows.append(rows)

        if search.best_score_ > best_score:
            best_name = name
            best_score = float(search.best_score_)
            best_estimator = clone(search.best_estimator_)
            best_params = search.best_params_

    if best_estimator is None:
        raise RuntimeError("Grid search did not produce a selected model")

    best_estimator.fit(X, y)
    joblib.dump(best_estimator, MODEL_DIR / "selected_model.pkl")
    pd.concat(all_rows, ignore_index=True).to_csv(MODEL_DIR / "gridsearch_results.csv", index=False)

    metadata = {
        "candidate": best_name,
        "selection_metric": "roc_auc",
        "mean_cv_roc_auc": best_score,
        "best_params": best_params,
        "cv": "expanding date-based time series split",
        "n_folds": len(cv),
        "test_start": TEST_START,
        "features": FEATURE_COLUMNS,
    }
    (MODEL_DIR / "selected_model.txt").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Selected: {best_name}")
    print(f"Mean CV ROC-AUC: {best_score:.6f}")
    print(f"Saved model to {MODEL_DIR / 'selected_model.pkl'}")


if __name__ == "__main__":
    main()
