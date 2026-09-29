from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from scripts.config import CV_DIR, FEATURES_FILE, MODEL_DIR, RANDOM_STATE, TEST_START, ensure_directories
from scripts.cross_validation import (
    assert_temporal_folds,
    expanding_date_splits,
    fold_indices,
    save_cv_plots,
    unique_dates,
)
from scripts.features_engineering import FEATURE_COLUMNS, split_train_test


def _metrics(y_true, probability) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "logloss": log_loss(y_true, probability, labels=[0, 1]),
    }


def _feature_importance(estimator, X_val: pd.DataFrame, y_val: pd.Series) -> pd.Series:
    model = estimator.named_steps["model"]
    if "pca" in estimator.named_steps:
        result = permutation_importance(
            estimator,
            X_val,
            y_val,
            scoring="roc_auc",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return pd.Series(result.importances_mean, index=X_val.columns)

    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=X_val.columns)
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=X_val.columns)

    result = permutation_importance(
        estimator,
        X_val,
        y_val,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X_val.columns)


def main() -> None:
    ensure_directories()
    dataset = pd.read_parquet(FEATURES_FILE)
    train, _ = split_train_test(dataset)
    train = train.loc[train["target"].notna()].copy()
    X = train[FEATURE_COLUMNS]
    y = train["target"].astype(int)

    save_cv_plots(train)

    selected = joblib.load(MODEL_DIR / "selected_model.pkl")
    folds = expanding_date_splits(unique_dates(train))
    assert_temporal_folds(folds, TEST_START)

    metric_rows = []
    importance_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices(train, folds), start=1):
        estimator = clone(selected)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        estimator.fit(X_train, y_train)

        train_prob = estimator.predict_proba(X_train)[:, 1]
        val_prob = estimator.predict_proba(X_val)[:, 1]
        for split_name, values in (
            ("train", _metrics(y_train, train_prob)),
            ("validation", _metrics(y_val, val_prob)),
        ):
            metric_rows.append({"fold": fold_id, "split": split_name, **values})

        importance = _feature_importance(estimator, X_val, y_val)
        top = importance.sort_values(ascending=False).head(10)
        for rank, (feature, value) in enumerate(top.items(), start=1):
            importance_rows.append(
                {"fold": fold_id, "rank": rank, "feature": feature, "importance": float(value)}
            )

    metrics = pd.DataFrame(metric_rows).set_index(["fold", "split"])
    metrics.to_csv(CV_DIR / "ml_metrics_train.csv")
    pd.DataFrame(importance_rows).to_csv(CV_DIR / "top_10_feature_importance.csv", index=False)

    val_auc = metrics.xs("validation", level="split")["auc"]
    train_auc = metrics.xs("train", level="split")["auc"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(val_auc) + 1)
    width = 0.38
    ax.bar(x - width / 2, train_auc.values, width=width, label="Train AUC")
    ax.bar(x + width / 2, val_auc.values, width=width, label="Validation AUC")
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Selected model ROC-AUC by time-series fold")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CV_DIR / "metric_train.png", dpi=160)
    plt.close(fig)

    summary = metrics.groupby(level="split").median(numeric_only=True)
    summary.to_csv(CV_DIR / "metric_train.csv")
    print(summary)


if __name__ == "__main__":
    main()
