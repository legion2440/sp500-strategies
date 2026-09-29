from __future__ import annotations

from collections import OrderedDict

from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.config import RANDOM_STATE


def candidate_pipelines() -> OrderedDict[str, tuple[Pipeline, dict]]:
    return OrderedDict(
        {
            "logistic": (
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                    ]
                ),
                {
                    "model__C": [0.1, 1.0, 10.0],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "pca_logistic": (
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(random_state=RANDOM_STATE)),
                        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                    ]
                ),
                {
                    "pca__n_components": [0.90, 0.95],
                    "model__C": [0.1, 1.0, 10.0],
                },
            ),
            "random_forest": (
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=400,
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                                max_features="sqrt",
                            ),
                        ),
                    ]
                ),
                {
                    "model__max_depth": [6, 12, None],
                    "model__min_samples_leaf": [1, 5, 20],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "hist_gradient_boosting": (
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            HistGradientBoostingClassifier(
                                random_state=RANDOM_STATE,
                                early_stopping=False,
                            ),
                        ),
                    ]
                ),
                {
                    "model__learning_rate": [0.03, 0.08],
                    "model__max_leaf_nodes": [15, 31],
                    "model__l2_regularization": [0.0, 1.0],
                },
            ),
        }
    )
