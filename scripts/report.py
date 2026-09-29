from __future__ import annotations

import json

import pandas as pd

from scripts.config import MODEL_DIR, STRATEGY_DIR
from scripts.features_engineering import FEATURE_COLUMNS


def main() -> None:
    model_meta = json.loads((MODEL_DIR / "selected_model.txt").read_text(encoding="utf-8"))
    results = pd.read_csv(STRATEGY_DIR / "results.csv")

    selected = model_meta["candidate"]
    params = json.dumps(model_meta["best_params"], indent=2, sort_keys=True)
    metrics_table = results.to_markdown(index=False)

    text = f"""# SP500 Strategies Report

## Features

The model uses {len(FEATURE_COLUMNS)} leakage-safe features computed independently for each ticker.

Feature set:

{", ".join(f"`{name}`" for name in FEATURE_COLUMNS)}

The target at day `D` is the sign of the return between `D+1` and `D+2`.

## Pipeline

Selected candidate: `{selected}`.

Best parameters:

```json
{params}
```

The candidate pipelines use median imputation. Linear models additionally use standard scaling; the PCA candidate also performs dimensionality reduction before logistic regression.

## Cross-validation

Model selection uses a date-based expanding time-series split with {model_meta["n_folds"]} folds. The first training window contains at least 504 trading days (>2 years), validation always follows training, and the test period beginning at `{model_meta["test_start"]}` is excluded from model selection.

Both required CV visualizations are generated at:

- `results/cross-validation/timeseries_cv.png`
- `results/cross-validation/blocking_cv.png`

Fold metrics and feature importance are written to:

- `results/cross-validation/ml_metrics_train.csv`
- `results/cross-validation/metric_train.csv`
- `results/cross-validation/metric_train.png`
- `results/cross-validation/top_10_feature_importance.csv`

## Strategy

The main comparison includes binary long-only, ternary long/short, probability-weighted long-only, and top-10 / bottom-10 stock picking.

Daily gross exposure is normalized so `sum(abs(weights)) = 1`. The PnL contribution for a signal created on day `D` is `weight(D, i) * return(D+1, D+2, i)`.

The primary plotted strategy is `stock_picking_10`.

## Results

{metrics_table}

![Strategy vs S&P 500](strategy.png)

## Bonus experiments

Optional analysis includes transaction costs, turnover, threshold/K robustness, probability calibration, and an LSTM experiment.

Generated bonus artifacts are stored under `results/bonus/`.
"""
    (STRATEGY_DIR / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
