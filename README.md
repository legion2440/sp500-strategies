# SP500 Strategies

A leakage-safe machine-learning and backtesting project for the 01-edu / Tomorrow School AI track. The pipeline builds technical features for S&P 500 constituents, selects a classifier with temporal cross-validation, creates an out-of-fold probability signal, converts it into trading strategies, and compares strategy PnL with the S&P 500 benchmark.

## 📋 TOC

- [🚀 Quick start](#-quick-start)
- [📝 About](#-about)
- [🔄 Pipeline](#-pipeline)
- [🧮 Features and target](#-features-and-target)
- [🕒 Temporal cross-validation](#-temporal-cross-validation)
- [🧠 Model selection](#-model-selection)
- [📡 ML signal](#-ml-signal)
- [📈 Backtesting](#-backtesting)
- [✨ Bonus experiments](#-bonus-experiments)
- [🧪 Verification](#-verification)
- [📊 Generated artifacts](#-generated-artifacts)
- [📁 Project structure](#-project-structure)
- [⚠️ Limitations](#️-limitations)
- [🧑‍💻 Author](#-author)

## 🚀 Quick start

### Requirements

- Python 3.11+
- the two CSV datasets supplied by the assignment

Create an environment and install dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS / WSL
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the assignment data in:

```text
data/HistoricalData.csv
data/all_stocks_5yr.csv
```

Run the full mandatory pipeline:

```bash
python run_pipeline.py
```

Or run stages separately:

```bash
python scripts/features_engineering.py
python scripts/gridsearch.py
python scripts/model_selection.py
python scripts/create_signal.py
python scripts/backtest.py
```

Run optional robustness/calibration experiments:

```bash
python scripts/bonus_analysis.py
```

Optional LSTM:

```bash
python -m pip install -r requirements-bonus.txt
python scripts/bonus_lstm.py
```

## 📝 About

The project implements the assignment flow without using future information during feature construction or model selection:

1. compute ticker-level technical features;
2. split chronologically, with the test period starting in 2017;
3. implement both expanding and blocking temporal cross-validation;
4. compare several ML pipelines using train folds only;
5. save the selected model and its hyperparameters;
6. generate the train signal strictly out-of-fold;
7. fit the final selected pipeline on all train data and predict the untouched test period;
8. convert probabilities into daily investment weights;
9. multiply the day-D signal by return(D+1,D+2);
10. compare strategy PnL with the S&P 500.

No performance number is claimed in this README before a local run generates the actual artifacts.

## 🔄 Pipeline

```text
all_stocks_5yr.csv
        |
        v
features_engineering.py
        |
        v
data/sp500_features.parquet
        |
        v
gridsearch.py
        |
        v
selected_model.pkl / selected_model.txt
        |
        +----------------------+
        |                      |
        v                      v
model_selection.py       create_signal.py
        |                      |
        v                      v
CV metrics + FI           ml_signal.csv
                               |
                               v
                          backtest.py
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
           strategy.png                  report.md
```

## 🧮 Features and target

All rolling indicators are computed independently for each ticker.

Feature families:

- returns over 1, 2, 5, 10 and 20 trading days;
- log return, intraday range, open/close move and overnight gap;
- SMA and EMA distance ratios;
- RSI;
- MACD, signal and histogram;
- rate of change;
- rolling volatility;
- ATR;
- Bollinger width and percent-B;
- volume change and relative volume.

The target follows the assignment exactly:

```text
target(D) = sign(return(D+1, D+2))
```

The generated table uses a `(date, ticker)` MultiIndex.

## 🕒 Temporal cross-validation

Two date-based schemes are implemented.

### Expanding split

```text
Fold 1: [ TRAIN -------- ] [ VAL ]
Fold 2: [ TRAIN ------------- ] [ VAL ]
Fold 3: [ TRAIN ------------------ ] [ VAL ]
```

### Rolling blocking split

```text
Fold 1: [ TRAIN -------- ] [ VAL ]
Fold 2:      [ TRAIN -------- ] [ VAL ]
Fold 3:           [ TRAIN -------- ] [ VAL ]
```

Both enforce:

- 10 folds;
- at least 505 trading days in each training window;
- validation strictly after training;
- no train/validation date overlap;
- no validation date inside the 2017+ test period;
- all rows from one date remain on the same side of the split.

Grid search uses the expanding split. Both schemes are plotted for audit evidence.

## 🧠 Model selection

Candidate pipelines:

- Logistic Regression;
- PCA + Logistic Regression;
- Random Forest;
- HistGradientBoostingClassifier.

All pipelines use median imputation. Linear models use standard scaling; the PCA variant also performs dimensionality reduction.

Selection metric: mean validation ROC-AUC across temporal folds.

The 2017+ test period is excluded from hyperparameter search and pipeline selection.

Selected-model artifacts:

```text
results/selected-model/selected_model.pkl
results/selected-model/selected_model.txt
```

## 📡 ML signal

Train probabilities are created fold by fold:

```text
fold 1 train -> fold 1 validation predictions
fold 2 train -> fold 2 validation predictions
...
concatenate validation predictions
```

The pipeline is never fitted once and used to predict the entire train dataset.

After OOF generation, the selected pipeline is fitted on the complete train period and used for the test signal.

Output:

```text
results/selected-model/ml_signal.csv
```

with `date`, `ticker`, `probability_up` and `scope` (`train_oof` or `test`).

## 📈 Backtesting

Implemented strategies:

- binary long-only: `p > 0.50`;
- ternary long/short: long above 0.55, short below 0.45;
- probability-weighted long-only;
- top-10 / bottom-10 stock picking.

Daily gross exposure is normalized:

```text
sum(abs(weights)) = 1
```

PnL follows the predicted interval:

```text
PnL(D, i) = weight(D, i) * return(D+1, D+2, i)
```

The default plot compares the stock-picking strategy and S&P 500 cumulative PnL on the same axis and marks the train/test boundary. Because the subject assumes the same $1 of capital each day, daily PnL is summed rather than reinvested.

Metrics:

- PnL;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- Sortino ratio;
- max drawdown;
- Calmar ratio;
- win rate.

## ✨ Bonus experiments

### Transaction costs and turnover

`bonus_analysis.py` tests 0, 5, 10 and 20 bps cost assumptions and deducts costs from daily turnover.

### Robustness grid

Long-only thresholds:

```text
0.50 / 0.52 / 0.55 / 0.60
```

Stock-picking sizes:

```text
K = 5 / 10 / 20 / 50
```

### Probability calibration

The bonus layer saves a 10-bin calibration table and Brier score.

### Optional LSTM

The LSTM uses rolling 20-day feature sequences and is intentionally isolated from the mandatory scikit-learn path.

```text
20 days x features
        |
        v
      LSTM
        |
        v
      P(up)
```

## 🧪 Verification

Run:

```bash
pytest -q
```

Tests cover:

- target construction within ticker;
- future-price mutation not changing older features;
- 10-fold expanding CV;
- 10-fold rolling blocking CV;
- train-before-validation ordering;
- daily gross-exposure normalization;
- market-neutral top-K / bottom-K weights.

After the local pipeline run, inspect:

```text
results/cross-validation/timeseries_cv.png
results/cross-validation/blocking_cv.png
results/cross-validation/ml_metrics_train.csv
results/cross-validation/top_10_feature_importance.csv
results/selected-model/selected_model.txt
results/selected-model/ml_signal.csv
results/strategy/results.csv
results/strategy/strategy.png
results/strategy/report.md
```

## 📊 Generated artifacts

```text
results/
├── cross-validation/
│   ├── blocking_cv.png
│   ├── timeseries_cv.png
│   ├── metric_train.csv
│   ├── metric_train.png
│   ├── ml_metrics_train.csv
│   └── top_10_feature_importance.csv
├── selected-model/
│   ├── gridsearch_results.csv
│   ├── ml_signal.csv
│   ├── selected_model.pkl
│   └── selected_model.txt
├── strategy/
│   ├── backtest_metadata.json
│   ├── daily_returns.csv
│   ├── report.md
│   ├── results.csv
│   └── strategy.png
└── bonus/
    ├── calibration.csv
    ├── calibration_summary.txt
    ├── robustness.csv
    └── lstm/
```

Generated datasets and result artifacts are ignored by Git by default.

## 📁 Project structure

```text
sp500-strategies/
├── data/
│   └── .gitkeep
├── results/
│   ├── bonus/
│   ├── cross-validation/
│   ├── selected-model/
│   └── strategy/
├── scripts/
│   ├── backtest.py
│   ├── bonus_analysis.py
│   ├── bonus_lstm.py
│   ├── config.py
│   ├── create_signal.py
│   ├── cross_validation.py
│   ├── data.py
│   ├── features_engineering.py
│   ├── gridsearch.py
│   ├── metrics.py
│   ├── model_selection.py
│   ├── modeling.py
│   ├── report.py
│   └── strategy.py
├── tests/
│   ├── test_cross_validation.py
│   ├── test_features.py
│   └── test_strategy.py
├── .gitignore
├── README.md
├── requirements-bonus.txt
├── requirements.txt
└── run_pipeline.py
```

## ⚠️ Limitations

- The assignment dataset simplifies historical S&P 500 membership and therefore contains survivor bias.
- The supplied historical period should not be interpreted as evidence of current live-trading performance.
- Transaction costs use a simple turnover-based bps model; slippage, spread, borrow fees and market impact are not modeled.
- Technical indicators do not guarantee predictive power; conclusions should come from temporal CV and the untouched test period.
- The optional LSTM is a bonus comparison and is not part of mandatory model selection.
- No repository performance values are asserted before the pipeline is run locally on the supplied data.

## 🧑‍💻 Author

- Nazar Yestayev (@nyestaye / @legion2440)
