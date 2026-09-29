from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.config import BONUS_DIR, FEATURES_FILE, TEST_START, RANDOM_STATE, ensure_directories
from scripts.features_engineering import FEATURE_COLUMNS, split_train_test


def make_sequences(df: pd.DataFrame, window: int):
    Xs, ys, keys = [], [], []
    for ticker, group in df.reset_index().groupby("ticker"):
        group = group.sort_values("date").dropna(subset=["target"])
        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = group["target"].to_numpy(dtype=np.float32)
        dates = group["date"].to_numpy()
        for i in range(window - 1, len(group)):
            Xs.append(values[i - window + 1 : i + 1])
            ys.append(labels[i])
            keys.append((pd.Timestamp(dates[i]), ticker))
    return np.asarray(Xs), np.asarray(ys), keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional LSTM experiment")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("Install optional dependency: pip install -r requirements-bonus.txt") from exc

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    ensure_directories()

    dataset = pd.read_parquet(FEATURES_FILE)
    train, test = split_train_test(dataset)
    median = train[FEATURE_COLUMNS].median()
    train = train.copy()
    test = test.copy()
    train[FEATURE_COLUMNS] = train[FEATURE_COLUMNS].fillna(median)
    test[FEATURE_COLUMNS] = test[FEATURE_COLUMNS].fillna(median)

    scaler = StandardScaler()
    scaler.fit(train[FEATURE_COLUMNS])
    train.loc[:, FEATURE_COLUMNS] = scaler.transform(train[FEATURE_COLUMNS])
    test.loc[:, FEATURE_COLUMNS] = scaler.transform(test[FEATURE_COLUMNS])

    X_train, y_train, _ = make_sequences(train, args.window)
    X_test, y_test, keys = make_sequences(test, args.window)

    model = tf.keras.Sequential([
        tf.keras.layers.Input((args.window, len(FEATURE_COLUMNS))),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"])
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=256,
        shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=2,
    )
    test_metrics = dict(zip(model.metrics_names, model.evaluate(X_test, y_test, verbose=0)))
    output_dir = BONUS_DIR / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(keys, columns=["date", "ticker"]).assign(
        probability_up=model.predict(X_test, verbose=0).ravel()
    ).to_csv(output_dir / "test_signal.csv", index=False)


if __name__ == "__main__":
    main()
