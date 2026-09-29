from __future__ import annotations

import pandas as pd

from scripts.config import INDEX_FILE, STOCKS_FILE


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    aliases = {
        "name": "ticker",
        "symbol": "ticker",
        "vol.": "volume",
        "adj_close": "close",
        "close/last": "close",
    }
    return out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})


def load_stocks(path=STOCKS_FILE) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing stock columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
            )
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df.sort_values(["ticker", "date"]).drop_duplicates(["date", "ticker"], keep="last")


def load_index(path=INDEX_FILE) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))

    if "date" not in df.columns:
        for candidate in ("datetime", "timestamp"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break

    close_candidates = ("close", "price", "last")
    close_col = next((c for c in close_candidates if c in df.columns), None)
    if "date" not in df.columns or close_col is None:
        raise ValueError("HistoricalData.csv must contain a date column and a close/price/last column")

    if close_col != "close":
        df = df.rename(columns={close_col: "close"})

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["close"].dtype == object:
        df["close"] = (
            df["close"].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
