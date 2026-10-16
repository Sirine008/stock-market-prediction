"""Forecast the next N business days with a trained model.

Usage:
    python -m src.predict --ticker AMZN --days 7
    python -m src.predict --ticker BTC --days 14 --model ridge
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loader import load_data
from .models import predict_future_prices

MODEL_FILES = {"lstm": "lstm.keras", "ridge": "ridge.joblib", "rf": "rf.joblib"}
NAME_TO_KEY = {"LSTM": "lstm", "Ridge Regression": "ridge", "Random Forest": "rf"}


def best_model_key(ticker: str) -> str:
    path = config.REPORTS_DIR / f"metrics_{ticker}.json"
    if path.exists():
        best = json.loads(path.read_text())["best_model"]
        return NAME_TO_KEY.get(best, "ridge")  # naive best -> fall back to ridge
    return "ridge"


def load_predict_fn(ticker: str, key: str):
    path = config.MODELS_DIR / f"{ticker}_{MODEL_FILES[key]}"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m src.train` first.")
    if key == "lstm":
        from tensorflow.keras.models import load_model
        model = load_model(path)
        return lambda w: model.predict(w[..., None], verbose=0).ravel()[0]
    model = joblib.load(path)
    return lambda w: model.predict(w)[0]


def forecast(ticker: str, days: int = 7, model_key: str | None = None) -> pd.DataFrame:
    df = load_data()
    model_key = model_key or best_model_key(ticker)
    series = df[ticker]
    last_prices = series.values[-config.WINDOW_SIZE:]
    prices = predict_future_prices(load_predict_fn(ticker, model_key), last_prices, days)
    dates = pd.bdate_range(series.index[-1] + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({"Date": dates, f"{ticker}_predicted": np.round(prices, 2),
                         "model": model_key})


def main():
    p = argparse.ArgumentParser(description="Forecast future prices")
    p.add_argument("--ticker", choices=config.TICKERS, default="AMZN")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--model", choices=list(MODEL_FILES), help="default: best on validation")
    a = p.parse_args()
    out = forecast(a.ticker, a.days, a.model)
    print(out.to_string(index=False))
    path = config.REPORTS_DIR / f"forecast_{a.ticker}.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
