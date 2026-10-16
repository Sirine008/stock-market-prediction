"""Train and compare models for one or all tickers.

Usage:
    python -m src.train                 # all tickers
    python -m src.train --ticker AMZN   # one ticker
    python -m src.train --no-lstm       # skip TensorFlow (fast)
"""
import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .data_loader import chronological_split, load_data  # noqa: E402
from .features import build_datasets, returns_to_prices  # noqa: E402
from .models import (NaiveLastValue, build_random_forest, build_ridge,  # noqa: E402
                     evaluate, train_lstm)


def train_ticker(df: pd.DataFrame, ticker: str, use_lstm: bool = True) -> dict:
    print(f"\n========== {ticker} ==========")
    series = df[ticker]
    train, val, test = chronological_split(series, config.VAL_RATIO, config.TEST_RATIO)
    print(f"train {train.index[0].date()} -> {train.index[-1].date()} ({len(train)})")
    print(f"val   {val.index[0].date()} -> {val.index[-1].date()} ({len(val)})")
    print(f"test  {test.index[0].date()} -> {test.index[-1].date()} ({len(test)})")

    sp = build_datasets(train, val, test, config.WINDOW_SIZE)
    X_tr, y_tr = sp["train"]["X"], sp["train"]["y"]
    X_va, y_va = sp["val"]["X"], sp["val"]["y"]
    X_te = sp["test"]["X"]

    models = {
        "Naive (last value)": NaiveLastValue(),
        "Ridge Regression": build_ridge(),
        "Random Forest": build_random_forest(),
    }
    preds_val, preds_test = {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds_val[name] = m.predict(X_va)
        preds_test[name] = m.predict(X_te)

    lstm_model = None
    if use_lstm:
        lstm_model, history = train_lstm(X_tr, y_tr, X_va, y_va,
                                         config.LSTM_EPOCHS, config.LSTM_BATCH_SIZE)
        preds_val["LSTM"] = lstm_model.predict(X_va[..., None], verbose=0).ravel()
        preds_test["LSTM"] = lstm_model.predict(X_te[..., None], verbose=0).ravel()
        _plot_loss(history, ticker)

    # ---- metrics in real price units
    v, t = sp["val"], sp["test"]
    price_test = {k: returns_to_prices(p, t["last"]) for k, p in preds_test.items()}
    results = {}
    for name in preds_test:
        results[name] = {
            "val": evaluate(v["price"], returns_to_prices(preds_val[name], v["last"]), v["last"]),
            "test": evaluate(t["price"], price_test[name], t["last"]),
        }
    best = min(results, key=lambda k: results[k]["val"]["RMSE"])
    print(pd.DataFrame({k: v["test"] for k, v in results.items()}).T.round(3))
    print(f"Best model on validation: {best}")

    # ---- save artefacts
    joblib.dump(models["Ridge Regression"], config.MODELS_DIR / f"{ticker}_ridge.joblib")
    joblib.dump(models["Random Forest"], config.MODELS_DIR / f"{ticker}_rf.joblib")
    if lstm_model is not None:
        lstm_model.save(config.MODELS_DIR / f"{ticker}_lstm.keras")
    with open(config.REPORTS_DIR / f"metrics_{ticker}.json", "w") as f:
        json.dump({"ticker": ticker, "best_model": best, "results": results}, f, indent=2)

    backtest = pd.DataFrame({"Date": t["dates"], "Actual": t["price"],
                             **{k: v for k, v in price_test.items()}})
    backtest.round(4).to_csv(config.REPORTS_DIR / f"backtest_{ticker}.csv", index=False)

    _plot_test(t["dates"], t["price"], price_test, ticker)
    return {"ticker": ticker, "best": best, "results": results}


def _plot_test(dates, y_true, preds, ticker):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, y_true, label="Actual", color="black", linewidth=2)
    for name, p in preds.items():
        if name.startswith("Naive"):
            continue
        ax.plot(dates, p, label=name, alpha=0.8)
    ax.set_title(f"{ticker} — test set: actual vs predicted (next-day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{ticker}_test_predictions.png", dpi=120)
    plt.close(fig)


def _plot_loss(history, ticker):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="validation")
    ax.set_title(f"{ticker} — LSTM training loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{ticker}_lstm_loss.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument("--ticker", choices=config.TICKERS, help="train only this ticker")
    parser.add_argument("--no-lstm", action="store_true", help="skip the LSTM model")
    args = parser.parse_args()

    df = load_data()
    tickers = [args.ticker] if args.ticker else config.TICKERS
    rows = []
    for t in tickers:
        out = train_ticker(df, t, use_lstm=not args.no_lstm)
        for model_name, r in out["results"].items():
            rows.append({"ticker": t, "model": model_name, **r["test"],
                         "best_on_val": model_name == out["best"]})
    summary = pd.DataFrame(rows)
    summary_path = config.REPORTS_DIR / "metrics_summary.csv"
    if args.ticker and summary_path.exists():
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[old.ticker != args.ticker], summary])
    summary.round(4).to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
