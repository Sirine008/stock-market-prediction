"""Turn a price series into supervised (window -> next value) samples.

Each 60-day window is normalised by its own last price, and the target is the
next-day % change. This makes samples scale-free: a model trained when AMZN
was at $300 still works when it is at $1,800. Predicting raw scaled prices
(as the reference repo does) breaks as soon as prices leave the training range.
"""
import numpy as np
import pandas as pd


def window_to_features(window: np.ndarray) -> np.ndarray:
    """Express a price window as % distance from its last price."""
    window = np.asarray(window, dtype="float64")
    return (window / window[..., -1:] - 1.0) * 100.0


def make_windows(prices: np.ndarray, window: int):
    """X[i] = normalised prices[i-window:i], y[i] = % change to prices[i]."""
    X, y, last = [], [], []
    for i in range(window, len(prices)):
        w = prices[i - window:i]
        X.append(window_to_features(w))
        y.append((prices[i] / w[-1] - 1.0) * 100.0)
        last.append(w[-1])
    return np.array(X), np.array(y), np.array(last)


def build_datasets(train: pd.Series, val: pd.Series, test: pd.Series, window: int):
    """Windowed X / y / last_price / target_price / dates for each split.

    Val and test windows may look back into the previous split (that history
    is known at prediction time), but targets never overlap.
    """
    full = pd.concat([train, val, test])
    prices = full.values.astype("float64")
    X, y, last = make_windows(prices, window)
    target = prices[window:]
    dates = full.index[window:]

    n_tr = len(train) - window
    n_va = len(val)
    cuts = {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, None)}
    return {k: {"X": X[s], "y": y[s], "last": last[s], "price": target[s], "dates": dates[s]}
            for k, s in cuts.items()}


def returns_to_prices(pct_change: np.ndarray, last_price: np.ndarray) -> np.ndarray:
    return last_price * (1.0 + np.asarray(pct_change) / 100.0)
