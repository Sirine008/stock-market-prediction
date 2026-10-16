"""Load and clean the historical price data."""
import pandas as pd

from .config import DATA_PATH, TICKERS


def load_data(file_path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV, parse dates, sort chronologically and fill small gaps."""
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").set_index("Date")
    df = df[TICKERS].apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill()
    return df


def chronological_split(series: pd.Series, val_ratio: float, test_ratio: float):
    """Split a time series into train / val / test WITHOUT shuffling.

    Shuffling a time series (as random train_test_split does) leaks future
    information into training and gives over-optimistic scores.
    """
    n = len(series)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))
    return series.iloc[:val_start], series.iloc[val_start:test_start], series.iloc[test_start:]
