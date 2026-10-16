"""Model definitions.

The LSTM architecture and the recursive forecasting logic are adapted from
AbhishekRDJ/Stock-Price-Prediction-Using-Machine-Learning (model.py).
Classical baselines are added so the LSTM can be compared fairly.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import RANDOM_STATE


# ---------------------------------------------------------------- baselines
class NaiveLastValue:
    """Predicts tomorrow = today (0 % change). Any useful model must beat this."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def build_ridge():
    return Ridge(alpha=1e-3)


def build_random_forest():
    return RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1,
                                 random_state=RANDOM_STATE)


# --------------------------------------------------------------------- LSTM
def build_lstm_model(input_shape):
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(X_train, y_train, X_val, y_val, epochs, batch_size):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(RANDOM_STATE)
    model = build_lstm_model((X_train.shape[1], 1))
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train[..., None], y_train,
        validation_data=(X_val[..., None], y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early], verbose=2,
    )
    return model, history


# ---------------------------------------------------------------- forecasting
def predict_future_prices(predict_fn, last_prices: np.ndarray, days: int = 7):
    """Recursive multi-step forecast: each predicted price is fed back as input.

    predict_fn takes a (1, window) array of normalised features and returns the
    predicted next-day % change.
    """
    from .features import window_to_features

    window = np.asarray(last_prices, dtype="float64").copy()
    preds = []
    for _ in range(days):
        pct = float(predict_fn(window_to_features(window).reshape(1, -1)))
        nxt = window[-1] * (1.0 + pct / 100.0)
        preds.append(nxt)
        window = np.append(window[1:], nxt)
    return np.array(preds)


# -------------------------------------------------------------------- metrics
def evaluate(y_true, y_pred, y_prev):
    """Price-scale regression metrics + directional accuracy."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    pred_move = np.sign(y_pred - y_prev)
    # A model that never predicts a move (the naive baseline) has no direction.
    direction = (float(np.mean(np.sign(y_true - y_prev) == pred_move) * 100)
                 if np.any(pred_move != 0) else float("nan"))
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "Direction_acc_%": direction}
