from __future__ import annotations

import numpy as np
import pandas as pd

from models.forecasting import forecast_model
from utils.preprocessing import safe_mape


def expanding_window_backtest(series: pd.Series, models: list[str], multivariate: pd.DataFrame | None = None, origins: int = 6, block_horizon: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Past-only expanding-window validation over non-overlapping recent forecast blocks."""
    y = series.dropna().astype(float).tail(320)
    minimum_train = 80
    possible = max(0, (len(y) - minimum_train) // block_horizon)
    n_origins = min(origins, possible)
    if n_origins < 2:
        return pd.DataFrame(), pd.DataFrame(), {"failures": {"all": "Insufficient observations for expanding-window validation."}}
    first_origin = len(y) - n_origins * block_horizon
    records = []
    failures: dict[str, str] = {}
    for model in models:
        for start in range(first_origin, len(y), block_horizon):
            train = y.iloc[:start]
            actual = y.iloc[start:start + block_horizon]
            if actual.empty:
                continue
            mv_train = None
            if multivariate is not None:
                mv_train = multivariate.loc[:train.index[-1]]
            try:
                result = forecast_model(model, train, len(actual), 0.95, mv_train, optimize=False)
                predicted = pd.Series(result.mean.values[:len(actual)], index=actual.index)
                for date, observed, estimate in zip(actual.index, actual.values, predicted.values):
                    records.append({"Date": date, "Model": model, "Actual": float(observed), "Forecast": float(estimate), "Error": float(observed - estimate)})
            except Exception as exc:
                failures[model] = f"{type(exc).__name__}: {exc}"
                break
    predictions = pd.DataFrame(records)
    if predictions.empty:
        return pd.DataFrame(), predictions, {"failures": failures}
    rows = []
    for model, group in predictions.groupby("Model"):
        error = group["Actual"] - group["Forecast"]
        rows.append({
            "Model": model,
            "MAE": float(error.abs().mean()),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAPE (%)": safe_mape(group["Actual"], group["Forecast"]),
            "Forecast Horizon": f"1–{block_horizon} weeks",
            "Predictions": len(group),
        })
    metrics = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    if not metrics.empty:
        metrics["Selected"] = False
        metrics.loc[0, "Selected"] = True
    return metrics, predictions, {"failures": failures, "origins": n_origins, "criterion": "Lowest expanding-window RMSE; MAE breaks ties."}
