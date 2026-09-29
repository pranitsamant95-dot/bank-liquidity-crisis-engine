from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR


@dataclass
class ForecastResult:
    model: str
    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    residuals: pd.Series
    details: dict


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(series.index[-1] + pd.offsets.Week(weekday=4), periods=horizon, freq="W-FRI")


def _z(confidence: float) -> float:
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def seasonality_justified(series: pd.Series, period: int = 13) -> bool:
    clean = series.dropna()
    return len(clean) >= period * 6 and abs(clean.autocorr(period)) >= 0.25


def available_models(series: pd.Series, multivariate: pd.DataFrame | None = None) -> list[str]:
    models = ["Naive", "ARIMA", "Exponential Smoothing"]
    if seasonality_justified(series):
        models.append("SARIMA")
    if multivariate is not None and len(multivariate.dropna()) >= 80 and multivariate.shape[1] >= 2:
        models.append("VAR")
    return models


def _naive(series: pd.Series, horizon: int, confidence: float) -> ForecastResult:
    idx = _future_index(series, horizon)
    last = float(series.iloc[-1])
    residuals = series.diff().dropna()
    sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else 1.0
    steps = np.sqrt(np.arange(1, horizon + 1))
    mean = pd.Series(last, index=idx)
    margin = _z(confidence) * sigma * steps
    return ForecastResult("Naive", mean, pd.Series(mean.values - margin, idx).clip(0, 100), pd.Series(mean.values + margin, idx).clip(0, 100), residuals, {"assumption": "Random walk; last observation is the forecast."})


def _arima(series: pd.Series, horizon: int, confidence: float, optimize: bool) -> ForecastResult:
    best = None
    candidates = [(1, 1, 1)] if not optimize else [(p, 1, q) for p in (0, 1, 2) for q in (0, 1, 2) if p + q > 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in candidates:
            try:
                fit = ARIMA(series, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
                if np.isfinite(fit.aic) and (best is None or fit.aic < best[0]):
                    best = (fit.aic, order, fit)
            except Exception:
                continue
    if best is None:
        raise RuntimeError("No ARIMA candidate converged.")
    _, order, fit = best
    pred = fit.get_forecast(horizon)
    ci = pred.conf_int(alpha=1 - confidence)
    idx = _future_index(series, horizon)
    return ForecastResult("ARIMA", pd.Series(pred.predicted_mean.values, idx).clip(0, 100), pd.Series(ci.iloc[:, 0].values, idx).clip(0, 100), pd.Series(ci.iloc[:, 1].values, idx).clip(0, 100), pd.Series(fit.resid, index=series.index[-len(fit.resid):]).dropna(), {"order": order, "aic": float(fit.aic), "parameters": {str(k): float(v) for k, v in fit.params.items()}})


def _sarima(series: pd.Series, horizon: int, confidence: float) -> ForecastResult:
    period = 13
    if not seasonality_justified(series, period):
        raise ValueError("Quarterly (13-week) seasonality is not sufficiently supported by the observed autocorrelation.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 0, period), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
    pred = fit.get_forecast(horizon)
    ci = pred.conf_int(alpha=1 - confidence)
    idx = _future_index(series, horizon)
    return ForecastResult("SARIMA", pd.Series(pred.predicted_mean.values, idx).clip(0, 100), pd.Series(ci.iloc[:, 0].values, idx).clip(0, 100), pd.Series(ci.iloc[:, 1].values, idx).clip(0, 100), pd.Series(fit.resid, index=series.index[-len(fit.resid):]).dropna(), {"order": (1, 1, 1), "seasonal_order": (1, 0, 0, period), "aic": float(fit.aic)})


def _ets(series: pd.Series, horizon: int, confidence: float) -> ForecastResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(series, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated").fit(optimized=True)
    idx = _future_index(series, horizon)
    mean = pd.Series(np.asarray(fit.forecast(horizon)), idx).clip(0, 100)
    residuals = pd.Series(np.asarray(fit.resid), index=series.index).dropna()
    sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else 1.0
    margin = _z(confidence) * sigma * np.sqrt(np.arange(1, horizon + 1))
    return ForecastResult("Exponential Smoothing", mean, pd.Series(mean.values - margin, idx).clip(0, 100), pd.Series(mean.values + margin, idx).clip(0, 100), residuals, {"trend": "damped additive", "smoothing_parameters": {k: float(v) for k, v in fit.params.items() if np.isscalar(v) and v is not None}})


def _var(series: pd.Series, multivariate: pd.DataFrame, horizon: int, confidence: float) -> ForecastResult:
    columns = [series.name] + [c for c in multivariate.columns if c != series.name]
    levels = pd.concat([series.rename(series.name), multivariate.drop(columns=[series.name], errors="ignore")], axis=1)
    levels = levels.loc[:, ~levels.columns.duplicated()].dropna().tail(260)
    if len(levels) < 80 or levels.shape[1] < 2:
        raise ValueError("VAR requires at least 80 complete multivariate observations.")
    differences = levels.diff().dropna()
    maxlags = min(4, max(1, len(differences) // 20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = VAR(differences).select_order(maxlags=maxlags)
        lag = int(selection.selected_orders.get("aic") or 1)
        lag = max(1, min(lag, maxlags))
        fit = VAR(differences).fit(lag)
    point, lower_d, upper_d = fit.forecast_interval(differences.values[-lag:], steps=horizon, alpha=1 - confidence)
    last = levels.iloc[-1].values
    level_point = last + np.cumsum(point, axis=0)
    level_lower = last + np.cumsum(lower_d, axis=0)
    level_upper = last + np.cumsum(upper_d, axis=0)
    target = levels.columns.get_loc(series.name)
    idx = _future_index(series, horizon)
    residuals = pd.Series(fit.resid.iloc[:, target].values, index=fit.resid.index)
    return ForecastResult("VAR", pd.Series(level_point[:, target], idx).clip(0, 100), pd.Series(level_lower[:, target], idx).clip(0, 100), pd.Series(level_upper[:, target], idx).clip(0, 100), residuals, {"lag_order": lag, "variables": list(levels.columns), "aic": float(fit.aic)})


def forecast_model(model: str, series: pd.Series, horizon: int, confidence: float = 0.95, multivariate: pd.DataFrame | None = None, optimize: bool = True) -> ForecastResult:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float).tail(520)
    if len(clean) < 40:
        raise ValueError("At least 40 valid weekly stress observations are required for forecasting.")
    if model == "Naive":
        return _naive(clean, horizon, confidence)
    if model == "ARIMA":
        return _arima(clean, horizon, confidence, optimize)
    if model == "SARIMA":
        return _sarima(clean, horizon, confidence)
    if model == "Exponential Smoothing":
        return _ets(clean, horizon, confidence)
    if model == "VAR":
        if multivariate is None:
            raise ValueError("Multivariate component data is unavailable.")
        return _var(clean, multivariate, horizon, confidence)
    raise ValueError(f"Unknown model: {model}")
