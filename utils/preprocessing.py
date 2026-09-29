from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def validate_time_series(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = frame.copy()
    original_rows = len(data)
    duplicates = int(data.index.duplicated().sum())
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")
    missing = data.isna().sum().to_dict()
    numeric_share = float(data.notna().sum().sum() / max(data.size, 1))
    inferred = pd.infer_freq(data.index) if len(data.index) >= 3 else None
    report = {
        "Rows": len(data),
        "Original rows": original_rows,
        "Duplicate timestamps removed": duplicates,
        "Start": data.index.min(),
        "End": data.index.max(),
        "Inferred frequency": inferred or "Irregular/unknown",
        "Numeric coverage": numeric_share,
        "Missing values": missing,
    }
    return data, report


def robust_clip(series: pd.Series, lower: float = 0.005, upper: float = 0.995) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 20:
        return series
    lo, hi = valid.quantile([lower, upper])
    return series.clip(lo, hi)


def adf_report(series: pd.Series) -> dict:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 25 or clean.nunique() < 3:
        return {"available": False, "message": "At least 25 non-constant observations are required."}
    try:
        statistic, pvalue, used_lag, nobs, critical, _ = adfuller(clean, autolag="AIC")
        evidence = (
            "The series provides evidence against a unit root at the 5% significance level; it is consistent with stationarity."
            if pvalue < 0.05
            else "The test does not reject a unit root at the 5% level; differencing or another transformation may be appropriate."
        )
        return {"available": True, "statistic": statistic, "pvalue": pvalue, "used_lag": used_lag, "nobs": nobs, "critical": critical, "interpretation": evidence}
    except Exception as exc:
        return {"available": False, "message": f"ADF test failed: {exc}"}


def safe_mape(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    mask = np.isfinite(a) & np.isfinite(f) & (np.abs(a) > 1e-8)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100)
