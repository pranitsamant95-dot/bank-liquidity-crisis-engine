from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


COMPONENT_LABELS = {
    "deposit_deterioration": "Deposit growth deterioration",
    "liquid_asset_deterioration": "Liquid-asset deterioration",
    "reserve_deterioration": "Reserve deterioration",
    "funding_cost": "Funding cost",
    "credit_spread": "Credit spread",
    "market_volatility": "Market volatility",
    "curve_inversion": "Yield-curve inversion",
}


def _past_rolling_z(series: pd.Series, window: int = 156, min_periods: int = 52) -> pd.Series:
    """Compare the current value with rolling moments available before the current week."""
    history = series.shift(1)
    mean = history.rolling(window, min_periods=min_periods).mean()
    std = history.rolling(window, min_periods=min_periods).std(ddof=0).replace(0, np.nan)
    return ((series - mean) / std).clip(-4, 4)


def _usable(frame: pd.DataFrame, column: str) -> bool:
    return column in frame and frame[column].notna().sum() >= 65


def construct_liquidity_index(raw: pd.DataFrame, window: int = 156) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x = raw.copy().sort_index()
    transformed = pd.DataFrame(index=x.index)

    # 13-week changes in balance-sheet stocks are annualized. Components are optional so
    # a single unavailable provider series does not crash the whole application.
    if _usable(x, "deposits"):
        transformed["deposit_deterioration"] = -(x["deposits"].pct_change(13, fill_method=None) * 400 / 13)
    if _usable(x, "securities"):
        transformed["liquid_asset_deterioration"] = -(x["securities"].pct_change(13, fill_method=None) * 400 / 13)
    if _usable(x, "reserves"):
        transformed["reserve_deterioration"] = -(x["reserves"].pct_change(13, fill_method=None) * 400 / 13)
    if _usable(x, "fed_funds"):
        transformed["funding_cost"] = x["fed_funds"]
    if _usable(x, "credit_spread"):
        transformed["credit_spread"] = x["credit_spread"]
    if _usable(x, "vix"):
        transformed["market_volatility"] = x["vix"]
    if _usable(x, "treasury_10y") and _usable(x, "treasury_3m"):
        transformed["curve_inversion"] = -(x["treasury_10y"] - x["treasury_3m"])

    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    if transformed.shape[1] < 4:
        missing = [COMPONENT_LABELS[k] for k in COMPONENT_LABELS if k not in transformed]
        raise ValueError("At least four usable stress components are required. Unavailable: " + ", ".join(missing))

    zscores = transformed.apply(lambda s: _past_rolling_z(s, window=window))
    availability = zscores.notna().sum(axis=1)
    minimum_components = min(4, transformed.shape[1])
    composite_z = zscores.mean(axis=1, skipna=True).where(availability >= minimum_components)
    score = pd.Series(norm.cdf(composite_z) * 100, index=x.index, name="Stress Index").clip(0, 100)

    out = x.copy()
    out["Stress Index"] = score
    out["Composite z"] = composite_z
    out["Available components"] = availability
    out["Regime"] = classify_regime(score)
    contributions = zscores.div(availability.replace(0, np.nan), axis=0)
    contributions.columns = [COMPONENT_LABELS[c] for c in contributions.columns]

    unavailable = [COMPONENT_LABELS[k] for k in COMPONENT_LABELS if k not in transformed]
    metadata = {
        "method": "Equal-weight mean of available past-only rolling z-scores, mapped through the standard normal CDF to 0–100.",
        "window": window,
        "minimum_history": 52,
        "weights": f"Equal among available components; at least {minimum_components} components are required.",
        "unavailable_components": unavailable,
        "thresholds": {"Normal": [0, 25], "Watch": [25, 50], "Elevated Stress": [50, 75], "Severe Stress": [75, 100]},
    }
    return out, contributions, {"transformed": transformed, "zscores": zscores, **metadata}


def classify_regime(score: pd.Series) -> pd.Series:
    return pd.cut(score, bins=[-np.inf, 25, 50, 75, np.inf], labels=["NORMAL", "WATCH", "ELEVATED STRESS", "SEVERE STRESS"], right=False).astype("string")


def regime_for_value(value: float) -> str:
    if not np.isfinite(value):
        return "UNAVAILABLE"
    if value < 25:
        return "NORMAL"
    if value < 50:
        return "WATCH"
    if value < 75:
        return "ELEVATED STRESS"
    return "SEVERE STRESS"
