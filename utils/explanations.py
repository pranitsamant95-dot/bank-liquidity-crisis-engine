from __future__ import annotations

import numpy as np
import pandas as pd


def indicator_diagnostics(transformed: pd.DataFrame, zscores: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    latest_date = zscores.dropna(how="all").index.max()
    for raw_name, label in zip(zscores.columns, contributions.columns):
        z = zscores[raw_name].dropna()
        t = transformed[raw_name].dropna()
        if z.empty or t.empty:
            continue
        latest_z = float(z.iloc[-1])
        recent_change = float(t.iloc[-1] - t.iloc[-5]) if len(t) >= 5 else np.nan
        percentile = float((t.rank(pct=True).iloc[-1]) * 100)
        contribution = float(contributions.loc[latest_date, label]) if latest_date in contributions.index and pd.notna(contributions.loc[latest_date, label]) else np.nan
        rows.append({
            "Indicator": label,
            "Direction": "Raises stress" if latest_z > 0 else "Reduces stress",
            "Recent 4-week change": recent_change,
            "Historical percentile": percentile,
            "Rolling z-score": latest_z,
            "Index contribution (z units)": contribution,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["Absolute contribution"] = result["Index contribution (z units)"].abs()
        result = result.sort_values("Absolute contribution", ascending=False).drop(columns="Absolute contribution")
    return result


def dynamic_warning_explanation(index: pd.Series, diagnostics: pd.DataFrame, forecast, previous_regime: str, current_regime: str) -> str:
    clean = index.dropna()
    if clean.empty:
        return "A warning explanation cannot be generated because the stress index is unavailable."
    change_window = min(18, len(clean) - 1)
    delta = float(clean.iloc[-1] - clean.iloc[-1 - change_window]) if change_window > 0 else 0.0
    movement = "increased" if delta > 1 else "decreased" if delta < -1 else "was broadly stable"
    drivers = []
    if not diagnostics.empty:
        for _, row in diagnostics.head(3).iterrows():
            direction = "stress-raising" if row["Rolling z-score"] > 0 else "stress-reducing"
            drivers.append(f"{row['Indicator'].lower()} ({direction}, z={row['Rolling z-score']:+.2f})")
    driver_text = ", ".join(drivers) if drivers else "no individual component with sufficient current data"
    forecast_text = "No model forecast is available."
    if forecast is not None and len(forecast.mean):
        projected = float(forecast.mean.iloc[-1] - clean.iloc[-1])
        direction = "higher" if projected > 1 else "lower" if projected < -1 else "broadly unchanged"
        width = float(forecast.upper.iloc[-1] - forecast.lower.iloc[-1])
        forecast_text = f"The {forecast.model} forecast ends {direction} ({projected:+.1f} index points), with a final interval width of {width:.1f} points."
    transition = f"from {previous_regime} to {current_regime}" if previous_regime != current_regime else f"within {current_regime}"
    return (
        f"Liquidity stress {movement} by {delta:+.1f} points over the last {change_window} observations and is currently {transition}. "
        f"The strongest observable components are {driver_text}. {forecast_text} "
        "This is a statistical early-warning assessment, not a prediction of bank failure."
    )
