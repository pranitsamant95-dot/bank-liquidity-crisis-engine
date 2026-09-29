from __future__ import annotations

import numpy as np
import pandas as pd


def detect_mean_shift(series: pd.Series, short_window: int = 4, long_window: int = 26, threshold: float = 1.5) -> dict:
    clean = series.dropna()
    if len(clean) < long_window + short_window:
        return {"detected": False, "score": np.nan, "message": "Insufficient history for change-point screening."}
    recent = clean.iloc[-short_window:].mean()
    baseline = clean.iloc[-(long_window + short_window):-short_window]
    scale = baseline.std(ddof=0)
    score = float((recent - baseline.mean()) / scale) if scale and np.isfinite(scale) else 0.0
    detected = abs(score) >= threshold
    direction = "upward" if score > 0 else "downward"
    message = f"A {direction} mean-shift signal is {'present' if detected else 'not present'} (standardized shift {score:+.2f})."
    return {"detected": detected, "score": score, "message": message}
