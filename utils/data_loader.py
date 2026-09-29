from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    label: str
    unit: str
    source: str
    frequency: str


SERIES: Dict[str, SeriesSpec] = {
    "deposits": SeriesSpec("DPSACBW027SBOG", "Commercial bank deposits", "Billions of USD", "Federal Reserve H.8 via FRED", "Weekly"),
    "loans": SeriesSpec("TOTLL", "Bank loans and leases", "Billions of USD", "Federal Reserve H.8 via FRED", "Weekly"),
    "securities": SeriesSpec("USGSEC", "Treasury and agency securities at banks", "Billions of USD", "Federal Reserve H.8 via FRED", "Weekly"),
    "reserves": SeriesSpec("WRESBAL", "Reserve balances with Federal Reserve Banks", "Billions of USD", "Federal Reserve via FRED", "Weekly"),
    "fed_funds": SeriesSpec("DFF", "Effective federal funds rate", "Percent", "Federal Reserve Bank of New York via FRED", "Daily"),
    "credit_spread": SeriesSpec("BAMLH0A0HYM2", "US high-yield option-adjusted spread", "Percentage points", "ICE BofA via FRED", "Daily"),
    "vix": SeriesSpec("VIXCLS", "CBOE volatility index", "Index", "CBOE via FRED", "Daily"),
    "treasury_10y": SeriesSpec("DGS10", "10-year Treasury yield", "Percent", "Federal Reserve H.15 via FRED", "Daily"),
    "treasury_3m": SeriesSpec("DGS3MO", "3-month Treasury yield", "Percent", "Federal Reserve H.15 via FRED", "Daily"),
}


def _session() -> requests.Session:
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "bank-liquidity-academic-engine/1.0"})
    return session


def _download_fred(series_id: str, start: str, session: requests.Session) -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + series_id + "&cosd=" + start
    response = session.get(url, timeout=25)
    response.raise_for_status()
    from io import StringIO

    frame = pd.read_csv(StringIO(response.text))
    date_col = frame.columns[0]
    value_col = frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col].replace(".", pd.NA), errors="coerce")
    series = frame.dropna(subset=[date_col]).set_index(date_col)[value_col].sort_index()
    series.name = series_id
    if series.dropna().empty:
        raise ValueError(f"FRED returned no numeric observations for {series_id}.")
    return series


def _to_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    weekly = frame.resample("W-FRI").last()
    # Limited carry-forward bridges holidays and short publication gaps without hiding long outages.
    weekly = weekly.ffill(limit=4)
    return weekly


def load_financial_data(start: str = "2000-01-01", snapshot_path: str | Path | None = None) -> Tuple[pd.DataFrame, dict]:
    """Load official FRED series. If live retrieval is incomplete, use the bundled official snapshot.

    Returns (weekly_frame, status). The status object identifies live/snapshot coverage and failures.
    """
    session = _session()
    downloaded: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}
    for key, spec in SERIES.items():
        try:
            downloaded[key] = _download_fred(spec.series_id, start, session)
        except Exception as exc:  # network and provider errors are reported to the UI
            failures[key] = f"{type(exc).__name__}: {exc}"

    live = _to_weekly(pd.concat(downloaded, axis=1)) if downloaded else pd.DataFrame()
    snapshot = Path(snapshot_path) if snapshot_path else Path(__file__).resolve().parents[1] / "data" / "fred_snapshot.csv"
    snapshot_frame = pd.DataFrame()
    if snapshot.exists():
        snapshot_frame = pd.read_csv(snapshot, parse_dates=["date"]).set_index("date").sort_index()
        snapshot_frame = _to_weekly(snapshot_frame)

    if live.empty and snapshot_frame.empty:
        raise RuntimeError("Live FRED retrieval failed and no bundled official-data snapshot is available.")

    if live.empty:
        combined = snapshot_frame
        mode = "official snapshot"
    else:
        combined = live.combine_first(snapshot_frame)
        missing_live = set(SERIES).difference(combined.columns)
        if missing_live:
            failures.update({key: "Series unavailable in live and snapshot data" for key in missing_live})
        mode = "live FRED" if not failures else ("live FRED with official snapshot fallback" if not snapshot_frame.empty else "partial live FRED coverage")

    combined = combined.reindex(columns=list(SERIES)).apply(pd.to_numeric, errors="coerce")
    combined = combined.loc[pd.Timestamp(start):]
    coverage = {col: int(combined[col].notna().sum()) for col in combined.columns}
    status = {
        "mode": mode,
        "failures": failures,
        "coverage": coverage,
        "last_observation": combined.dropna(how="all").index.max(),
        "snapshot_path": str(snapshot) if snapshot.exists() else None,
    }
    return combined, status


def source_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Key": key, "FRED series": spec.series_id, "Indicator": spec.label, "Unit": spec.unit, "Frequency": spec.frequency, "Source": spec.source}
         for key, spec in SERIES.items()]
    )
