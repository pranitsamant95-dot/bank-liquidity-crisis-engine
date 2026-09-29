from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from io import StringIO
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

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MAX_WORKERS = 5
REQUEST_TIMEOUT = (3.05, 7.0)
MAX_LIVE_SECONDS = 11.0


def _session() -> requests.Session:
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS))
    session.headers.update({"User-Agent": "bank-liquidity-academic-engine/1.1"})
    return session


def _download_fred(spec: SeriesSpec, start: str) -> pd.Series:
    url = f"{FRED_URL}?id={spec.series_id}&cosd={start}"
    session = _session()
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
    finally:
        session.close()

    if frame.shape[1] < 2:
        raise ValueError(f"FRED returned an unexpected response for {spec.series_id}.")
    date_col, value_col = frame.columns[0], frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col].replace(".", pd.NA), errors="coerce")
    series = frame.dropna(subset=[date_col]).set_index(date_col)[value_col].sort_index()
    series.name = spec.series_id
    if series.dropna().empty:
        raise ValueError(f"FRED returned no numeric observations for {spec.series_id}.")
    return series


def _to_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    weekly = frame.resample("W-FRI").last()
    # Limited carry-forward bridges holidays and short publication gaps without hiding long outages.
    return weekly.ffill(limit=4)


def _read_snapshot(snapshot: Path) -> pd.DataFrame:
    if not snapshot.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(snapshot, parse_dates=["date"]).set_index("date").sort_index()
    except Exception:
        return pd.DataFrame()
    return _to_weekly(frame)


def _snapshot_is_sufficient(frame: pd.DataFrame, start: str) -> bool:
    if frame.empty:
        return False
    required = [c for c in SERIES if c in frame.columns]
    if len(required) != len(SERIES):
        return False
    usable = frame.loc[pd.Timestamp(start):, required].dropna(how="all")
    return len(usable) >= 80


def _download_concurrently(start: str) -> tuple[dict[str, pd.Series], dict[str, str]]:
    downloaded: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="fred")
    futures = {executor.submit(_download_fred, spec, start): key for key, spec in SERIES.items()}
    try:
        for future in as_completed(futures, timeout=MAX_LIVE_SECONDS):
            key = futures[future]
            try:
                downloaded[key] = future.result()
            except Exception as exc:
                failures[key] = f"{type(exc).__name__}: {exc}"
    except FuturesTimeoutError:
        for future, key in futures.items():
            if not future.done():
                future.cancel()
                failures[key] = "Live FRED request timed out; snapshot fallback applied where available."
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return downloaded, failures


def load_financial_data(
    start: str = "2000-01-01",
    snapshot_path: str | Path | None = None,
    refresh_live: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Load official FRED series with a fast official-snapshot path and bounded concurrent refresh.

    The bundled snapshot is preferred when it contains all required series and enough history.
    Live FRED retrieval is used when explicitly requested or when no sufficient snapshot exists.
    No synthetic observations are created.
    """
    snapshot = Path(snapshot_path) if snapshot_path else Path(__file__).resolve().parents[1] / "data" / "fred_snapshot.csv"
    snapshot_frame = _read_snapshot(snapshot)
    snapshot_ready = _snapshot_is_sufficient(snapshot_frame, start)

    if snapshot_ready and not refresh_live:
        combined = snapshot_frame.reindex(columns=list(SERIES)).apply(pd.to_numeric, errors="coerce")
        combined = combined.loc[pd.Timestamp(start):]
        return combined, {
            "mode": "official snapshot",
            "failures": {},
            "coverage": {col: int(combined[col].notna().sum()) for col in combined.columns},
            "last_observation": combined.dropna(how="all").index.max(),
            "snapshot_path": str(snapshot),
        }

    downloaded, failures = _download_concurrently(start)
    live = _to_weekly(pd.concat(downloaded, axis=1)) if downloaded else pd.DataFrame()

    if live.empty and snapshot_frame.empty:
        raise RuntimeError("Live FRED retrieval failed and no bundled official-data snapshot is available.")

    if live.empty:
        combined = snapshot_frame
        mode = "official snapshot"
    elif snapshot_frame.empty:
        combined = live
        mode = "live FRED" if not failures else "live FRED: partial"
    else:
        # Live observations take precedence; snapshot fills missing columns/observations.
        combined = live.combine_first(snapshot_frame)
        mode = "live FRED" if not failures else "live FRED: partial"

    combined = combined.reindex(columns=list(SERIES)).apply(pd.to_numeric, errors="coerce")
    combined = combined.loc[pd.Timestamp(start):]
    for key in SERIES:
        if combined[key].notna().sum() == 0:
            failures.setdefault(key, "No usable observation in live data or official snapshot.")

    status = {
        "mode": mode,
        "failures": failures,
        "coverage": {col: int(combined[col].notna().sum()) for col in combined.columns},
        "last_observation": combined.dropna(how="all").index.max() if not combined.empty else None,
        "snapshot_path": str(snapshot) if snapshot.exists() else None,
    }
    if combined.dropna(how="all").empty:
        raise RuntimeError("No usable official observations were available from FRED or the bundled snapshot.")
    return combined, status


def source_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Key": key, "FRED series": spec.series_id, "Indicator": spec.label, "Unit": spec.unit, "Frequency": spec.frequency, "Source": spec.source}
         for key, spec in SERIES.items()]
    )
