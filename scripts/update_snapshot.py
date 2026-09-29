from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_financial_data  # noqa: E402


def main() -> None:
    destination = ROOT / "data" / "fred_snapshot.csv"
    temporary_missing = ROOT / "data" / "__no_existing_snapshot__.csv"
    frame, status = load_financial_data(start="2000-01-01", snapshot_path=temporary_missing)
    if status["failures"]:
        failed = ", ".join(status["failures"])
        raise RuntimeError(f"Refusing to create an incomplete snapshot. Failed live series: {failed}")
    frame.index.name = "date"
    frame.to_csv(destination, float_format="%.6f")
    print(f"Wrote {len(frame):,} official weekly rows to {destination}")
    print(f"Coverage: {frame.index.min().date()} to {frame.index.max().date()}")


if __name__ == "__main__":
    main()
