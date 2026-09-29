# Data directory

The application downloads official FRED observations at runtime. No fabricated fallback data are stored here.

To create an optional official-data resilience snapshot while connected to FRED, run:

```bash
python scripts/update_snapshot.py
```

This writes `data/fred_snapshot.csv`. Commit that file if the deployment must continue operating during a complete provider outage. The UI clearly reports when snapshot values are used.
