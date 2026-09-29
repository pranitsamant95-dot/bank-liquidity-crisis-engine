# Bank Liquidity Crisis Prediction Engine

A production-oriented academic Streamlit application for monitoring and forecasting US banking-system liquidity stress from official public time series. The system estimates a transparent **Liquidity Stress Index (LSI)**, classifies the current analytical regime, compares genuine time-series models with chronological validation, and explains warnings using calculated indicator contributions.

> This project does not predict bank failure. It identifies temporal patterns associated with increasing system-level liquidity stress and forecasts the near-term stress index.

## Features

- Automatic retrieval of nine Federal Reserve/FRED series; no upload required
- Transparent 0–100 Liquidity Stress Index with seven past-only rolling z-score components
- NORMAL, WATCH, ELEVATED STRESS, and SEVERE STRESS project regimes
- Interactive historical timeline, contribution analysis, and forecast intervals
- ADF stationarity test with plain-English interpretation
- Rolling mean, rolling volatility, ACF, PACF, transformed-indicator correlation, and optional Granger tests
- Naive, ARIMA, conditional SARIMA, damped Exponential Smoothing, and conditional VAR models
- Expanding-window, non-shuffled model validation with MAE, RMSE, and MAPE
- Historical analysis for the Global Financial Crisis, COVID-19 market stress, and 2023 US banking turmoil
- Dynamic “Why is the system issuing this warning?” narrative derived from actual z-scores, regime changes, and forecasts
- Graceful handling of provider failures, missing series, insufficient samples, and model non-convergence
- Institutional dark UI designed for information density rather than decoration

## Application Pages

1. Executive Dashboard
2. Liquidity Stress
3. Time-Series Analysis
4. Forecasting
5. Early-Warning Signals
6. Model Performance
7. Historical Stress Events
8. Methodology
9. Data Sources

## Architecture

```text
bank-liquidity-crisis-engine/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   └── style.css
├── data/
│   ├── .gitkeep
│   └── README.md
├── models/
│   ├── __init__.py
│   ├── anomaly_detection.py
│   ├── evaluation.py
│   ├── forecasting.py
│   └── liquidity_index.py
├── scripts/
│   └── update_snapshot.py
└── utils/
    ├── __init__.py
    ├── data_loader.py
    ├── explanations.py
    └── preprocessing.py
```

## Data Sources

The application downloads CSV data from the Federal Reserve Bank of St. Louis FRED graph endpoint. No API key is required.

| FRED ID | Description | Original source |
|---|---|---|
| `DPSACBW027SBOG` | Deposits, all commercial banks | Federal Reserve H.8 |
| `TOTLL` | Loans and leases in bank credit | Federal Reserve H.8 |
| `USGSEC` | Treasury and agency securities at banks | Federal Reserve H.8 |
| `WRESBAL` | Reserve balances with Federal Reserve Banks | Federal Reserve |
| `DFF` | Effective federal funds rate | Federal Reserve Bank of New York |
| `BAMLH0A0HYM2` | ICE BofA US high-yield option-adjusted spread | ICE BofA via FRED |
| `VIXCLS` | CBOE volatility index | CBOE via FRED |
| `DGS10` | 10-year Treasury yield | Federal Reserve H.15 |
| `DGS3MO` | 3-month Treasury yield | Federal Reserve H.15 |

Daily and weekly observations are aligned to Friday weeks. Carry-forward is capped at four weeks. The application does not silently create synthetic financial observations.

### Optional official-data snapshot

For environments that must survive a total FRED outage, run `python scripts/update_snapshot.py` while FRED is reachable and commit the generated `data/fred_snapshot.csv`. The loader combines live observations with that official snapshot and reports fallback use in the UI. The repository remains deployable without a snapshot; if all live downloads fail and no snapshot exists, the app displays a clean diagnostic rather than fabricated results.

## Liquidity Stress Index

Seven stress-oriented components are constructed:

1. Deposit-growth deterioration
2. Bank liquid-asset (Treasury and agency securities) deterioration
3. Reserve deterioration
4. Effective funding cost
5. High-yield credit spread
6. Market volatility
7. Yield-curve inversion

Bank balance-sheet growth rates use annualized 13-week changes. Each component is standardized against a 156-week rolling window whose mean and standard deviation are shifted by one week:

```text
z(j,t) = [x(j,t) - mean(j,t-1)] / std(j,t-1)
Composite(t) = mean of available z(j,t), requiring at least four components
LSI(t) = 100 × standard_normal_CDF(Composite(t))
```

Z-scores are bounded at ±4 to limit the influence of isolated data errors while retaining extreme stress. Equal weighting is intentional: there is no official weekly crisis target suitable for estimating stable supervised weights, and equal weights reduce overfitting. The bands 0–25, 25–50, 50–75, and 75–100 are **project-defined analytical thresholds, not official regulatory thresholds**.

## Forecasting and Evaluation

- **Naive:** random-walk baseline
- **ARIMA:** final order selected by AIC from a compact candidate grid
- **SARIMA:** enabled only when 13-week autocorrelation provides evidence of seasonality
- **Exponential Smoothing:** damped additive trend
- **VAR:** enabled only with sufficient complete multivariate history; fitted to first differences and integrated to levels

Validation uses six recent expanding-window origins. At each origin, only past observations are fitted; the next non-overlapping four-week block is forecast. Models are ranked by aggregate RMSE, with MAE as a tie-breaker. Rows are never randomly shuffled. Historical index construction is past-only, avoiding look-ahead leakage.

## Local Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone <your-repository-url>
cd bank-liquidity-crisis-engine
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

## Streamlit Community Cloud Deployment

1. Create a GitHub repository and copy every project file into it.
2. Push the repository to GitHub.
3. Sign in at [share.streamlit.io](https://share.streamlit.io/).
4. Select **Create app** and choose the repository, branch, and `app.py` as the entry point.
5. Choose Python 3.11 if the deployment UI offers a runtime selection.
6. Deploy. Streamlit installs `requirements.txt` automatically.

No secret is required because the FRED graph CSV endpoint does not require an API key. Do not create or hardcode a FRED key for this implementation. If organization-level network policy blocks FRED, generate and commit the optional official snapshot before deployment.

## Caching and Failure Handling

- Source downloads are cached for six hours.
- Index construction, validation, and forecasts are cached by their inputs.
- HTTP retries cover transient 429 and 5xx responses.
- Short publication gaps are forward-filled for at most four weeks.
- Failed models are excluded from comparison; final forecast failure falls back to the Naive model and is disclosed.
- Insufficient data, invalid model conditions, and total provider outages produce clear UI messages rather than false estimates.

## Interpretation and Academic Integrity

The warning narrative is deterministic and data-driven. It uses observed regime movement, recent score change, component z-scores, component contributions, historical percentiles, model endpoint, and interval width. It does not expose or claim hidden reasoning. Granger results represent incremental predictive association and must not be read as economic causality.

Historical event pages are descriptive backtests. They do not claim that the system predicted the event unless a prior threshold crossing is actually present, and even a crossing is not proof of successful event prediction.

## Limitations

- The data describe the aggregate US banking system, not an individual institution.
- FRED observations can be revised and have publication lags.
- Weekly alignment can conceal intraweek stress.
- The LSI is not calibrated against confidential supervisory or bank-failure data.
- Equal weights are transparent but may not be optimal in every regime.
- Structural breaks can reduce forecast reliability.
- Normal-theory and model-based intervals can understate tail risk.
- MAPE is secondary because index values near zero can destabilize percentage errors; RMSE is the selection metric.

## Disclaimer

This application is an academic analytical system and does not constitute financial, investment, banking, or regulatory advice. Forecasts and stress classifications are model outputs subject to data limitations and statistical uncertainty.
