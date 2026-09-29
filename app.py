from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.stattools import acf, pacf, grangercausalitytests

from models.anomaly_detection import detect_mean_shift
from models.evaluation import expanding_window_backtest
from models.forecasting import available_models, forecast_model
from models.liquidity_index import COMPONENT_LABELS, construct_liquidity_index, regime_for_value
from utils.data_loader import SERIES, load_financial_data, source_table
from utils.explanations import dynamic_warning_explanation, indicator_diagnostics
from utils.preprocessing import adf_report, validate_time_series

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parent
DISCLAIMER = "This application is an academic analytical system and does not constitute financial, investment, banking, or regulatory advice. Forecasts and stress classifications are model outputs subject to data limitations and statistical uncertainty."
PLOT = {
    "template": "plotly_dark", "paper_bgcolor": "#0b0f14", "plot_bgcolor": "#0e141b",
    "font": {"color": "#c7d0d9", "family": "Inter, Arial, sans-serif", "size": 12},
    "xaxis": {"gridcolor": "#26313c", "zerolinecolor": "#26313c"},
    "yaxis": {"gridcolor": "#26313c", "zerolinecolor": "#26313c"},
    "legend": {"orientation": "h", "y": 1.08, "x": 0}, "margin": {"l": 45, "r": 20, "t": 55, "b": 40},
}
REGIME_COLORS = {"NORMAL": "#42a36f", "WATCH": "#d4a72c", "ELEVATED STRESS": "#d5793b", "SEVERE STRESS": "#d15b5b"}

st.set_page_config(page_title="Bank Liquidity Crisis Prediction Engine", page_icon="▥", layout="wide", initial_sidebar_state="expanded")
css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(ttl=21600, show_spinner="Retrieving official financial time series…")
def get_data(start: str):
    return load_financial_data(start=start, snapshot_path=ROOT / "data" / "fred_snapshot.csv")


@st.cache_data(show_spinner="Constructing the liquidity stress index…")
def get_index(raw: pd.DataFrame):
    return construct_liquidity_index(raw)


@st.cache_data(show_spinner="Running expanding-window model validation…")
def get_backtest(series: pd.Series, models: tuple[str, ...], multivariate: pd.DataFrame):
    return expanding_window_backtest(series, list(models), multivariate, origins=6, block_horizon=4)


@st.cache_data(show_spinner="Fitting the selected forecast model…")
def get_forecast(model: str, series: pd.Series, horizon: int, confidence: float, multivariate: pd.DataFrame):
    return forecast_model(model, series, horizon, confidence, multivariate, optimize=True)


def title(name: str, description: str):
    st.markdown('<div class="terminal-header">SYSTEMIC LIQUIDITY RISK / WEEKLY MONITOR</div>', unsafe_allow_html=True)
    st.title(name)
    st.markdown(f'<div class="subtitle">{description}</div>', unsafe_allow_html=True)


def footer():
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 440, y_title: str | None = None) -> go.Figure:
    fig.update_layout(**PLOT, height=height, hovermode="x unified")
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def stress_chart(series: pd.Series, threshold: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=25, fillcolor="#42a36f", opacity=.07, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor="#d4a72c", opacity=.06, line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor="#d5793b", opacity=.06, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="#d15b5b", opacity=.08, line_width=0)
    fig.add_trace(go.Scatter(x=series.index, y=series, name="Liquidity Stress Index", line={"color": "#d9e1e8", "width": 1.8}))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#d4a72c", annotation_text=f"User warning threshold {threshold:.0f}")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title="Liquidity Stress Index and project-defined regimes")
    return style_figure(fig, y_title="Stress index (0–100)")


def forecast_chart(history: pd.Series, forecast) -> go.Figure:
    hist = history.dropna().tail(156)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Observed", line={"color": "#d9e1e8", "width": 1.7}))
    if forecast is not None:
        x_band = list(forecast.mean.index) + list(forecast.mean.index[::-1])
        y_band = list(forecast.upper.values) + list(forecast.lower.values[::-1])
        fig.add_trace(go.Scatter(x=x_band, y=y_band, fill="toself", fillcolor="rgba(93,137,179,.18)", line={"color": "rgba(0,0,0,0)"}, name="Forecast interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=forecast.mean.index, y=forecast.mean, name=f"{forecast.model} forecast", line={"color": "#5d89b3", "width": 2.2, "dash": "dash"}))
    fig.update_layout(title="Observed stress and near-term forecast")
    fig.update_yaxes(range=[0, 100])
    return style_figure(fig, y_title="Stress index (0–100)")


def previous_distinct_regime(regimes: pd.Series) -> str:
    clean = regimes.dropna().astype(str)
    if clean.empty:
        return "UNAVAILABLE"
    current = clean.iloc[-1]
    prior = clean.iloc[:-1][clean.iloc[:-1] != current]
    return prior.iloc[-1] if not prior.empty else current


def regime_delta(index: pd.Series, periods: int = 4) -> tuple[float, str]:
    clean = index.dropna()
    if len(clean) <= periods:
        return np.nan, "Unavailable"
    delta = float(clean.iloc[-1] - clean.iloc[-1 - periods])
    return delta, "Increasing" if delta > 1 else "Decreasing" if delta < -1 else "Stable"


def fmt(value: float, digits: int = 1) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.{digits}f}"


# Global controls and data pipeline
with st.sidebar:
    st.markdown("### LIQUIDITY RISK ENGINE")
    st.caption("Official-data time-series monitor")
    page = st.radio("Navigation", ["Executive Dashboard", "Liquidity Stress", "Time-Series Analysis", "Forecasting", "Early-Warning Signals", "Model Performance", "Historical Stress Events", "Methodology", "Data Sources"], label_visibility="collapsed")
    st.divider()
    st.markdown("#### ANALYSIS CONTROLS")
    lookback = st.selectbox("Historical lookback", ["3 years", "5 years", "10 years", "All available"], index=1)
    horizon = st.slider("Forecast horizon (weeks)", 1, 26, 8)
    confidence = st.select_slider("Forecast confidence", options=[0.80, 0.90, 0.95, 0.99], value=0.95, format_func=lambda x: f"{x:.0%}")
    warning_threshold = st.slider("Custom warning threshold", 25, 85, 50)

try:
    raw, source_status = get_data("2000-01-01")
    raw, quality = validate_time_series(raw)
    indexed, contributions, meta = get_index(raw)
except Exception as exc:
    st.error("The analytical pipeline could not be initialized.")
    st.exception(exc)
    st.info("Check internet access and confirm that data/fred_snapshot.csv is present. No synthetic values are generated.")
    st.stop()

stress = indexed["Stress Index"].dropna()
if len(stress) < 80:
    st.error("Insufficient common history is available to estimate a reliable stress index and forecast.")
    st.stop()

multivariate = pd.concat([indexed["Stress Index"], meta["zscores"].add_prefix("z_")], axis=1).dropna()
models = available_models(stress, multivariate)
metrics, bt_predictions, bt_info = get_backtest(stress, tuple(models), multivariate)
auto_model = metrics.iloc[0]["Model"] if not metrics.empty else "Naive"
with st.sidebar:
    model_choice = st.selectbox("Forecast model", ["Auto (validation winner)"] + models)
selected_model = auto_model if model_choice.startswith("Auto") else model_choice
try:
    forecast = get_forecast(selected_model, stress, horizon, confidence, multivariate)
    forecast_error = None
except Exception as exc:
    forecast_error = str(exc)
    try:
        forecast = get_forecast("Naive", stress, horizon, confidence, multivariate)
        selected_model = "Naive"
    except Exception:
        forecast = None

end = stress.index.max()
years = {"3 years": 3, "5 years": 5, "10 years": 10}
start_display = stress.index.min() if lookback == "All available" else end - pd.DateOffset(years=years[lookback])
display_stress = stress.loc[start_display:]
latest = float(stress.iloc[-1])
current_regime = regime_for_value(latest)
previous_regime = previous_distinct_regime(indexed["Regime"])
delta4, trend = regime_delta(stress, 4)
acceleration = float(stress.diff().diff().tail(4).mean())
change_point = detect_mean_shift(stress)
diagnostics = indicator_diagnostics(meta["transformed"], meta["zscores"], contributions)
explanation = dynamic_warning_explanation(stress, diagnostics, forecast, previous_regime, current_regime)

with st.sidebar:
    st.divider()
    st.caption(f"Data mode: {source_status['mode']}")
    st.caption(f"Latest weekly observation: {end:%d %b %Y}")
    if source_status["failures"]:
        st.warning(f"{len(source_status['failures'])} live series issue(s); missing-series policy applied.")

if page == "Executive Dashboard":
    title("Bank Liquidity Crisis Prediction Engine", "Early-warning analytics for banking-system liquidity stress; not a bank-failure prediction.")
    color_class = current_regime.lower().replace(" ", "-")
    st.markdown(f'<div class="status-strip">CURRENT CLASSIFICATION &nbsp; <strong class="regime-{color_class}">{current_regime}</strong> &nbsp; | &nbsp; Data through {end:%d %b %Y} &nbsp; | &nbsp; {source_status["mode"]}</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Liquidity Stress", fmt(latest), f"{delta4:+.1f} / 4 weeks")
    c2.metric("Current Regime", current_regime)
    c3.metric("Stress Trend", trend, f"acceleration {acceleration:+.2f}")
    projected = float(forecast.mean.iloc[-1]) if forecast is not None else np.nan
    c4.metric(f"{horizon}-Week Forecast", fmt(projected), f"{projected-latest:+.1f} points" if np.isfinite(projected) else None)
    c5.metric("Model Used", selected_model, "validation-selected" if model_choice.startswith("Auto") else "user-selected")
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(stress_chart(display_stress, warning_threshold), use_container_width=True)
    with right:
        st.plotly_chart(forecast_chart(stress, forecast), use_container_width=True)
    st.subheader("Key Signals")
    show = diagnostics.head(5).copy()
    st.dataframe(show, use_container_width=True, hide_index=True, column_config={"Recent 4-week change": st.column_config.NumberColumn(format="%.3f"), "Historical percentile": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"), "Rolling z-score": st.column_config.NumberColumn(format="%+.2f"), "Index contribution (z units)": st.column_config.NumberColumn(format="%+.3f")})
    st.subheader("Why is the system issuing this warning?")
    st.info(explanation)
    if forecast_error:
        st.warning(f"The requested model failed and the Naive fallback was used: {forecast_error}")

elif page == "Liquidity Stress":
    title("Liquidity Stress", "Transparent composite index, component behavior, and historical regime classification.")
    st.plotly_chart(stress_chart(display_stress, warning_threshold), use_container_width=True)
    tab1, tab2, tab3 = st.tabs(["Contributions", "Regime History", "Index Construction"])
    with tab1:
        latest_contrib = contributions.loc[:end].iloc[-1].dropna().sort_values()
        colors = ["#42a36f" if v < 0 else "#d15b5b" for v in latest_contrib.values]
        fig = go.Figure(go.Bar(x=latest_contrib.values, y=latest_contrib.index, orientation="h", marker_color=colors))
        fig.update_layout(title="Current component contribution; positive values raise measured stress")
        st.plotly_chart(style_figure(fig, 430, "Contribution to composite z-score"), use_container_width=True)
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)
    with tab2:
        regime_frame = indexed.loc[start_display:, ["Stress Index", "Regime"]].dropna()
        counts = regime_frame["Regime"].value_counts().rename_axis("Regime").reset_index(name="Weeks")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.caption("Thresholds are project-defined analytical bands and are not official regulatory thresholds.")
    with tab3:
        st.markdown(f"**Method.** {meta['method']}")
        st.markdown("**Stress direction.** Deposit, bank-securities, and reserve growth enter with a negative sign; falling growth raises stress. Funding rates, credit spreads, VIX, and yield-curve inversion enter positively.")
        st.markdown("**Weights.** Equal weights avoid estimating unstable supervised weights against a non-existent official weekly liquidity-crisis target. Available components are reweighted equally; at least four are required.")
        st.markdown("**Anti-leakage.** Each z-score compares the current observation with a rolling mean and standard deviation shifted by one week, so future data never enters a historical score.")

elif page == "Time-Series Analysis":
    title("Time-Series Analysis", "Exploration, stationarity testing, serial dependence, volatility, and cross-indicator structure.")
    options = {"Liquidity Stress Index": "Stress Index", **{spec.label: key for key, spec in SERIES.items()}}
    label = st.selectbox("Series", list(options))
    key = options[label]
    selected = indexed[key].dropna() if key in indexed else stress
    selected = selected.loc[start_display:]
    if len(selected) < 30:
        st.warning("The selected date range has fewer than 30 observations; broaden the lookback.")
    rolling = pd.DataFrame({"Observed": selected, "13-week mean": selected.rolling(13).mean(), "13-week standard deviation": selected.rolling(13).std()})
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12, subplot_titles=("Level and rolling mean", "Rolling volatility"))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling["Observed"], name="Observed", line={"color": "#d9e1e8"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling["13-week mean"], name="13-week mean", line={"color": "#5d89b3"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling["13-week standard deviation"], name="13-week std", line={"color": "#d4a72c"}), row=2, col=1)
    st.plotly_chart(style_figure(fig, 600), use_container_width=True)
    report = adf_report(selected)
    st.subheader("Augmented Dickey–Fuller Test")
    if report["available"]:
        st.metric("ADF p-value", f"{report['pvalue']:.4f}", f"test statistic {report['statistic']:.3f}")
        st.info(f"ADF p-value = {report['pvalue']:.4f}. {report['interpretation']}")
    else:
        st.warning(report["message"])
    clean = selected.dropna()
    max_lags = min(30, max(1, len(clean) // 2 - 1))
    if len(clean) >= 30:
        acf_values = acf(clean, nlags=max_lags, fft=True)
        pacf_values = pacf(clean, nlags=max_lags, method="ywm")
        lag_frame = pd.DataFrame({"Lag": np.arange(len(acf_values)), "ACF": acf_values, "PACF": pacf_values})
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Autocorrelation function", "Partial autocorrelation function"))
        fig.add_trace(go.Bar(x=lag_frame["Lag"], y=lag_frame["ACF"], name="ACF", marker_color="#5d89b3"), row=1, col=1)
        fig.add_trace(go.Bar(x=lag_frame["Lag"], y=lag_frame["PACF"], name="PACF", marker_color="#d4a72c"), row=1, col=2)
        st.plotly_chart(style_figure(fig, 400), use_container_width=True)
        st.caption("ACF summarizes correlation with prior lags; PACF isolates each lag after controlling for shorter lags. Slow ACF decay can indicate persistence or non-stationarity.")
    st.subheader("Indicator Correlation")
    corr = meta["transformed"].loc[start_display:].corr(min_periods=30)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=[[0, "#3f6c8f"], [.5, "#101820"], [1, "#a85252"]], zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title="Correlation of transformed stress-oriented indicators")
    st.plotly_chart(style_figure(fig, 550), use_container_width=True)

elif page == "Forecasting":
    title("Forecasting", "Near-term stress projections with model-specific uncertainty and observable diagnostics.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Model", selected_model)
    c2.metric("Forecast Endpoint", fmt(float(forecast.mean.iloc[-1])) if forecast else "N/A")
    c3.metric("Confidence Level", f"{confidence:.0%}")
    st.plotly_chart(forecast_chart(stress, forecast), use_container_width=True)
    if forecast:
        table = pd.DataFrame({"Forecast": forecast.mean, "Lower": forecast.lower, "Upper": forecast.upper})
        table.index.name = "Week"
        st.dataframe(table.style.format("{:.2f}"), use_container_width=True)
        st.subheader("Forecast Diagnostics")
        left, right = st.columns(2)
        with left:
            residuals = forecast.residuals.dropna()
            fig = go.Figure(go.Scatter(x=residuals.index, y=residuals, mode="lines", line={"color": "#8fa9bd"}, name="Residual"))
            fig.add_hline(y=0, line_color="#59636d")
            fig.update_layout(title="Model residuals")
            st.plotly_chart(style_figure(fig, 350, "Residual"), use_container_width=True)
        with right:
            st.json(forecast.details, expanded=True)
        st.subheader("Forecast Explanation")
        st.info(explanation)
    else:
        st.error("No forecast is available for the current data.")

elif page == "Early-Warning Signals":
    title("Early-Warning Signals", "Regime transitions, acceleration, user-defined threshold signals, and mean-shift screening.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Regime", current_regime)
    c2.metric("Previous Distinct Regime", previous_regime)
    c3.metric("4-Week Direction", trend, f"{delta4:+.1f} points")
    c4.metric("Change-Point Screen", "DETECTED" if change_point["detected"] else "Not detected", f"z={change_point['score']:+.2f}" if np.isfinite(change_point["score"]) else None)
    st.plotly_chart(stress_chart(display_stress, warning_threshold), use_container_width=True)
    above = display_stress >= warning_threshold
    transitions = above & ~above.shift(1, fill_value=False)
    signals = pd.DataFrame({"Stress Index": display_stress[transitions], "Signal": f"Crossed {warning_threshold}"})
    st.subheader("Historical Warning Signals")
    if signals.empty:
        st.info("No upward crossing of the selected warning threshold occurred in the displayed period.")
    else:
        st.dataframe(signals.sort_index(ascending=False), use_container_width=True)
    st.subheader("Why is the system issuing this warning?")
    st.info(explanation)
    st.caption(change_point["message"] + " This is a simple standardized mean-shift screen, not a formal causal break test.")

elif page == "Model Performance":
    title("Model Performance", "Chronological expanding-window validation; no random shuffling and no future information in training windows.")
    if metrics.empty:
        st.warning("Model comparison is unavailable: insufficient common history or all candidate fits failed.")
    else:
        display_metrics = metrics.copy()
        display_metrics["Selection"] = np.where(display_metrics["Selected"], "SELECTED", "")
        st.dataframe(display_metrics.drop(columns="Selected").style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE (%)": "{:.2f}"}), use_container_width=True, hide_index=True)
        st.success(f"Selection rule: {bt_info['criterion']} Winner: {auto_model}.")
        fig = go.Figure()
        for model, group in bt_predictions.groupby("Model"):
            fig.add_trace(go.Scatter(x=group["Date"], y=group["Forecast"], mode="lines+markers", name=model, opacity=.75))
        actual = bt_predictions.drop_duplicates("Date").sort_values("Date")
        fig.add_trace(go.Scatter(x=actual["Date"], y=actual["Actual"], mode="lines+markers", name="Actual", line={"color": "#ffffff", "width": 2.5}))
        fig.update_layout(title="Expanding-window forecasts versus realized stress")
        st.plotly_chart(style_figure(fig, 480, "Stress index"), use_container_width=True)
    if bt_info.get("failures"):
        with st.expander("Unavailable model diagnostics"):
            st.json(bt_info["failures"])
    st.markdown("Each origin fits only observations available at that date, forecasts the next four weeks, then expands the training window. Metrics aggregate six recent, non-overlapping origins. MAPE is shown because the 0–100 index is positive; RMSE is the selection criterion because larger misses are especially undesirable in risk monitoring.")

elif page == "Historical Stress Events":
    title("Historical Stress Events", "Descriptive backtests around widely recognized market-stress windows; no claim of event prediction.")
    events = {
        "Global Financial Crisis": (pd.Timestamp("2007-07-01"), pd.Timestamp("2009-06-30")),
        "COVID-19 market stress": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30")),
        "2023 US banking turmoil": (pd.Timestamp("2023-03-01"), pd.Timestamp("2023-05-31")),
    }
    event_name = st.selectbox("Historical window", list(events))
    event_start, event_end = events[event_name]
    window = stress.loc[event_start - pd.DateOffset(months=9):event_end + pd.DateOffset(months=6)]
    if window.empty:
        st.warning("The current data coverage does not include this event.")
    else:
        fig = stress_chart(window)
        fig.add_vrect(x0=event_start, x1=event_end, fillcolor="#d15b5b", opacity=.10, line_width=0, annotation_text=event_name)
        st.plotly_chart(fig, use_container_width=True)
        before = stress.loc[event_start - pd.DateOffset(weeks=13):event_start - pd.DateOffset(days=1)]
        during = stress.loc[event_start:event_end]
        crossing = stress.loc[:event_start]
        crossing = crossing[(crossing >= 50) & (crossing.shift(1) < 50)]
        lead = (event_start - crossing.index[-1]).days if not crossing.empty else None
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pre-event Mean", fmt(float(before.mean())))
        c2.metric("Event-window Maximum", fmt(float(during.max())))
        c3.metric("Event-window Change", fmt(float(during.iloc[-1] - during.iloc[0])) if len(during) > 1 else "N/A")
        c4.metric("Most Recent Watch/Elevated Crossing", f"{lead} days before start" if lead is not None and lead >= 0 else "No prior crossing")
        st.info("This panel is descriptive. A threshold crossing is not evidence that the system predicted or caused the event; event dates are used only for retrospective comparison.")

elif page == "Methodology":
    title("Methodology", "Reproducible academic design, assumptions, model selection, and limitations.")
    st.markdown(r"""
### 1. Data collection
Official public series are downloaded from FRED's CSV endpoint and aligned to Friday weeks. A bundled snapshot of the same official observations is used only when live retrieval fails; synthetic data are never silently substituted.

### 2. Preprocessing and quality control
Timestamps are parsed, duplicates removed, values coerced to numeric, and series sorted chronologically. Daily and weekly series are converted to weekly last observations. Carry-forward is limited to four weeks so long outages remain visible. Extreme transformed observations are bounded only at the z-score stage (±4), preserving the raw data.

### 3. Exploratory time-series analysis
The application reports rolling means, rolling standard deviations, percentage changes, transformed-indicator correlations, ACF, PACF, and an Augmented Dickey–Fuller test with a plain-English interpretation.

### 4. Liquidity Stress Index
Seven components are oriented so positive values represent greater stress: deposit-growth deterioration, liquid-asset deterioration, reserve deterioration, funding cost, high-yield credit spread, VIX, and yield-curve inversion. For component $j$:

$$z_{j,t}=\frac{x_{j,t}-\mu_{j,t-1}^{(156)}}{\sigma_{j,t-1}^{(156)}}$$

The rolling moments are shifted one period and use only information available before week $t$. The composite and reported score are:

$$C_t=\frac{1}{N_t}\sum_j z_{j,t}, \qquad LSI_t=100\Phi(C_t)$$

At least four components are required. Equal weights are used because there is no official weekly liquidity-crisis target from which stable supervised weights could be estimated. The 0–25, 25–50, 50–75, and 75–100 bands are project-defined—not regulatory thresholds.

### 5. Forecasting models
The Naive random-walk baseline is compared with ARIMA, damped additive Exponential Smoothing, SARIMA only when 13-week autocorrelation supports seasonality, and VAR only when sufficient complete multivariate history exists. ARIMA order is selected by AIC from a small candidate grid for the final fit. VAR is fit to first differences and integrated back to levels.

### 6. Validation and model selection
Validation uses expanding windows. Each model is trained on the past, forecasts a non-overlapping four-week block, then retrains after the origin advances. The lowest aggregate RMSE wins; MAE breaks ties. No rows are shuffled. Historical index features use shifted rolling statistics to avoid look-ahead leakage.

### 7. Early warning and interpretation
Current regimes follow the project bands. Trend is the four-week score change; acceleration is the recent mean second difference. A transparent mean-shift screen compares the recent four-week mean with the preceding 26 weeks. Driver text is generated from actual z-scores, contributions, percentiles, and model intervals.

### 8. Granger causality
Granger tests, when run below, test whether lagged values improve prediction; they do not establish economic causality. Results should be interpreted with stationarity and multiple-testing limitations in mind.

### 9. Limitations
FRED revisions, publication lags, changing bank definitions, aggregation across institutions, structural breaks, and model misspecification affect results. The composite is not calibrated to supervisory loss data. Confidence intervals are conditional on model assumptions and can understate tail risk.
""")
    with st.expander("Optional Granger-causality screen"):
        candidate = st.selectbox("Stress-oriented component", list(meta["zscores"].columns), format_func=lambda x: COMPONENT_LABELS[x])
        pair = pd.concat([stress.rename("stress"), meta["zscores"][candidate].rename("component")], axis=1).dropna().diff().dropna().tail(260)
        if len(pair) >= 60:
            try:
                result = grangercausalitytests(pair[["stress", "component"]], maxlag=4, verbose=False)
                rows = [{"Lag": lag, "F-test p-value": values[0]["ssr_ftest"][1]} for lag, values in result.items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption("Null hypothesis: lagged component values do not add predictive information for changes in the stress index. Rejection is predictive association, not causation.")
            except Exception as exc:
                st.warning(f"The test could not be estimated: {exc}")
        else:
            st.warning("At least 60 complete differenced observations are required.")

elif page == "Data Sources":
    title("Data Sources", "Official series inventory, live-retrieval status, fallback policy, and data-quality diagnostics.")
    st.dataframe(source_table(), use_container_width=True, hide_index=True)
    st.markdown("All series are retrieved from the Federal Reserve Bank of St. Louis FRED CSV service. H.8 series originate with the Federal Reserve Board. ICE BofA spread and CBOE VIX observations are redistributed through FRED subject to their source notes.")
    st.subheader("Retrieval Status")
    status_display = {"Mode": source_status["mode"], "Latest observation": str(source_status["last_observation"]), "Snapshot available": bool(source_status["snapshot_path"]), "Live failures": len(source_status["failures"])}
    st.json(status_display)
    if source_status["failures"]:
        st.warning("One or more live downloads failed. The official snapshot was used where available; affected series are listed below.")
        st.json(source_status["failures"])
    st.subheader("Data Quality")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Weekly Rows", quality["Rows"])
    q2.metric("Duplicate Timestamps", quality["Duplicate timestamps removed"])
    q3.metric("Numeric Coverage", f"{quality['Numeric coverage']:.1%}")
    q4.metric("Inferred Frequency", quality["Inferred frequency"])
    missing = pd.DataFrame({"Series": list(quality["Missing values"]), "Missing observations": list(quality["Missing values"].values())})
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.markdown("**API configuration:** No API key is required for the FRED graph CSV endpoint used by this project. Consequently, no secret is required on Streamlit Community Cloud. The code never hardcodes credentials.")

footer()
