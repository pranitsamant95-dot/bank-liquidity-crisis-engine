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
    "template": "plotly_dark",
    "paper_bgcolor": "#0b1016",
    "plot_bgcolor": "#0f151c",
    "font": {"color": "#cbd3db", "family": "Inter, Arial, sans-serif", "size": 12},
    "xaxis": {"gridcolor": "#202b35", "zerolinecolor": "#202b35", "showline": True, "linecolor": "#2b3742"},
    "yaxis": {"gridcolor": "#202b35", "zerolinecolor": "#202b35", "showline": True, "linecolor": "#2b3742"},
    "legend": {"orientation": "h", "y": 1.04, "x": 0},
    "margin": {"l": 48, "r": 22, "t": 55, "b": 42},
}
REGIME_COLORS = {"NORMAL": "#5e9d7a", "WATCH": "#c39b45", "ELEVATED STRESS": "#bd7950", "SEVERE STRESS": "#ad5e62"}
PAGES = [
    "OVERVIEW", "LIQUIDITY", "TIME SERIES", "FORECAST", "EARLY WARNING",
    "MODEL VALIDATION", "EVENTS", "METHODOLOGY", "DATA",
]
PAGE_MAP = {
    "OVERVIEW": "Executive Dashboard",
    "LIQUIDITY": "Liquidity Stress",
    "TIME SERIES": "Time-Series Analysis",
    "FORECAST": "Forecasting",
    "EARLY WARNING": "Early-Warning Signals",
    "MODEL VALIDATION": "Model Performance",
    "EVENTS": "Historical Stress Events",
    "METHODOLOGY": "Methodology",
    "DATA": "Data Sources",
}

st.set_page_config(page_title="Bank Liquidity Crisis Prediction Engine", page_icon="▥", layout="wide", initial_sidebar_state="collapsed")
css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(ttl=21600, show_spinner=False)
def get_data(start: str, refresh_live: bool = False):
    return load_financial_data(start=start, snapshot_path=ROOT / "data" / "fred_snapshot.csv", refresh_live=refresh_live)


@st.cache_data(show_spinner=False)
def get_index(raw: pd.DataFrame):
    return construct_liquidity_index(raw)


@st.cache_data(show_spinner=False)
def run_validation(series: pd.Series, models: tuple[str, ...], multivariate: pd.DataFrame):
    return expanding_window_backtest(series, list(models), multivariate, origins=4, block_horizon=4)


@st.cache_data(show_spinner=False)
def forecast_for_page(model: str, series: pd.Series, horizon: int, confidence: float, multivariate: pd.DataFrame, optimize: bool = True):
    return forecast_model(model, series, horizon, confidence, multivariate, optimize=optimize)


@st.cache_data(show_spinner=False)
def get_diagnostics(transformed: pd.DataFrame, zscores: pd.DataFrame, contributions: pd.DataFrame):
    return indicator_diagnostics(transformed, zscores, contributions)


def style_figure(fig: go.Figure, height: int = 440, y_title: str | None = None) -> go.Figure:
    fig.update_layout(**PLOT, height=height, hovermode="x unified")
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def fmt(value: float, digits: int = 1) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.{digits}f}"


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


def stress_chart(series: pd.Series, threshold: float | None = None, title: str = "Liquidity Stress Timeline") -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=25, fillcolor="#5e9d7a", opacity=.035, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor="#c39b45", opacity=.035, line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor="#bd7950", opacity=.035, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="#ad5e62", opacity=.045, line_width=0)
    fig.add_trace(go.Scatter(x=series.index, y=series, name="Liquidity Stress Index", line={"color": "#dce3e9", "width": 2.0}))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dot", line_color="#c39b45", annotation_text=f"Threshold {threshold:.0f}")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title=title)
    return style_figure(fig, 500, "Stress index (0–100)")


def forecast_chart(history: pd.Series, result) -> go.Figure:
    hist = history.dropna().tail(156)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Observed", line={"color": "#dce3e9", "width": 1.8}))
    if result is not None:
        x_band = list(result.mean.index) + list(result.mean.index[::-1])
        y_band = list(result.upper.values) + list(result.lower.values[::-1])
        fig.add_trace(go.Scatter(x=x_band, y=y_band, fill="toself", fillcolor="rgba(76,120,160,.16)", line={"color": "rgba(0,0,0,0)"}, name="Confidence interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=result.mean.index, y=result.mean, name=f"{result.model} forecast", line={"color": "#6d9bc4", "width": 2.2, "dash": "dash"}))
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title="Observed stress and near-term projection")
    return style_figure(fig, 500, "Stress index (0–100)")


def render_nav() -> str:
    if "page" not in st.session_state or st.session_state.page not in PAGES:
        st.session_state.page = "OVERVIEW"
    st.markdown('<div class="terminal-bar">', unsafe_allow_html=True)
    top = st.columns([2.7, 1.25, 1.1, 1.1, 1.1, 1.1])
    with top[0]:
        st.markdown('<div class="brand">BANK LIQUIDITY RISK ENGINE <span>WEEKLY SYSTEM MONITOR</span></div>', unsafe_allow_html=True)
    with top[1]:
        st.markdown('<div class="system-label">SYSTEM<br><b>ANALYTICAL</b></div>', unsafe_allow_html=True)
    with top[2]:
        if st.button("REFRESH DATA", use_container_width=True, key="refresh_data"):
            st.session_state.refresh_requested = True
            get_data.clear()
            st.rerun()
    with top[3]:
        st.markdown('<div class="terminal-clock">RISK<br><b>WEEKLY</b></div>', unsafe_allow_html=True)
    with top[4]:
        st.markdown('<div class="terminal-clock">UTC<br><b>DATA FEED</b></div>', unsafe_allow_html=True)
    with top[5]:
        st.markdown('<div class="terminal-clock">MODE<br><b>OFFICIAL</b></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    nav_cols = st.columns(len(PAGES))
    for col, label in zip(nav_cols, PAGES):
        with col:
            active = st.session_state.page == label
            if st.button(label, key=f"nav_{label}", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.page = label
                st.rerun()
    st.markdown('<div class="nav-rule"></div>', unsafe_allow_html=True)
    return st.session_state.page


def render_controls() -> tuple[str, int, float, int]:
    c1, c2, c3, c4 = st.columns([1.1, 1.0, 1.0, 1.0])
    with c1:
        lookback = st.selectbox("LOOKBACK", ["3 years", "5 years", "10 years", "All available"], index=1, key="lookback")
    with c2:
        horizon = st.selectbox("HORIZON", [4, 8, 12, 16, 26], index=1, format_func=lambda x: f"{x} WEEKS", key="horizon")
    with c3:
        confidence = st.selectbox("INTERVAL", [0.80, 0.90, 0.95, 0.99], index=2, format_func=lambda x: f"{x:.0%}", key="confidence")
    with c4:
        warning_threshold = st.slider("WATCH THRESHOLD", 25, 85, 50, key="warning_threshold")
    return lookback, horizon, confidence, warning_threshold


def render_header(name: str, description: str):
    st.markdown(f'<div class="section-kicker">RISK TERMINAL / {name.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-description">{description}</div>', unsafe_allow_html=True)


def render_system_status(source_status: dict, end: pd.Timestamp):
    mode = str(source_status.get("mode", "unknown")).upper()
    live_class = "status-live" if mode == "LIVE FRED" else "status-snapshot"
    failure_text = f" / {len(source_status.get('failures', {}))} SERIES ISSUE(S)" if source_status.get("failures") else ""
    st.markdown(
        f'<div class="system-strip"><span class="dot"></span><b>{mode}</b><span>DATA THROUGH {end:%d %b %Y}</span><span>LAST OBSERVATION {end:%H:%M} UTC</span><span class="{live_class}">{"PARTIAL" if source_status.get("failures") else "COMPLETE"}{failure_text}</span></div>',
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


# Top navigation happens before any expensive model validation or forecasting.
page = render_nav()
lookback, horizon, confidence, warning_threshold = render_controls()
refresh_requested = bool(st.session_state.pop("refresh_requested", False))

with st.status("Loading official financial data…", expanded=False) as data_status:
    try:
        raw, source_status = get_data("2000-01-01", refresh_live=refresh_requested)
        raw, quality = validate_time_series(raw)
        data_status.update(label="Official financial data ready", state="complete")
    except Exception as exc:
        data_status.update(label="Official data unavailable", state="error")
        st.error("The analytical data pipeline could not be initialized.")
        st.info("Confirm that FRED is reachable or that data/fred_snapshot.csv is present. No synthetic values are generated.")
        st.exception(exc)
        st.stop()

with st.status("Constructing liquidity stress index…", expanded=False) as index_status:
    try:
        indexed, contributions, meta = get_index(raw)
        stress = indexed["Stress Index"].dropna()
        if len(stress) < 80:
            raise ValueError("Insufficient common history is available to estimate a reliable stress index and forecast.")
        index_status.update(label="Liquidity stress index ready", state="complete")
    except Exception as exc:
        index_status.update(label="Stress index unavailable", state="error")
        st.error(str(exc))
        st.stop()

# Cheap shared state only. Expensive validation and non-baseline forecasting are page-gated below.
multivariate = pd.concat([indexed["Stress Index"], meta["zscores"].add_prefix("z_")], axis=1).dropna()
models = available_models(stress, multivariate)
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

def get_page_forecast(model: str, optimize: bool = True):
    try:
        return forecast_for_page(model, stress, horizon, confidence, multivariate, optimize=optimize), None
    except Exception as exc:
        if model != "Naive":
            try:
                fallback = forecast_for_page("Naive", stress, horizon, confidence, multivariate, optimize=False)
                return fallback, str(exc)
            except Exception as fallback_exc:
                return None, f"{exc}; Naive fallback failed: {fallback_exc}"
        return None, str(exc)


def signal_matrix(diagnostics: pd.DataFrame) -> pd.DataFrame:
    out = diagnostics.copy().head(8)
    if out.empty:
        return out
    def status(row):
        z = row.get("Rolling z-score", np.nan)
        if not np.isfinite(z):
            return "UNAVAILABLE"
        if z >= 1.5:
            return "ELEVATED"
        if z >= 0.75:
            return "WATCH"
        return "NORMAL"
    out["STATUS"] = out.apply(status, axis=1)
    out["CHANGE"] = np.where(out["Recent 4-week change"] > 0.01, "↑", np.where(out["Recent 4-week change"] < -0.01, "↓", "→"))
    return out[["Indicator", "STATUS", "CHANGE", "Rolling z-score", "Index contribution (z units)"]]


if page == "OVERVIEW":
    render_header("Executive Dashboard", "System-level liquidity stress monitoring, near-term projection, and observable drivers.")
    render_system_status(source_status, end)
    diagnostics = get_diagnostics(meta["transformed"], meta["zscores"], contributions)
    baseline_forecast, forecast_error = get_page_forecast("Naive", optimize=False)
    explanation = dynamic_warning_explanation(stress, diagnostics, baseline_forecast, previous_regime, current_regime)
    projected = float(baseline_forecast.mean.iloc[-1]) if baseline_forecast is not None else np.nan
    regime_class = current_regime.lower().replace(" ", "-")

    st.markdown(
        f'<div class="risk-ribbon"><div><span class="eyebrow">CURRENT LIQUIDITY STRESS</span><strong>{latest:.1f}</strong></div><div><span class="eyebrow">REGIME</span><b class="regime-{regime_class}">{current_regime}</b></div><div><span class="eyebrow">4-WEEK DIRECTION</span><b>{delta4:+.1f} / {trend.upper()}</b></div><div><span class="eyebrow">{horizon}-WEEK BASELINE</span><b>{fmt(projected)}</b></div><div><span class="eyebrow">MODEL</span><b>NAIVE BASELINE</b></div></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(stress_chart(display_stress, warning_threshold), use_container_width=True, config={"displayModeBar": False})
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown('<div class="panel-heading">SIGNAL MATRIX</div>', unsafe_allow_html=True)
        st.dataframe(signal_matrix(diagnostics), use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="panel-heading">SYSTEM INTERPRETATION</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="terminal-note">{explanation}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="micro-grid"><span>ACCELERATION</span><b>{acceleration:+.2f}</b><span>CHANGE-POINT</span><b>{"DETECTED" if change_point["detected"] else "NOT DETECTED"}</b><span>DATA MODE</span><b>{source_status["mode"].upper()}</b></div>', unsafe_allow_html=True)
    if source_status["failures"]:
        st.caption("LIVE DATA: PARTIAL. Affected series are listed in DATA. The application does not fabricate replacements.")
    footer()

elif page == "LIQUIDITY":
    render_header("Liquidity Stress", "Composite stress index, component contributions, regimes, and index construction.")
    render_system_status(source_status, end)
    st.plotly_chart(stress_chart(display_stress, warning_threshold), use_container_width=True, config={"displayModeBar": False})
    tab1, tab2, tab3 = st.tabs(["CONTRIBUTIONS", "REGIME HISTORY", "INDEX CONSTRUCTION"])
    diagnostics = get_diagnostics(meta["transformed"], meta["zscores"], contributions)
    with tab1:
        latest_contrib = contributions.loc[:end].iloc[-1].dropna().sort_values()
        fig = go.Figure(go.Bar(x=latest_contrib.values, y=latest_contrib.index, orientation="h", marker_color=["#5e9d7a" if v < 0 else "#ad5e62" for v in latest_contrib.values]))
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
    footer()

elif page == "TIME SERIES":
    render_header("Time-Series Analysis", "Exploration, stationarity testing, serial dependence, volatility, and cross-indicator structure.")
    options = {"Liquidity Stress Index": "Stress Index", **{spec.label: key for key, spec in SERIES.items()}}
    label = st.selectbox("SERIES", list(options))
    key = options[label]
    selected = indexed[key].dropna() if key in indexed else stress
    selected = selected.loc[start_display:]
    if len(selected) < 30:
        st.warning("The selected date range has fewer than 30 observations; broaden the lookback.")
    rolling = pd.DataFrame({"Observed": selected, "13-week mean": selected.rolling(13).mean(), "13-week standard deviation": selected.rolling(13).std()})
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12, subplot_titles=("Level and rolling mean", "Rolling volatility"))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling["Observed"], name="Observed", line={"color": "#dce3e9"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling["13-week mean"], name="13-week mean", line={"color": "#6d9bc4"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling["13-week standard deviation"], name="13-week std", line={"color": "#c39b45"}), row=2, col=1)
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
        fig.add_trace(go.Bar(x=lag_frame["Lag"], y=lag_frame["ACF"], name="ACF", marker_color="#6d9bc4"), row=1, col=1)
        fig.add_trace(go.Bar(x=lag_frame["Lag"], y=lag_frame["PACF"], name="PACF", marker_color="#c39b45"), row=1, col=2)
        st.plotly_chart(style_figure(fig, 400), use_container_width=True)
    st.subheader("Indicator Correlation")
    corr = meta["transformed"].loc[start_display:].corr(min_periods=30)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=[[0, "#3f6c8f"], [.5, "#101820"], [1, "#a85252"]], zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title="Correlation of transformed stress-oriented indicators")
    st.plotly_chart(style_figure(fig, 550), use_container_width=True)
    footer()

elif page == "FORECAST":
    render_header("Forecasting", "Near-term stress projections with model-specific uncertainty and observable diagnostics.")
    model_choice = st.selectbox("MODEL", ["Naive", "ARIMA", "Exponential Smoothing"] + [m for m in models if m in {"SARIMA", "VAR"}], key="forecast_model")
    optimize = model_choice != "Naive"
    with st.status(f"Preparing {model_choice} forecast…", expanded=False) as forecast_status:
        forecast, forecast_error = get_page_forecast(model_choice, optimize=optimize)
        forecast_status.update(label=f"{forecast.model if forecast is not None else 'Forecast'} ready", state="complete" if forecast is not None else "error")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Model", forecast.model if forecast else "Unavailable")
    c2.metric("Forecast Endpoint", fmt(float(forecast.mean.iloc[-1])) if forecast else "N/A")
    c3.metric("Confidence Level", f"{confidence:.0%}")
    st.plotly_chart(forecast_chart(stress, forecast), use_container_width=True)
    if forecast_error:
        st.warning(f"The requested model failed and the Naive fallback was used: {forecast_error}")
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
            st.plotly_chart(style_figure(fig, 350, "Residual"), use_container_width=True)
        with right:
            st.json(forecast.details, expanded=True)
        diagnostics = get_diagnostics(meta["transformed"], meta["zscores"], contributions)
        st.subheader("Forecast Explanation")
        st.info(dynamic_warning_explanation(stress, diagnostics, forecast, previous_regime, current_regime))
    footer()

elif page == "EARLY WARNING":
    render_header("Early-Warning Signals", "Regime transitions, acceleration, threshold crossings, and mean-shift screening.")
    diagnostics = get_diagnostics(meta["transformed"], meta["zscores"], contributions)
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
    baseline_forecast, _ = get_page_forecast("Naive", optimize=False)
    st.info(dynamic_warning_explanation(stress, diagnostics, baseline_forecast, previous_regime, current_regime))
    st.caption(change_point["message"] + " This is a simple standardized mean-shift screen, not a formal causal break test.")
    footer()

elif page == "MODEL VALIDATION":
    render_header("Model Validation", "Chronological expanding-window validation. No random shuffling and no future information in training windows.")
    st.markdown('<div class="validation-banner"><b>VALIDATION IS EXPLICIT</b><span>Heavy model comparison is not executed during application startup.</span></div>', unsafe_allow_html=True)
    if "validation_result" not in st.session_state:
        st.session_state.validation_result = None
    if st.button("RUN EXPANDING-WINDOW VALIDATION", type="primary", use_container_width=False):
        with st.status("Running bounded expanding-window validation…", expanded=False) as validation_status:
            st.session_state.validation_result = run_validation(stress, tuple(models), multivariate)
            validation_status.update(label="Validation complete", state="complete")
    result = st.session_state.validation_result
    if result is None:
        st.info("Validation not yet run. The dashboard uses a transparent Naive baseline until validation is explicitly requested.")
    else:
        metrics, bt_predictions, bt_info = result
        if metrics.empty:
            st.warning("Model comparison is unavailable: insufficient common history or all candidate fits failed.")
        else:
            display_metrics = metrics.copy()
            display_metrics["Selection"] = np.where(display_metrics["Selected"], "SELECTED", "")
            st.dataframe(display_metrics.drop(columns="Selected").style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE (%)": "{:.2f}"}), use_container_width=True, hide_index=True)
            st.markdown(f'<div class="selection-note">Selection rule: {bt_info["criterion"]} / validated sample: {bt_info.get("sample_size", "N/A")} observations / origins: {bt_info.get("origins", "N/A")}</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for model, group in bt_predictions.groupby("Model"):
                fig.add_trace(go.Scatter(x=group["Date"], y=group["Forecast"], mode="lines+markers", name=model, opacity=.75))
            actual = bt_predictions.drop_duplicates("Date").sort_values("Date")
            fig.add_trace(go.Scatter(x=actual["Date"], y=actual["Actual"], mode="lines+markers", name="Actual", line={"color": "#ffffff", "width": 2.2}))
            st.plotly_chart(style_figure(fig, 480, "Stress index"), use_container_width=True)
        if bt_info.get("failures"):
            with st.expander("Unavailable model diagnostics"):
                st.json(bt_info["failures"])
    st.markdown("Each origin fits only observations available at that date, forecasts the next four weeks, then expands the training window. Metrics aggregate recent, non-overlapping origins. RMSE is the primary selection criterion and MAE breaks ties.")
    footer()

elif page == "EVENTS":
    render_header("Historical Stress Events", "Descriptive retrospective windows around recognized market-stress periods; no claim of event prediction.")
    events = {
        "Global Financial Crisis": (pd.Timestamp("2007-07-01"), pd.Timestamp("2009-06-30")),
        "COVID-19 market stress": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30")),
        "2023 US banking turmoil": (pd.Timestamp("2023-03-01"), pd.Timestamp("2023-05-31")),
    }
    event_name = st.selectbox("HISTORICAL WINDOW", list(events))
    event_start, event_end = events[event_name]
    window = stress.loc[event_start - pd.DateOffset(months=9):event_end + pd.DateOffset(months=6)]
    if window.empty:
        st.warning("The current data coverage does not include this event.")
    else:
        fig = stress_chart(window, title="Stress around selected historical window")
        fig.add_vrect(x0=event_start, x1=event_end, fillcolor="#ad5e62", opacity=.08, line_width=0, annotation_text=event_name)
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
    footer()

elif page == "METHODOLOGY":
    render_header("Methodology", "Reproducible academic design, assumptions, model selection, and limitations.")
    st.markdown(r"""
### 1. Data collection
Official public series are downloaded from FRED's CSV endpoint and aligned to Friday weeks. A bundled snapshot of the same official observations is used when available and live retrieval is unavailable or not requested. Synthetic data are never silently substituted.

### 2. Preprocessing and quality control
Timestamps are parsed, duplicates removed, values coerced to numeric, and series sorted chronologically. Daily and weekly series are converted to weekly last observations. Carry-forward is limited to four weeks so long outages remain visible. Extreme transformed observations are bounded only at the z-score stage (±4), preserving the raw data.

### 3. Exploratory time-series analysis
The application reports rolling means, rolling standard deviations, percentage changes, transformed-indicator correlations, ACF, PACF, and an Augmented Dickey–Fuller test with a plain-English interpretation.

### 4. Liquidity Stress Index
Seven components are oriented so positive values represent greater stress: deposit-growth deterioration, liquid-asset deterioration, reserve deterioration, funding cost, high-yield credit spread, VIX, and yield-curve inversion. Rolling moments are shifted one period and use only information available before the observation being scored.

### 5. Forecasting models
The Naive random-walk baseline is always available. ARIMA and damped additive Exponential Smoothing remain supported. SARIMA is available only when 13-week seasonality is supported by observed autocorrelation. VAR is available only when sufficient complete multivariate history exists. Model fitting is bounded to recent history where appropriate.

### 6. Validation and model selection
Validation uses expanding windows. Each model is trained on the past, forecasts a non-overlapping four-week block, then retrains after the origin advances. The lowest aggregate RMSE is selected, with MAE as the tie-breaker. No rows are shuffled. Validation is explicitly triggered from the Model Validation page and is not required to render the first dashboard.

### 7. Early warning and interpretation
Current regimes follow the project bands. Trend is the four-week score change; acceleration is the recent mean second difference. A transparent mean-shift screen compares the recent four-week mean with the preceding 26 weeks. Driver text is generated from actual z-scores, contributions, percentiles, and model intervals.

### 8. Granger causality
Granger tests, when run below, test whether lagged values improve prediction; they do not establish economic causality. Results should be interpreted with stationarity and multiple-testing limitations in mind.

### 9. Limitations
FRED revisions, publication lags, changing bank definitions, aggregation across institutions, structural breaks, and model misspecification affect results. The composite is not calibrated to supervisory loss data. Confidence intervals are conditional on model assumptions and can understate tail risk.
""")
    with st.expander("OPTIONAL GRANGER-CAUSALITY SCREEN"):
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
    footer()

elif page == "DATA":
    render_header("Data Sources", "Official series inventory, retrieval status, fallback policy, and data-quality diagnostics.")
    render_system_status(source_status, end)
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
