"""Streamlit dashboard — professional dark fintech UI.

Run:  streamlit run app.py
(Pre-trained models are included; retrain with:  python -m src.train)
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_loader import load_data
from src.predict import MODEL_FILES, NAME_TO_KEY, best_model_key, forecast

st.set_page_config(page_title="StockSight · ML Forecasting", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- design tokens
C = {
    "bg": "#0A0E1A", "card": "#121829", "card2": "#161D33", "border": "#222B45",
    "text": "#E6EAF5", "muted": "#8A93AD", "blue": "#5B8CFF", "violet": "#9B6BFF",
    "green": "#22C55E", "red": "#F43F5E", "amber": "#F5B544", "grid": "#1C2440",
}
MODEL_COLORS = {"Naive (last value)": C["muted"], "Ridge Regression": C["amber"],
                "Random Forest": C["violet"], "LSTM": C["blue"]}
ASSETS = {"AMZN": "Amazon.com Inc.", "DPZ": "Domino's Pizza Inc.",
          "BTC": "Bitcoin / USD", "NFLX": "Netflix Inc."}

_CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #18214099 0%, transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, #2a1b4d66 0%, transparent 55%), {C['bg']}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1440px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

.topbar {{ display:flex; align-items:center; justify-content:space-between;
          padding: 14px 22px; border:1px solid {C['border']}; border-radius:16px;
          background: linear-gradient(180deg, {C['card2']}, {C['card']}); margin-bottom: 18px; }}
.brand {{ display:flex; align-items:center; gap:12px; font-weight:700; font-size:1.15rem; color:{C['text']}; }}
.logo {{ width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
        background: linear-gradient(135deg, {C['blue']}, {C['violet']}); box-shadow: 0 6px 20px #5b8cff55; }}
.nav {{ display:flex; gap:26px; color:{C['muted']}; font-size:.88rem; }}
.nav span.active {{ color:{C['muted']}; }}
.badge {{ font-size:.72rem; padding:4px 10px; border-radius:999px; border:1px solid {C['border']};
         color:{C['muted']}; background:{C['bg']}; }}

.asset-title {{ font-size:1.6rem; font-weight:700; color:{C['text']}; margin:0; line-height:1.2; }}
.asset-sub {{ color:{C['muted']}; font-size:.86rem; margin-top:2px; }}
.pill {{ display:inline-block; font-size:.72rem; font-weight:600; padding:3px 10px; border-radius:999px;
        background:#5b8cff1f; color:{C['blue']}; border:1px solid #5b8cff44; margin-left:10px; vertical-align:middle; }}

.kpi {{ background: linear-gradient(180deg, {C['card2']}, {C['card']}); border:1px solid {C['border']};
       border-radius:14px; padding:16px 18px; height:100%; }}
.kpi .label {{ color:{C['muted']}; font-size:.78rem; letter-spacing:.02em; }}
.kpi .value {{ color:{C['text']}; font-size:1.55rem; font-weight:700; margin-top:6px;
              font-family:'JetBrains Mono', monospace; letter-spacing:-.02em; }}
.kpi .delta {{ font-size:.8rem; font-weight:600; margin-top:4px; }}
.up {{ color:{C['green']}; }} .down {{ color:{C['red']}; }} .flat {{ color:{C['muted']}; }}

.panel-title {{ color:{C['text']}; font-weight:600; font-size:1rem; margin-bottom:2px; }}
.panel-sub {{ color:{C['muted']}; font-size:.8rem; margin-bottom:6px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {C['card']}; border-color:{C['border']} !important;
       border-radius:16px !important; }}

.ftable {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
.ftable th {{ text-align:left; color:{C['muted']}; font-weight:500; padding:8px 6px; border-bottom:1px solid {C['border']}; }}
.ftable td {{ padding:7px 6px; border-bottom:1px solid {C['grid']}; color:{C['text']}; font-family:'JetBrains Mono', monospace; }}
.ftable tr:hover td {{ background:{C['card2']}; }}
.ftable td.up {{ color:{C['green']}; }} .ftable td.down {{ color:{C['red']}; }}

.note {{ border:1px solid #f5b54444; background:#f5b5440f; color:#E9CF98; border-radius:12px;
        padding:12px 16px; font-size:.84rem; margin-top:8px; }}
div[data-testid="stSegmentedControl"] button, .stSegmentedControl button {{ font-size:.82rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:10px;
        padding:6px 16px; color:{C['muted']}; }}
.stTabs [aria-selected="true"] {{ color:{C['text']} !important; border-color:{C['blue']} !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
</style>
"""
# Blank lines would end the HTML block in Streamlit markdown, so strip them.
st.markdown("\n".join(l for l in _CSS.splitlines() if l.strip()), unsafe_allow_html=True)


# ------------------------------------------------------------------- data access
@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_forecast(ticker, days, model_key):
    return forecast(ticker, days, model_key)


@st.cache_data
def get_metrics(ticker):
    p = config.REPORTS_DIR / f"metrics_{ticker}.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def get_backtest(ticker):
    p = config.REPORTS_DIR / f"backtest_{ticker}.csv"
    return pd.read_csv(p, parse_dates=["Date"]) if p.exists() else None


def fmt_price(x):
    return f"${x:,.2f}"


def delta_html(pct, suffix=""):
    cls = "up" if pct > 0 else "down" if pct < 0 else "flat"
    arrow = "▲" if pct > 0 else "▼" if pct < 0 else "•"
    return f'<div class="delta {cls}">{arrow} {pct:+.2f}%{suffix}</div>'


def kpi(label, value, delta=""):
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>{delta}</div>'


def style_fig(fig, height=430):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C["muted"], size=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["card2"], bordercolor=C["border"], font_color=C["text"]),
        legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)", font_color=C["text"]),
        xaxis=dict(gridcolor=C["grid"], zeroline=False, showline=False),
        yaxis=dict(gridcolor=C["grid"], zeroline=False, showline=False, tickprefix="$", side="right"),
    )
    return fig


df = get_data()

# ------------------------------------------------------------------------ top bar
st.markdown(f"""
<div class="topbar">
  <div class="brand"><div class="logo">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.4"
         stroke-linecap="round" stroke-linejoin="round"><polyline points="3 17 9 11 13 15 21 7"/>
         <polyline points="15 7 21 7 21 13"/></svg></div>StockSight</div>
  <div class="nav"><span class="active">ML price forecasting · AMZN · DPZ · BTC · NFLX</span></div>
  <span class="badge">Data: May 2013 → May 2019 · Educational, not financial advice</span>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------ controls
c_asset, c_model, c_h = st.columns([2.2, 2.2, 1.6], vertical_alignment="bottom")
with c_asset:
    ticker = st.segmented_control("Asset", list(ASSETS), default="AMZN", key="ticker") or "AMZN"
available = [k for k, f in MODEL_FILES.items() if (config.MODELS_DIR / f"{ticker}_{f}").exists()]
if not available:
    st.error("No trained models found. Run `python -m src.train`, then reload.")
    st.stop()
labels = {v: k for k, v in NAME_TO_KEY.items()}
best_key = best_model_key(ticker)
with c_model:
    model_key = st.segmented_control(
        "Model", available, default=best_key if best_key in available else available[0],
        format_func=lambda k: labels[k] + (" ★" if k == best_key else ""), key=f"model_{ticker}",
    ) or best_key
with c_h:
    days = st.select_slider("Forecast horizon (business days)", [1, 3, 5, 7, 10, 14, 21, 30], value=7)

series = df[ticker]
fc = get_forecast(ticker, days, model_key)
fc_col = f"{ticker}_predicted"
info = get_metrics(ticker)
table = pd.DataFrame({k: v["test"] for k, v in info["results"].items()}).T if info else None

last, prev = series.iloc[-1], series.iloc[-2]
end_price = fc[fc_col].iloc[-1]
rets = series.pct_change().dropna()
vol_daily = rets.iloc[-60:].std()

# ------------------------------------------------------------------------ header + KPIs
st.markdown(
    f'<div style="margin:10px 0 14px"><p class="asset-title">{ASSETS[ticker]} '
    f'<span style="color:{C["muted"]};font-weight:500">· {ticker}</span>'
    f'<span class="pill">{labels[model_key]}</span></p>'
    f'<div class="asset-sub">Last close {series.index[-1]:%d %b %Y} · '
    f'{days}-day recursive forecast</div></div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(kpi("Last close", fmt_price(last), delta_html((last / prev - 1) * 100, " today")),
            unsafe_allow_html=True)
k2.markdown(kpi(f"Forecast in {days}d", fmt_price(end_price), delta_html((end_price / last - 1) * 100)),
            unsafe_allow_html=True)
if table is not None:
    sel, naive = table.loc[labels[model_key]], table.loc["Naive (last value)"]
    gap = (sel["RMSE"] / naive["RMSE"] - 1) * 100
    cls = "up" if gap < 0 else "down"
    k3.markdown(kpi("Test RMSE", f"{sel['RMSE']:,.2f}",
                    f'<div class="delta {cls}">{gap:+.1f}% vs naive baseline</div>'),
                unsafe_allow_html=True)
    k4.markdown(kpi("Test MAPE", f"{sel['MAPE_%']:.2f}%", '<div class="delta flat">mean abs. % error</div>'),
                unsafe_allow_html=True)
    k5.markdown(kpi("Direction accuracy", f"{sel['Direction_acc_%']:.1f}%",
                    '<div class="delta flat">50% ≈ coin flip</div>'), unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------------ main row
left, right = st.columns([2.35, 1], gap="medium")

with left, st.container(border=True):
    t1, t2 = st.columns([1.4, 1], vertical_alignment="center")
    t1.markdown('<div class="panel-title">Price & forecast</div>'
                '<div class="panel-sub">Shaded band: 95% range implied by recent volatility</div>',
                unsafe_allow_html=True)
    with t2:
        rng = st.segmented_control("Range", ["1M", "3M", "6M", "1Y", "3Y", "ALL"], default="6M",
                                   label_visibility="collapsed", key="range") or "6M"
    span = {"1M": 31, "3M": 92, "6M": 183, "1Y": 365, "3Y": 1095, "ALL": 10_000}[rng]
    hist = series.loc[series.index[-1] - pd.Timedelta(days=span):]

    h = np.arange(1, days + 1)
    band = 1.96 * vol_daily * np.sqrt(h)
    upper, lower = fc[fc_col] * (1 + band), fc[fc_col] * (1 - band)
    fx = [hist.index[-1], *fc["Date"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, mode="lines", name="Historical",
                             line=dict(color=C["blue"], width=2.2),
                             fill="tozeroy", fillgradient=dict(type="vertical",
                             colorscale=[[0, "rgba(91,140,255,0)"], [1, "rgba(91,140,255,0.28)"]])))
    fig.add_trace(go.Scatter(x=fx, y=[last, *upper], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=[last, *lower], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(155,107,255,0.16)", name="95% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=[last, *fc[fc_col]], mode="lines+markers", name="Forecast",
                             line=dict(color=C["violet"], width=2.4, dash="dot"),
                             marker=dict(size=6, color=C["violet"], line=dict(color=C["bg"], width=1.5))))
    fig.add_vline(x=hist.index[-1], line=dict(color=C["border"], width=1, dash="dash"))
    style_fig(fig, 520)
    ymin = min(hist.min(), lower.min())
    ymax = max(hist.max(), upper.max())
    pad = (ymax - ymin) * 0.08
    fig.update_yaxes(range=[ymin - pad, ymax + pad])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Model comparison</div>'
                    '<div class="panel-sub">Test RMSE — lower is better</div>', unsafe_allow_html=True)
        if table is not None:
            tb = table.sort_values("RMSE", ascending=False)
            bar = go.Figure(go.Bar(
                x=tb["RMSE"], y=tb.index, orientation="h",
                marker=dict(color=[MODEL_COLORS[m] for m in tb.index],
                            line=dict(width=0), cornerradius=6),
                text=[f"{v:,.2f}" for v in tb["RMSE"]], textposition="outside",
                textfont=dict(color=C["text"], family="JetBrains Mono"),
                hovertemplate="%{y}: %{x:,.2f}<extra></extra>"))
            style_fig(bar, 210)
            bar.update_layout(showlegend=False, margin=dict(l=8, r=40, t=4, b=4), hovermode="closest")
            bar.update_xaxes(showticklabels=False, showgrid=False, range=[0, tb["RMSE"].max() * 1.25])
            bar.update_yaxes(tickprefix="", side="left", showgrid=False, tickfont=dict(color=C["text"]))
            st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

    with st.container(border=True):
        st.markdown('<div class="panel-title">Forecast schedule</div>'
                    f'<div class="panel-sub">{labels[model_key]} · next {days} business days</div>',
                    unsafe_allow_html=True)
        rows, ref = [], last
        for d, p in zip(fc["Date"], fc[fc_col]):
            ch = (p / ref - 1) * 100
            cls = "up" if ch > 0 else "down" if ch < 0 else "flat"
            rows.append(f"<tr><td>{d:%a %d %b}</td><td>{fmt_price(p)}</td>"
                        f'<td class="{cls}">{ch:+.2f}%</td></tr>')
            ref = p
        st.markdown('<div style="max-height:296px;overflow-y:auto"><table class="ftable">'
                    "<tr><th>Date</th><th>Price</th><th>Δ day</th></tr>"
                    + "".join(rows) + "</table></div>", unsafe_allow_html=True)
        st.download_button("⬇  Download CSV", fc.to_csv(index=False), f"forecast_{ticker}.csv",
                           "text/csv", width="stretch")

st.write("")

# ------------------------------------------------------------------------ tabs
tab_bt, tab_models, tab_method = st.tabs(["Backtest", "Model metrics", "Methodology"])

with tab_bt:
    bt = get_backtest(ticker)
    if bt is None:
        st.info("Run `python -m src.train` to generate backtest data.")
    else:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Next-day predictions on unseen data</div>'
                        f'<div class="panel-sub">Test period {bt.Date.min():%b %Y} → {bt.Date.max():%b %Y}. '
                        'Click legend items to toggle models.</div>', unsafe_allow_html=True)
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=bt.Date, y=bt.Actual, name="Actual",
                                    line=dict(color=C["text"], width=2.4)))
            for m in ["LSTM", "Random Forest", "Ridge Regression"]:
                if m in bt:
                    f2.add_trace(go.Scatter(x=bt.Date, y=bt[m], name=m, line=dict(color=MODEL_COLORS[m], width=1.6),
                                            opacity=0.9, visible=True if m == labels[model_key] else "legendonly"))
            style_fig(f2, 400)
            st.plotly_chart(f2, width="stretch", config={"displayModeBar": False})

with tab_models:
    if table is not None:
        with st.container(border=True):
            show = table.rename(columns={"MAPE_%": "MAPE (%)", "Direction_acc_%": "Direction acc. (%)"})
            st.dataframe(
                show.style.format("{:,.3f}", na_rep="—")
                    .highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], color="#22c55e33"),
                width="stretch")
            st.markdown(f'<div class="panel-sub">★ Best on validation: <b style="color:{C["text"]}">'
                        f'{info["best_model"]}</b>. A model that cannot beat “Naive (last value)” '
                        "has learned no exploitable signal.</div>", unsafe_allow_html=True)

with tab_method:
    m1, m2, m3 = st.columns(3)
    for col, title, body in [
        (m1, "Leak-free split", "70% train · 15% validation · 15% test, kept in time order. "
                                "Shuffling a time series lets the model see the future."),
        (m2, "Scale-free targets", "Each 60-day window is expressed relative to its last price and "
                                   "the model predicts the next-day % change, so it works at any price level."),
        (m3, "Honest benchmark", "Every model is compared with the naive “tomorrow = today” forecast. "
                                 "Close to 50% direction accuracy means the market is near a random walk."),
    ]:
        col.markdown(kpi(title, "", f'<div style="color:{C["muted"]};font-size:.85rem;margin-top:-4px">{body}</div>')
                     .replace('<div class="value"></div>', ""), unsafe_allow_html=True)

st.markdown('<div class="note">⚠ Multi-step forecasts feed each prediction back as input, so errors '
            "compound. Treat anything beyond a few days as illustrative, not as a trading signal.</div>",
            unsafe_allow_html=True)
