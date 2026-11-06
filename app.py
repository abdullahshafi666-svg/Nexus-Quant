"""
NEXUS-QUANT: Modular Quantitative Backtesting & Fundamental Analysis Platform
Streamlit Frontend with iOS Frosted Glass Dark Theme

Author: NEXUS-QUANT System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# Local module imports
from data_engine import (
    ASSET_UNIVERSE,
    download_data,
    load_all_universe,
    fetch_economic_calendar,
    get_high_impact_events,
    is_news_blackout,
)
from indicators import add_all_indicators
from backtester import BacktestEngine, TradeParams, BacktestResult
from fundamental_engine import (
    get_fundamental_card,
    calculate_key_metrics,
    calculate_piotroski_f_score,
    compare_fundamentals,
)
from ai_agent import AIAdvisor

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NEXUS-QUANT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

FROSTED_GLASS_CSS = """
<style>
/* ── ROOT PALETTE ─────────────────────────────────────────── */
:root {
    --canvas-bg:       #0B0C0E;
    --card-surface:    rgba(20, 22, 26, 0.82);
    --border-frame:    #272A30;
    --accent-volt:     #E2F952;
    --accent-ice:      #38BDF8;
    --bullish:         #22C55E;
    --bearish:         #EF4444;
    --text-primary:    #F3F4F6;
    --text-muted:      #6B7280;
}

/* ── GLOBAL OVERRIDES ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--canvas-bg) !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background-color: rgba(11, 12, 14, 0.92) !important;
    border-right: 1px solid var(--border-frame) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── FROSTED GLASS CARDS ──────────────────────────────────── */
[data-testid="stVerticalBlock"] > div > div[data-testid="stExpander"],
div[data-testid="stMetric"],
div.stTabs,
section[data-testid="stSidebar"] > div {
    background: var(--card-surface) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-frame) !important;
    border-radius: 14px !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--card-surface) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-frame) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] svg { display: none; }

/* ── HEADER BANNER ────────────────────────────────────────── */
.nexus-header {
    background: linear-gradient(135deg, rgba(226,249,82,0.08) 0%, rgba(56,189,248,0.06) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-frame);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
}
.nexus-header h1 {
    color: var(--accent-volt) !important;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}
.nexus-header p {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin: 0;
}

/* ── FROSTED PANEL WRAPPER ────────────────────────────────── */
.frost-panel {
    background: var(--card-surface);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-frame);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.frost-panel h3 {
    color: var(--accent-volt) !important;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── BUTTONS ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #E2F952 0%, #c8e040 100%) !important;
    color: #0B0C0E !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(226,249,82,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── TABS ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-volt) !important;
    border-bottom: 2px solid var(--accent-volt) !important;
}

/* ── INPUT WIDGETS ────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > textarea {
    background-color: rgba(20,22,26,0.9) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-frame) !important;
    border-radius: 10px !important;
}

/* ── DATAFRAME ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-frame) !important;
    border-radius: 10px !important;
}

/* ── CHAT INPUT ───────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: rgba(20,22,26,0.9) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-frame) !important;
    border-radius: 12px !important;
}

/* ── SCROLLBAR ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--canvas-bg); }
::-webkit-scrollbar-thumb { background: var(--border-frame); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3d44; }

/* ── EXPANDER ─────────────────────────────────────────────── */
[data-testid="stExpander"] details {
    border: 1px solid var(--border-frame) !important;
    border-radius: 14px !important;
    background: var(--card-surface) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
}

/* ── DIVIDER ──────────────────────────────────────────────── */
hr {
    border-color: var(--border-frame) !important;
}

/* ── SIDEBAR LABELS ───────────────────────────────────────── */
.sidebar-section-label {
    color: var(--accent-volt);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 18px;
    margin-bottom: 6px;
    padding-left: 2px;
}
</style>
"""

st.markdown(FROSTED_GLASS_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "data_loaded": False,
        "market_data": {},
        "calendar_df": None,
        "backtest_result": None,
        "ai_messages": [],
        "ai_provider": "gemini",
        "ai_model": "gemini-3.8-flash",
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", "").strip(),
        "selected_symbol": "BTC-USD",
        "selected_interval": "15m",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─────────────────────────────────────────────────────────────
# HELPER: PLOTLY THEME
# ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,16,20,0.6)",
    font=dict(color="#F3F4F6", family="Inter, system-ui, sans-serif"),
    xaxis=dict(
        gridcolor="rgba(39,42,48,0.5)",
        zerolinecolor="#272A30",
    ),
    yaxis=dict(
        gridcolor="rgba(39,42,48,0.5)",
        zerolinecolor="#272A30",
    ),
    margin=dict(l=40, r=20, t=40, b=30),
    legend=dict(
        bgcolor="rgba(20,22,26,0.8)",
        bordercolor="#272A30",
        borderwidth=1,
        font=dict(size=11),
    ),
)


def get_symbol_info(symbol: str) -> tuple[str, str, str]:
    """Return (category, interval, period) for a symbol from the universe."""
    for category, meta in ASSET_UNIVERSE.items():
        if isinstance(meta, dict):
            if symbol in meta.get("symbols", []):
                return category, meta.get("interval", "1d"), meta.get("period", "60d")
        elif isinstance(meta, list):
            for asset in meta:
                if isinstance(asset, dict) and asset.get("symbol") == symbol:
                    return category, asset.get("interval", "1d"), asset.get("period", "60d")
    return "Unknown", "1d", "1y"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar controls."""
    with st.sidebar:
        st.markdown(
            '<div class="nexus-header" style="padding:16px 20px;">'
            '<h1 style="font-size:1.5rem;">⚡ NEXUS-QUANT</h1>'
            '<p>Quantitative Research Terminal</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Asset Selection ──
        st.markdown('<div class="sidebar-section-label">Asset Selection</div>', unsafe_allow_html=True)

        all_symbols = []
        symbol_labels = {}
        for category, meta in ASSET_UNIVERSE.items():
            if isinstance(meta, dict):
                symbols = meta.get("symbols", [])
                interval = meta.get("interval", "1d")
                for sym in symbols:
                    all_symbols.append(sym)
                    symbol_labels[sym] = f"{sym}  ({category} · {interval})"
            elif isinstance(meta, list):
                for asset in meta:
                    sym = asset.get("symbol", "")
                    interval = asset.get("interval", "1d")
                    all_symbols.append(sym)
                    symbol_labels[sym] = f"{sym}  ({category} · {interval})"

        selected = st.selectbox(
            "Symbol",
            all_symbols,
            format_func=lambda s: symbol_labels.get(s, s),
            index=all_symbols.index(st.session_state.selected_symbol)
            if st.session_state.selected_symbol in all_symbols
            else 0,
            key="sb_symbol",
        )
        st.session_state.selected_symbol = selected
        cat, interval, period = get_symbol_info(selected)
        st.session_state.selected_interval = interval

        # ── Data Controls ──
        st.markdown('<div class="sidebar-section-label">Data Pipeline</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📡 Fetch", use_container_width=True):
                with st.spinner(f"Downloading {selected}..."):
                    try:
                        df = download_data(selected, interval, period)
                        if df is not None and not df.empty:
                            st.session_state.market_data[selected] = df
                            st.session_state.data_loaded = True
                            st.success(f"✓ {len(df)} bars")
                        else:
                            st.error("No data returned")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            if st.button("📡 All", use_container_width=True):
                with st.spinner("Downloading full universe..."):
                    try:
                        data = load_all_universe()
                        st.session_state.market_data.update(data)
                        st.session_state.data_loaded = True
                        st.success(f"✓ {len(data)} assets")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.button("📅 Load Calendar", use_container_width=True):
            with st.spinner("Fetching economic calendar..."):
                try:
                    cal = fetch_economic_calendar()
                    st.session_state.calendar_df = cal
                    st.success(f"✓ {len(cal)} events")
                except Exception as e:
                    st.error(f"Calendar: {e}")

        # ── Backtest Parameters ──
        st.markdown('<div class="sidebar-section-label">Backtest Parameters</div>', unsafe_allow_html=True)

        sl_buffer = st.number_input(
            "SL Buffer (points)", value=0.0002, format="%.4f", step=0.0001, key="sl_buf"
        )
        tp_rr = st.slider("TP Risk:Reward", 1.5, 4.0, 2.5, 0.1, key="tp_rr")
        be_rr = st.slider("Breakeven RR Trigger", 0.5, 2.0, 1.0, 0.1, key="be_rr")
        slippage = st.number_input(
            "Slippage", value=0.0001, format="%.4f", step=0.0001, key="slip"
        )
        spread = st.number_input(
            "Spread", value=0.0002, format="%.4f", step=0.0001, key="sprd"
        )
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.25, key="risk_pct")
        capital = st.number_input(
            "Initial Capital ($)", value=10000, step=1000, key="capital"
        )

        # ── Indicator Parameters ──
        st.markdown('<div class="sidebar-section-label">Indicator Settings</div>', unsafe_allow_html=True)

        fast_ma = st.number_input("Fast MA", value=20, min_value=5, max_value=100, key="fast_ma")
        slow_ma = st.number_input("Slow MA", value=50, min_value=10, max_value=200, key="slow_ma")
        rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=50, key="rsi_per")
        sr_left = st.number_input("S&R Left Bars", value=5, min_value=2, max_value=20, key="sr_l")
        sr_right = st.number_input("S&R Right Bars", value=5, min_value=2, max_value=20, key="sr_r")

        # ── AI Advisor (Fixed to Gemini 3.8 Flash) ──
        st.markdown('<div class="sidebar-section-label">AI Advisor (Gemini 3.8 Flash)</div>', unsafe_allow_html=True)

        st.session_state.ai_provider = "gemini"
        st.session_state.ai_model = "gemini-3.8-flash"

        saved_key = os.environ.get("GEMINI_API_KEY", "").strip()
        current_key = st.session_state.get("gemini_api_key") or saved_key

        api_key = st.text_input(
            "Gemini API Key",
            value=current_key,
            type="password",
            help="Saved permanently in your .env file. You never need to paste it again.",
            key="gemini_key_input",
            placeholder="AIzaSy...",
        )
        if api_key:
            st.session_state["gemini_api_key"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key
        elif saved_key:
            st.session_state["gemini_api_key"] = saved_key

        if saved_key or api_key:
            st.caption("🔒 Key loaded permanently from `.env`")

        # ── Branding (Bottom Left) ──
        st.markdown(
            """
            <div style="
                margin-top: 35px;
                padding: 12px 14px;
                background: rgba(20, 22, 26, 0.65);
                border: 1px solid #272A30;
                border-radius: 10px;
                text-align: center;
                color: #6B7280;
                font-size: 0.8rem;
                letter-spacing: 0.02em;
            ">
                ⚡ Made by <strong style="color: #E2F952;">Abdullah (Ronin)</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return {
            "symbol": selected,
            "interval": interval,
            "period": period,
            "category": cat,
            "sl_buffer": sl_buffer,
            "tp_rr": tp_rr,
            "be_rr": be_rr,
            "slippage": slippage,
            "spread": spread,
            "risk_pct": risk_pct,
            "capital": capital,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "rsi_period": rsi_period,
            "sr_left": sr_left,
            "sr_right": sr_right,
        }


# ─────────────────────────────────────────────────────────────
# CHART: CANDLESTICK + OVERLAYS
# ─────────────────────────────────────────────────────────────

def render_candlestick_chart(df: pd.DataFrame, symbol: str, show_fvg: bool = True, trades: list = None):
    """Render interactive Plotly candlestick with indicator overlays."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} Price Action", "RSI (14)", "Volume"],
    )

    # ── Candlestick ──
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444",
            increasing_fillcolor="#22C55E",
            decreasing_fillcolor="#EF4444",
            name="Price",
        ),
        row=1,
        col=1,
    )

    # ── Moving Averages ──
    if "MA_fast" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA_fast"],
                line=dict(color="#38BDF8", width=1.5),
                name="Fast MA",
                opacity=0.85,
            ),
            row=1,
            col=1,
        )
    if "MA_slow" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA_slow"],
                line=dict(color="#E2F952", width=1.5),
                name="Slow MA",
                opacity=0.85,
            ),
            row=1,
            col=1,
        )

    # ── Support & Resistance Lines ──
    if "support_level" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["support_level"],
                line=dict(color="#22C55E", width=1, dash="dot"),
                name="Support",
                opacity=0.6,
            ),
            row=1,
            col=1,
        )
    if "resistance_level" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["resistance_level"],
                line=dict(color="#EF4444", width=1, dash="dot"),
                name="Resistance",
                opacity=0.6,
            ),
            row=1,
            col=1,
        )

    # ── Pivot Points Markers ──
    if "pivot_high" in df.columns:
        ph = df[df["pivot_high"].notna()]
        fig.add_trace(
            go.Scatter(
                x=ph.index,
                y=ph["pivot_high"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=8, color="#EF4444"),
                name="Pivot High",
            ),
            row=1,
            col=1,
        )
    if "pivot_low" in df.columns:
        pl = df[df["pivot_low"].notna()]
        fig.add_trace(
            go.Scatter(
                x=pl.index,
                y=pl["pivot_low"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=8, color="#22C55E"),
                name="Pivot Low",
            ),
            row=1,
            col=1,
        )

    # ── Fair Value Gaps (shaded rectangles) ──
    if show_fvg and "fvg_type" in df.columns:
        fvg_rows = df[df["fvg_type"].notna()].tail(30)
        bar_delta = (df.index[1] - df.index[0]) * 6 if len(df) > 1 else pd.Timedelta(hours=2)
        for idx, row in fvg_rows.iterrows():
            fvg_top = row.get("fvg_top", 0)
            fvg_bot = row.get("fvg_bottom", 0)
            if pd.isna(fvg_top) or pd.isna(fvg_bot) or fvg_top == 0 or fvg_bot == 0:
                continue
            color = "rgba(34,197,94,0.18)" if row["fvg_type"] == "BULLISH_FVG" else "rgba(239,68,68,0.18)"
            border_color = "rgba(34,197,94,0.5)" if row["fvg_type"] == "BULLISH_FVG" else "rgba(239,68,68,0.5)"
            fig.add_shape(
                type="rect",
                x0=idx,
                x1=idx + bar_delta,
                y0=fvg_bot,
                y1=fvg_top,
                fillcolor=color,
                line=dict(color=border_color, width=1),
                row=1,
                col=1,
            )

    # ── Trade Markers ──
    if trades:
        for trade in trades:
            # Entry marker
            entry_color = "#22C55E" if trade.direction == "LONG" else "#EF4444"
            entry_symbol = "triangle-up" if trade.direction == "LONG" else "triangle-down"
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time],
                    y=[trade.entry_price],
                    mode="markers",
                    marker=dict(symbol=entry_symbol, size=12, color=entry_color, line=dict(width=1, color="white")),
                    name=f"{'▲' if trade.direction == 'LONG' else '▼'} Entry",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            # Exit marker
            if trade.exit_time and trade.exit_price:
                exit_color = "#22C55E" if trade.pnl > 0 else "#EF4444"
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_price],
                        mode="markers",
                        marker=dict(symbol="x", size=10, color=exit_color, line=dict(width=1, color="white")),
                        name="Exit",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

    # ── RSI ──
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                line=dict(color="#38BDF8", width=1.5),
                name="RSI",
            ),
            row=2,
            col=1,
        )
        # Overbought / Oversold zones
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.4)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.4)", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(107,114,128,0.3)", row=2, col=1)
        # Shade zones
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.05)", line_width=0, row=2, col=1)

    # ── Volume ──
    if "Volume" in df.columns:
        colors = [
            "#22C55E" if c >= o else "#EF4444" for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume", opacity=0.5),
            row=3,
            col=1,
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=720,
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_xaxes(gridcolor="rgba(39,42,48,0.4)")
    fig.update_yaxes(gridcolor="rgba(39,42,48,0.4)")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


# ─────────────────────────────────────────────────────────────
# CHART: EQUITY CURVE
# ─────────────────────────────────────────────────────────────

def render_equity_curve(equity_curve: pd.Series, initial_capital: float):
    """Render equity curve with drawdown overlay."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.7, 0.3], subplot_titles=["Equity Curve", "Drawdown %"],
    )

    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            fill="tozeroy",
            fillcolor="rgba(226,249,82,0.08)",
            line=dict(color="#E2F952", width=2),
            name="Equity",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=initial_capital, line_dash="dot", line_color="rgba(107,114,128,0.4)", row=1, col=1)

    # Drawdown
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak * 100
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)",
            line=dict(color="#EF4444", width=1.5),
            name="Drawdown %",
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(**PLOTLY_LAYOUT, height=450, showlegend=True)
    fig.update_yaxes(title_text="Equity ($)", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", tickformat=".2f", ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB: MARKET ANALYSIS
# ─────────────────────────────────────────────────────────────

def tab_market_analysis(params: dict):
    """Market analysis tab with chart and current setup."""
    symbol = params["symbol"]

    if symbol not in st.session_state.market_data:
        st.info(f"📡 Click **Fetch** in the sidebar to download data for **{symbol}**.")
        return

    df = st.session_state.market_data[symbol].copy()

    # Apply indicators
    try:
        df = add_all_indicators(
            df,
            fast_ma=params["fast_ma"],
            slow_ma=params["slow_ma"],
            rsi_period=params["rsi_period"],
            sr_left=params["sr_left"],
            sr_right=params["sr_right"],
        )
        st.session_state.market_data[symbol] = df
    except Exception as e:
        st.error(f"Indicator calculation error: {e}")
        return

    # ── Current Setup Summary ──
    latest = df.iloc[-1]
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        price = latest["Close"]
        prev_close = df.iloc[-2]["Close"] if len(df) > 1 else price
        delta = ((price - prev_close) / prev_close * 100)
        st.metric("Price", f"{price:.4f}" if price < 100 else f"{price:.2f}", f"{delta:+.2f}%")
    with col2:
        bias = latest.get("trend_bias", "N/A")
        bias_icon = "🟢" if bias == "LONG" else "🔴" if bias == "SHORT" else "⚪"
        st.metric("Trend Bias", f"{bias_icon} {bias}")
    with col3:
        rsi_val = latest.get("RSI", 0)
        rsi_mode = latest.get("rsi_mode", "N/A")
        st.metric("RSI", f"{rsi_val:.1f}", rsi_mode)
    with col4:
        support = latest.get("support_level", 0)
        st.metric("Support", f"{support:.4f}" if support and support < 100 else f"{support:.2f}" if support else "N/A")
    with col5:
        resist = latest.get("resistance_level", 0)
        st.metric("Resistance", f"{resist:.4f}" if resist and resist < 100 else f"{resist:.2f}" if resist else "N/A")

    # ── FVG Status ──
    fvg_cols = st.columns(4)
    with fvg_cols[0]:
        bull_fvg_top = latest.get("active_bull_fvg_top", None)
        bull_fvg_bot = latest.get("active_bull_fvg_bottom", None)
        if bull_fvg_top and not pd.isna(bull_fvg_top):
            st.markdown(f"<div class='frost-panel'><h3>🟢 Bullish FVG</h3>{bull_fvg_bot:.4f} → {bull_fvg_top:.4f}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='frost-panel'><h3>Bullish FVG</h3>None active</div>", unsafe_allow_html=True)
    with fvg_cols[1]:
        bear_fvg_top = latest.get("active_bear_fvg_top", None)
        bear_fvg_bot = latest.get("active_bear_fvg_bottom", None)
        if bear_fvg_top and not pd.isna(bear_fvg_top):
            st.markdown(f"<div class='frost-panel'><h3>🔴 Bearish FVG</h3>{bear_fvg_bot:.4f} → {bear_fvg_top:.4f}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='frost-panel'><h3>Bearish FVG</h3>None active</div>", unsafe_allow_html=True)
    with fvg_cols[2]:
        ifvg_type = latest.get("ifvg_type", None)
        if ifvg_type and not pd.isna(ifvg_type):
            ifvg_top = latest.get("ifvg_top", 0)
            ifvg_bot = latest.get("ifvg_bottom", 0)
            st.markdown(f"<div class='frost-panel'><h3>🔄 IFVG</h3>{ifvg_type}<br>{ifvg_bot:.4f} → {ifvg_top:.4f}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='frost-panel'><h3>IFVG</h3>None active</div>", unsafe_allow_html=True)
    with fvg_cols[3]:
        bars_count = len(df)
        date_range = f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
        st.markdown(f"<div class='frost-panel'><h3>📊 Data</h3>{bars_count} bars<br>{date_range}</div>", unsafe_allow_html=True)

    # ── Chart Controls ──
    col_fvg, col_range = st.columns([1, 2])
    with col_fvg:
        show_fvg = st.checkbox("Show FVG Zones", value=True, key="show_fvg")
    with col_range:
        display_bars = st.select_slider(
            "Visible Bars on Chart",
            options=[100, 200, 300, 500, 1000, "All"],
            value=200,
            key="visible_bars",
            help="Showing recent bars makes the chart load instantly.",
        )

    chart_df = df if display_bars == "All" else df.tail(display_bars)

    trades_to_show = None
    if st.session_state.backtest_result:
        trades_to_show = st.session_state.backtest_result.trades
    render_candlestick_chart(chart_df, symbol, show_fvg=show_fvg, trades=trades_to_show)

    # ── Raw Data ──
    with st.expander("📋 Raw Data (last 50 bars)"):
        display_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "MA_fast", "MA_slow",
                                      "RSI", "trend_bias", "rsi_mode", "support_level", "resistance_level",
                                      "fvg_type"] if c in df.columns]
        st.dataframe(df[display_cols].tail(50), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB: BACKTESTER
# ─────────────────────────────────────────────────────────────

def tab_backtester(params: dict):
    """Backtesting tab with engine execution and results."""
    symbol = params["symbol"]

    if symbol not in st.session_state.market_data:
        st.info(f"📡 Fetch data for **{symbol}** first.")
        return

    df = st.session_state.market_data[symbol].copy()

    # Ensure indicators are calculated
    if "trend_bias" not in df.columns:
        df = add_all_indicators(
            df,
            fast_ma=params["fast_ma"],
            slow_ma=params["slow_ma"],
            rsi_period=params["rsi_period"],
            sr_left=params["sr_left"],
            sr_right=params["sr_right"],
        )
        st.session_state.market_data[symbol] = df

    st.markdown(
        "<div class='frost-panel'><h3>⚡ Backtest Engine</h3>"
        "Deterministic bar-by-bar execution with SL buffer, BE trail at 1:1 RR, and configurable TP.</div>",
        unsafe_allow_html=True,
    )

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        run_bt = st.button("🚀 Run Backtest", use_container_width=True, key="run_bt")
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_bt"):
            st.session_state.backtest_result = None
            st.rerun()

    if run_bt:
        trade_params = TradeParams(
            sl_buffer_points=params["sl_buffer"],
            tp_rr_ratio=params["tp_rr"],
            breakeven_rr=params["be_rr"],
            slippage=params["slippage"],
            spread=params["spread"],
            risk_per_trade_pct=params["risk_pct"],
            initial_capital=params["capital"],
        )

        engine = BacktestEngine(trade_params)

        # News blackout
        news_times = None
        if st.session_state.calendar_df is not None:
            hi_events = get_high_impact_events(st.session_state.calendar_df)
            if not hi_events.empty and "datetime" in hi_events.columns:
                news_times = hi_events["datetime"].tolist()

        with st.spinner("Running backtest..."):
            try:
                result = engine.run(df, news_blackout_times=news_times)
                st.session_state.backtest_result = result
                st.success(f"✓ Backtest complete — {len(result.trades)} trades executed")
            except Exception as e:
                st.error(f"Backtest error: {e}")
                st.code(traceback.format_exc())
                return

    # ── Display Results ──
    result = st.session_state.backtest_result
    if result is None:
        st.info("Configure parameters in the sidebar and click **Run Backtest**.")
        return

    metrics = result.metrics

    # ── Metrics Dashboard ──
    st.markdown("---")
    st.markdown("<div class='frost-panel'><h3>📊 Performance Metrics</h3></div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Total Trades", metrics.get("total_trades", 0))
    with m2:
        wr = metrics.get("win_rate", 0)
        st.metric("Win Rate", f"{wr:.1f}%")
    with m3:
        pf = metrics.get("profit_factor", 0)
        st.metric("Profit Factor", f"{pf:.2f}")
    with m4:
        sr = metrics.get("sharpe_ratio", 0)
        st.metric("Sharpe Ratio", f"{sr:.2f}")
    with m5:
        mdd = metrics.get("max_drawdown_pct", 0)
        st.metric("Max Drawdown", f"{mdd:.2f}%")
    with m6:
        total_ret = metrics.get("total_return_pct", 0)
        color = "normal" if total_ret >= 0 else "inverse"
        st.metric("Total Return", f"{total_ret:.2f}%")

    m7, m8, m9, m10 = st.columns(4)
    with m7:
        st.metric("Total PnL", f"${metrics.get('total_pnl', 0):,.2f}")
    with m8:
        st.metric("Avg RR Achieved", f"{metrics.get('avg_rr_achieved', 0):.2f}")
    with m9:
        st.metric("Best Trade", f"${metrics.get('best_trade', 0):,.2f}")
    with m10:
        st.metric("Worst Trade", f"${metrics.get('worst_trade', 0):,.2f}")

    # ── Equity Curve ──
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        render_equity_curve(result.equity_curve, params["capital"])

    # ── Strategy & Setup Success Breakdown ──
    setup_data = metrics.get("setup_breakdown", {})
    trigger_data = metrics.get("trigger_breakdown", {})

    if setup_data or trigger_data:
        st.markdown("<div class='frost-panel'><h3>🎯 Success Rate by Setup (S&R vs. FVG vs. Reversals)</h3></div>", unsafe_allow_html=True)
        col_s, col_t = st.columns(2)

        with col_s:
            st.markdown("#### 📍 Location Setup (S&R vs. FVG)")
            if setup_data:
                s_rows = []
                for stype, d in setup_data.items():
                    s_rows.append({
                        "Setup": stype,
                        "Trades": d.get("trades", 0),
                        "Win Rate": f"{d.get('win_rate_pct', 0):.1f}%",
                        "Wins / Losses": f"{d.get('wins', 0)}W / {d.get('losses', 0)}L",
                        "Net PnL": f"${d.get('pnl', 0):+,.2f}",
                    })
                st.dataframe(pd.DataFrame(s_rows), use_container_width=True)

        with col_t:
            st.markdown("#### ⚡ Trigger Mode (Reversal vs. Breakout)")
            if trigger_data:
                t_rows = []
                for tmode, d in trigger_data.items():
                    t_rows.append({
                        "Trigger Mode": tmode,
                        "Trades": d.get("trades", 0),
                        "Win Rate": f"{d.get('win_rate_pct', 0):.1f}%",
                        "Wins / Losses": f"{d.get('wins', 0)}W / {d.get('losses', 0)}L",
                        "Net PnL": f"${d.get('pnl', 0):+,.2f}",
                    })
                st.dataframe(pd.DataFrame(t_rows), use_container_width=True)

    # ── Trade Log ──
    with st.expander(f"📋 Trade Log ({len(result.trades)} trades)", expanded=True):
        if result.trades:
            # Trade outcome counts
            tp_count = sum(1 for t in result.trades if t.status == "CLOSED_TP")
            sl_count = sum(1 for t in result.trades if t.status == "CLOSED_SL")
            be_count = sum(1 for t in result.trades if t.status == "CLOSED_BE")

            st.markdown(
                f"""
                <div style="display: flex; gap: 20px; margin-bottom: 14px; font-size: 0.92rem; background: rgba(20,22,26,0.5); padding: 10px 14px; border-radius: 8px; border: 1px solid #272A30;">
                    <span style="color: #22C55E; font-weight: 700;">🎯 Hit Take Profit (TP): {tp_count}</span>
                    <span style="color: #EF4444; font-weight: 700;">🛑 Hit Stop Loss (SL): {sl_count}</span>
                    <span style="color: #38BDF8; font-weight: 700;">🛡️ Saved at Breakeven (BE): {be_count}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            trade_data = []
            for t in result.trades:
                outcome_label = "⏳ Open"
                if t.status == "CLOSED_TP":
                    outcome_label = "🎯 HIT TP"
                elif t.status == "CLOSED_SL":
                    outcome_label = "🛑 HIT SL"
                elif t.status == "CLOSED_BE":
                    outcome_label = "🛡️ HIT BE"

                trade_data.append({
                    "Outcome": outcome_label,
                    "Setup Trigger": f"{getattr(t, 'setup_type', 'S&R Bounce')} · {getattr(t, 'trigger_mode', 'Reversal')}",
                    "Direction": "🟢 LONG" if t.direction == "LONG" else "🔴 SHORT",
                    "Entry Time": str(t.entry_time)[:16],
                    "Entry": f"{t.entry_price:.4f}" if t.entry_price < 100 else f"{t.entry_price:.2f}",
                    "SL": f"{t.stop_loss:.4f}" if t.stop_loss < 100 else f"{t.stop_loss:.2f}",
                    "TP": f"{t.take_profit:.4f}" if t.take_profit < 100 else f"{t.take_profit:.2f}",
                    "Exit": f"{t.exit_price:.4f}" if t.exit_price and t.exit_price < 100 else (f"{t.exit_price:.2f}" if t.exit_price else "OPEN"),
                    "PnL ($)": f"${t.pnl:+,.2f}",
                    "RR Achieved": f"{t.rr_achieved:.2f}R",
                    "Trailed to BE?": "✅ Trailed" if t.be_triggered else "No",
                    "Bars Held": t.bars_held,
                })
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
        else:
            st.info("No trades were generated by the backtest.")

    # ── Full Metrics JSON ──
    with st.expander("🔢 Full Metrics"):
        st.json(metrics)


# ─────────────────────────────────────────────────────────────
# TAB: FUNDAMENTAL ANALYSIS
# ─────────────────────────────────────────────────────────────

def tab_fundamentals(params: dict):
    """Fundamental analysis tab for equities."""
    st.markdown(
        "<div class='frost-panel'><h3>📑 Fundamental Analysis</h3>"
        "Financial health scoring, Piotroski F-Score, and balance sheet analysis for equities.</div>",
        unsafe_allow_html=True,
    )

    # Stock selection
    stock_tickers = ["NVDA", "TSLA", "AAPL", "MSFT"]
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_stocks = st.multiselect("Select Stocks", stock_tickers, default=["NVDA", "AAPL"], key="fund_stocks")
    with col2:
        custom_ticker = st.text_input("Custom Ticker", key="custom_ticker")
        if custom_ticker:
            selected_stocks.append(custom_ticker.upper())

    if not selected_stocks:
        st.info("Select at least one stock to analyze.")
        return

    if st.button("📊 Analyze Fundamentals", use_container_width=True, key="analyze_fund"):
        for ticker in selected_stocks:
            with st.spinner(f"Analyzing {ticker}..."):
                try:
                    card = get_fundamental_card(ticker)

                    st.markdown(f"""
                    <div class='frost-panel'>
                        <h3>{'🟢' if card.get('health_rating') == 'STRONG' else '🟡' if card.get('health_rating') == 'MODERATE' else '🔴'} {card.get('company_name', ticker)} ({ticker})</h3>
                        <p style="color: #6B7280; margin-bottom: 12px;">{card.get('sector', 'N/A')} · {card.get('industry', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    fc1, fc2, fc3, fc4 = st.columns(4)
                    metrics = card.get("key_metrics", {})
                    with fc1:
                        st.metric("Market Cap", metrics.get("market_cap", metrics.get("Market Cap", "N/A")))
                        st.metric("P/E (TTM)", metrics.get("pe_trailing", metrics.get("P/E (Trailing)", "N/A")))
                        st.metric("P/E (Fwd)", metrics.get("pe_forward", metrics.get("P/E (Forward)", "N/A")))
                    with fc2:
                        st.metric("EPS", metrics.get("eps_trailing", metrics.get("EPS (Trailing)", "N/A")))
                        st.metric("EPS Growth", metrics.get("eps_growth", metrics.get("EPS Growth (YoY)", "N/A")))
                        st.metric("Revenue Growth", metrics.get("revenue_growth", metrics.get("Revenue Growth (YoY)", "N/A")))
                    with fc3:
                        st.metric("FCF", metrics.get("free_cash_flow", metrics.get("Free Cash Flow", "N/A")))
                        st.metric("FCF Yield", metrics.get("fcf_yield", metrics.get("FCF Yield", "N/A")))
                        st.metric("D/E Ratio", metrics.get("debt_to_equity", metrics.get("Debt-to-Equity", "N/A")))
                    with fc4:
                        st.metric("ROE", metrics.get("roe", metrics.get("ROE", "N/A")))
                        st.metric("ROIC", metrics.get("roic", metrics.get("ROIC", "N/A")))
                        st.metric("Gross Margin", metrics.get("gross_margin", metrics.get("Gross Margin", "N/A")))

                    # Piotroski F-Score
                    fscore = card.get("piotroski", {})
                    total = fscore.get("total_score", 0)
                    interp = fscore.get("interpretation", "N/A")
                    score_color = "#22C55E" if total >= 7 else "#E2F952" if total >= 4 else "#EF4444"

                    st.markdown(f"""
                    <div class='frost-panel'>
                        <h3>Piotroski F-Score: <span style="color:{score_color}">{total}/9</span> — {interp}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    breakdown = fscore.get("breakdown", {})
                    if breakdown:
                        bd_cols = st.columns(3)
                        items = list(breakdown.items())
                        for idx, (key, val) in enumerate(items):
                            with bd_cols[idx % 3]:
                                icon = "✅" if val else "❌"
                                label = key.replace("_", " ").title()
                                st.markdown(f"{icon} {label}")

                    st.markdown("---")

                except Exception as e:
                    st.error(f"Error analyzing {ticker}: {e}")

    # ── Comparison Table ──
    if len(selected_stocks) > 1:
        with st.expander("📊 Side-by-Side Comparison"):
            if st.button("Generate Comparison", key="gen_compare"):
                with st.spinner("Comparing..."):
                    try:
                        comp_df = compare_fundamentals(selected_stocks)
                        st.dataframe(comp_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Comparison error: {e}")


# ─────────────────────────────────────────────────────────────
# TAB: ECONOMIC CALENDAR
# ─────────────────────────────────────────────────────────────

def tab_calendar():
    """Economic calendar tab."""
    st.markdown(
        "<div class='frost-panel'><h3>📅 Economic Calendar & News Shield</h3>"
        "High-impact events create a 30-minute no-trade blackout zone.</div>",
        unsafe_allow_html=True,
    )

    cal = st.session_state.calendar_df

    if cal is None or cal.empty:
        st.info("📅 Click **Load Calendar** in the sidebar to fetch economic events.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        impact_filter = st.multiselect(
            "Impact Level", ["High", "Medium", "Low"], default=["High"], key="impact_filt"
        )
    with col2:
        if "currency" in cal.columns:
            currencies = sorted(cal["currency"].dropna().unique().tolist())
            curr_filter = st.multiselect("Currency", currencies, default=currencies[:5] if len(currencies) > 5 else currencies, key="curr_filt")
        else:
            curr_filter = []

    filtered = cal.copy()
    if impact_filter and "impact" in filtered.columns:
        filtered = filtered[filtered["impact"].str.capitalize().isin(impact_filter)]
    if curr_filter and "currency" in filtered.columns:
        filtered = filtered[filtered["currency"].isin(curr_filter)]

    # ── High Impact Events ──
    if "impact" in filtered.columns:
        high_events = filtered[filtered["impact"].str.lower() == "high"]
        if not high_events.empty:
            st.markdown("<div class='frost-panel'><h3>🔴 High-Impact Events (Red Folder)</h3></div>", unsafe_allow_html=True)
            for _, ev in high_events.iterrows():
                dt = ev.get("datetime", ev.get("date", ""))
                name = ev.get("event_name", "Unknown")
                curr = ev.get("currency", "")
                forecast = ev.get("forecast", "")
                previous = ev.get("previous", "")
                st.markdown(
                    f"<div class='frost-panel' style='border-left: 3px solid #EF4444;'>"
                    f"<strong style='color:#EF4444;'>{dt}</strong> · {curr} · <strong>{name}</strong><br>"
                    f"<span style='color:#6B7280;'>Forecast: {forecast} | Previous: {previous}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Full Calendar Table ──
    with st.expander(f"📋 Full Calendar ({len(filtered)} events)"):
        st.dataframe(filtered, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB: AI ADVISOR
# ─────────────────────────────────────────────────────────────

def tab_ai_advisor(params: dict):
    """AI advisor chat interface."""
    st.markdown(
        "<div class='frost-panel'><h3>🤖 NEXUS-QUANT AI Advisor</h3>"
        "AI Engine: <strong style='color:#E2F952;'>Google Gemini 3.8 Flash</strong> · Vision & Multi-Screenshot Analysis"
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick action buttons
    col1, col2, col3, col4 = st.columns(4)

    symbol = params["symbol"]

    with col1:
        analyze_setup_btn = st.button("🎯 Analyze Setup", use_container_width=True, key="ai_setup")
    with col2:
        critique_bt_btn = st.button("📊 Critique Backtest", use_container_width=True, key="ai_bt")
    with col3:
        fund_opinion_btn = st.button("📑 Fund. Opinion", use_container_width=True, key="ai_fund")
    with col4:
        regime_btn = st.button("🌊 Market Regime", use_container_width=True, key="ai_regime")

    # Show warning banner if API key is missing
    current_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    if not current_key:
        st.warning(
            "⚠️ **Gemini API Key is missing!** "
            "Please paste your free Google Gemini API key into the **'Gemini API Key'** box in the left sidebar."
        )

    # Initialize advisor
    try:
        advisor = AIAdvisor(
            provider="gemini",
            model="gemini-3.8-flash",
            api_key=current_key,
        )
    except Exception as e:
        st.error(f"AI Advisor initialization error: {e}")
        advisor = None

    # Handle quick actions
    if advisor:
        if analyze_setup_btn:
            if symbol not in st.session_state.market_data:
                st.warning(f"⚠️ Please click **Fetch** in the sidebar to load data for {symbol} first!")
            elif not current_key:
                st.error("❌ Please paste your Gemini API key in the sidebar first!")
            else:
                df = st.session_state.market_data[symbol]
                latest = df.iloc[-1]
                setup = {
                    "symbol": symbol,
                    "price": float(latest["Close"]),
                    "trend_bias": str(latest.get("trend_bias", "N/A")),
                    "rsi": float(latest.get("RSI", 0)),
                    "rsi_mode": str(latest.get("rsi_mode", "N/A")),
                    "support": float(latest.get("support_level", 0)) if pd.notna(latest.get("support_level")) else None,
                    "resistance": float(latest.get("resistance_level", 0)) if pd.notna(latest.get("resistance_level")) else None,
                    "ma_fast": float(latest.get("MA_fast", 0)) if pd.notna(latest.get("MA_fast")) else None,
                    "ma_slow": float(latest.get("MA_slow", 0)) if pd.notna(latest.get("MA_slow")) else None,
                    "active_bull_fvg": bool(pd.notna(latest.get("active_bull_fvg_top"))),
                    "active_bear_fvg": bool(pd.notna(latest.get("active_bear_fvg_top"))),
                }
                with st.spinner("Analyzing setup..."):
                    response = advisor.analyze_setup(setup)
                    st.session_state.ai_messages.append({"role": "assistant", "content": response})
                    st.rerun()

        if critique_bt_btn:
            if not st.session_state.backtest_result:
                st.warning("⚠️ Please run a backtest first in the **⚡ Backtester** tab before requesting an AI critique!")
            elif not current_key:
                st.error("❌ Please paste your Gemini API key in the sidebar first!")
            else:
                with st.spinner("Critiquing backtest..."):
                    response = advisor.analyze_backtest(st.session_state.backtest_result.metrics)
                    st.session_state.ai_messages.append({"role": "assistant", "content": response})
                    st.rerun()

        if fund_opinion_btn:
            cat, _, _ = get_symbol_info(symbol)
            if cat != "Stocks":
                st.warning(f"Fundamental analysis is only available for stocks, not {cat}.")
            elif not current_key:
                st.error("❌ Please paste your Gemini API key in the sidebar first!")
            else:
                with st.spinner("Analyzing fundamentals..."):
                    try:
                        card = get_fundamental_card(symbol)
                        response = advisor.analyze_fundamentals(card)
                        st.session_state.ai_messages.append({"role": "assistant", "content": response})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        if regime_btn:
            if symbol not in st.session_state.market_data:
                st.warning(f"⚠️ Please fetch data for {symbol} first!")
            elif not current_key:
                st.error("❌ Please paste your Gemini API key in the sidebar first!")
            else:
                df = st.session_state.market_data[symbol]
                recent = df.tail(100)
                regime_data = {
                    "symbol": symbol,
                    "price_range_pct": float((recent["High"].max() - recent["Low"].min()) / recent["Close"].mean() * 100),
                    "avg_volume": float(recent["Volume"].mean()) if "Volume" in recent.columns else 0,
                    "trend_bias": str(df.iloc[-1].get("trend_bias", "N/A")),
                    "rsi": float(df.iloc[-1].get("RSI", 0)),
                    "volatility_20": float(recent["Close"].pct_change().std() * np.sqrt(252) * 100),
                    "bars_analyzed": len(recent),
                }
                with st.spinner("Analyzing market regime..."):
                    response = advisor.chat(
                        f"Analyze the current market regime for {symbol}. "
                        f"Is this a trending, ranging, or volatile market? What strategy adjustments do you recommend?",
                        context=regime_data,
                    )
                    st.session_state.ai_messages.append({"role": "assistant", "content": response})
                    st.rerun()

    # ── Chat Display ──
    st.markdown("---")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.ai_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(
                    f"<div class='frost-panel' style='border-left: 3px solid #38BDF8;'>"
                    f"<strong style='color:#38BDF8;'>You</strong><br>{content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='frost-panel' style='border-left: 3px solid #E2F952;'>"
                    f"<strong style='color:#E2F952;'>NEXUS-QUANT AI</strong><br>{content}</div>",
                    unsafe_allow_html=True,
                )

    # ── Screenshots & Trade Library (Local Folder & Multi-Upload) ──
    pil_images = []
    with st.expander("📁 Trade Screenshots Library (Folder & Multi-Upload)", expanded=False):
        mode = st.radio("Source Mode", ["📂 Local Folder (300+ Trade Screenshots)", "📤 Manual File Upload"], horizontal=True, key="sc_mode")

        if mode == "📂 Local Folder (300+ Trade Screenshots)":
            st.markdown("<p style='color: #6B7280; font-size: 0.85rem;'>Point to any folder on your laptop containing your trade screenshots. NEXUS-QUANT AI will scan and inspect them directly.</p>", unsafe_allow_html=True)
            folder_path = st.text_input(
                "Enter Folder Path on Your PC",
                value=st.session_state.get("trade_folder_path", ""),
                placeholder="e.g. C:\\Users\\Abdullah\\Pictures\\Trades or C:\\Users\\Abdullah\\Downloads\\trades",
                key="folder_path_input"
            )
            st.session_state["trade_folder_path"] = folder_path

            if folder_path and Path(folder_path).is_dir():
                valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
                folder_images = [p for p in Path(folder_path).iterdir() if p.suffix.lower() in valid_exts]
                folder_images.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                st.success(f"✓ Found {len(folder_images)} trade screenshots in folder!")

                if folder_images:
                    col_sel, col_action = st.columns([2, 1])
                    with col_sel:
                        selected_idx = st.slider("Browse Screenshot Index", 1, len(folder_images), 1, key="img_slider")
                        chosen_path = folder_images[selected_idx - 1]
                        st.caption(f"Viewing: `{chosen_path.name}` ({selected_idx} of {len(folder_images)})")
                        from PIL import Image
                        try:
                            preview_img = Image.open(chosen_path)
                            pil_images.append(preview_img)
                            st.image(preview_img, caption=chosen_path.name, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")

                    with col_action:
                        st.markdown("<div style='padding-top: 30px;'></div>", unsafe_allow_html=True)
                        if st.button("🔍 AI Review This Trade", use_container_width=True, key="review_sc_btn"):
                            if advisor:
                                with st.spinner(f"Analyzing {chosen_path.name}..."):
                                    prompt = (
                                        f"Please analyze this trade screenshot ({chosen_path.name}). "
                                        "1. Identify the setup (Support/Resistance, Trend, Fair Value Gaps, or Reversal candle). "
                                        "2. Did this trade execute properly based on risk management? "
                                        "3. Critique the entry, Stop Loss, and Take Profit levels shown in this picture."
                                    )
                                    resp = advisor.chat(prompt, images=[preview_img])
                                    st.session_state.ai_messages.append({"role": "user", "content": f"Uploaded trade screenshot: `{chosen_path.name}`"})
                                    st.session_state.ai_messages.append({"role": "assistant", "content": resp})
                                    st.rerun()

            elif folder_path:
                st.warning(f"Folder not found: `{folder_path}`. Please verify the exact folder path.")

        else:
            uploaded_files = st.file_uploader(
                "Upload chart screenshots (PNG, JPG)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="advisor_image_uploader",
                help="Gemini 2.0 will directly read and analyze the charts, candles, and data in your images.",
            )
            if uploaded_files:
                from PIL import Image
                cols = st.columns(min(len(uploaded_files), 4))
                for i, f in enumerate(uploaded_files):
                    try:
                        img = Image.open(f)
                        pil_images.append(img)
                        with cols[i % 4]:
                            st.image(img, caption=f"Screenshot {i+1}", use_container_width=True)
                    except Exception as img_err:
                        st.warning(f"Could not load image {f.name}: {img_err}")

    # ── Chat Input ──
    user_input = st.chat_input("Ask NEXUS-QUANT AI anything about the markets (or your uploaded screenshots)...", key="ai_chat_input")
    if user_input and advisor:
        st.session_state.ai_messages.append({"role": "user", "content": user_input})

        # Build context from current state
        context = {"symbol": symbol}
        if symbol in st.session_state.market_data:
            latest = st.session_state.market_data[symbol].iloc[-1]
            context.update({
                "price": float(latest["Close"]),
                "trend_bias": str(latest.get("trend_bias", "N/A")),
                "rsi": float(latest.get("RSI", 0)),
            })
        if st.session_state.backtest_result:
            context["backtest_metrics"] = st.session_state.backtest_result.metrics

        with st.spinner("Analyzing text & screenshots..."):
            response = advisor.chat(user_input, context=context, images=pil_images if pil_images else None)
            st.session_state.ai_messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.ai_messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    # Header
    st.markdown(
        '<div class="nexus-header">'
        '<h1>⚡ NEXUS-QUANT</h1>'
        '<p>Modular Quantitative Backtesting & Fundamental Analysis Platform · Local-First · Zero-Cost AI</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    params = render_sidebar()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Market Analysis",
        "⚡ Backtester",
        "📑 Fundamentals",
        "📅 Calendar",
        "🤖 AI Advisor",
    ])

    with tab1:
        tab_market_analysis(params)

    with tab2:
        tab_backtester(params)

    with tab3:
        tab_fundamentals(params)

    with tab4:
        tab_calendar()

    with tab5:
        tab_ai_advisor(params)

    # ── Bottom Right Branding ──
    st.markdown(
        """
        <div style="
            text-align: right;
            padding: 35px 15px 15px 15px;
            color: #6B7280;
            font-size: 0.85rem;
            letter-spacing: 0.03em;
        ">
            ⚡ Made by <strong style="color: #E2F952;">Abdullah (Ronin)</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
