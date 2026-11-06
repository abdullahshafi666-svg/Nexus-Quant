"""
NEXUS-QUANT Technical Indicators
=================================
Pure pandas / numpy indicator calculations.  Every public function takes an
OHLCV ``DataFrame`` (columns: Open, High, Low, Close, Volume; DatetimeIndex)
and returns the **same** DataFrame with new columns appended.

Designed for Python 3.10+ on Windows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ``ValueError`` if *df* is missing any of *cols*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _require_min_rows(df: pd.DataFrame, n: int, context: str = "") -> None:
    """Raise ``ValueError`` if *df* has fewer than *n* rows."""
    if len(df) < n:
        raise ValueError(
            f"Insufficient data for {context}: need ≥{n} rows, got {len(df)}."
        )


# ===================================================================== #
# 1. Moving Averages
# ===================================================================== #

def add_moving_averages(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
) -> pd.DataFrame:
    """Add ``MA_fast`` and ``MA_slow`` SMA columns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    fast : int
        Window for the fast SMA (default 20).
    slow : int
        Window for the slow SMA (default 50).

    Returns
    -------
    pd.DataFrame
        *df* with ``MA_fast`` and ``MA_slow`` columns added.
    """
    _require_columns(df, ["Close"])
    df["MA_fast"] = df["Close"].rolling(window=fast, min_periods=fast).mean()
    df["MA_slow"] = df["Close"].rolling(window=slow, min_periods=slow).mean()
    return df


# ===================================================================== #
# 2. Trend Bias
# ===================================================================== #

def determine_trend_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``trend_bias`` column (``LONG`` / ``SHORT`` / ``NEUTRAL``).

    Requires ``MA_fast`` and ``MA_slow`` columns (call
    :func:`add_moving_averages` first).
    """
    _require_columns(df, ["Close", "MA_fast", "MA_slow"])

    conditions = [
        (df["Close"] > df["MA_fast"]) & (df["Close"] > df["MA_slow"]),
        (df["Close"] < df["MA_fast"]) & (df["Close"] < df["MA_slow"]),
    ]
    choices = ["LONG", "SHORT"]
    df["trend_bias"] = np.select(conditions, choices, default="NEUTRAL")
    return df


# ===================================================================== #
# 3. Pivot-based Support / Resistance
# ===================================================================== #

def calculate_pivot_sr(
    df: pd.DataFrame,
    left_bars: int = 5,
    right_bars: int = 5,
) -> pd.DataFrame:
    """Detect pivot highs and lows using a rolling-window approach.

    A **Pivot High** at bar *i* requires that ``High[i]`` is strictly greater
    than the highs of the *left_bars* bars to its left **and** the
    *right_bars* bars to its right.

    A **Pivot Low** at bar *i* requires that ``Low[i]`` is strictly less than
    the lows of the *left_bars* bars to its left **and** the *right_bars*
    bars to its right.

    Adds columns
    ~~~~~~~~~~~~~
    * ``pivot_high`` — the High value at pivot highs, else ``NaN``.
    * ``pivot_low`` — the Low value at pivot lows, else ``NaN``.
    * ``resistance_level`` — forward-filled most recent ``pivot_high``.
    * ``support_level`` — forward-filled most recent ``pivot_low``.
    """
    _require_columns(df, ["High", "Low"])

    n = len(df)
    window = left_bars + right_bars + 1

    if n < window:
        df["pivot_high"] = np.nan
        df["pivot_low"] = np.nan
        df["resistance_level"] = np.nan
        df["support_level"] = np.nan
        return df

    highs: np.ndarray = df["High"].to_numpy(dtype=np.float64)
    lows: np.ndarray = df["Low"].to_numpy(dtype=np.float64)

    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)

    for i in range(left_bars, n - right_bars):
        # --- pivot high ---------------------------------------------------
        left_max = highs[i - left_bars: i].max()
        right_max = highs[i + 1: i + 1 + right_bars].max()
        if highs[i] > left_max and highs[i] > right_max:
            pivot_high[i] = highs[i]

        # --- pivot low ----------------------------------------------------
        left_min = lows[i - left_bars: i].min()
        right_min = lows[i + 1: i + 1 + right_bars].min()
        if lows[i] < left_min and lows[i] < right_min:
            pivot_low[i] = lows[i]

    df["pivot_high"] = pivot_high
    df["pivot_low"] = pivot_low
    df["resistance_level"] = df["pivot_high"].ffill()
    df["support_level"] = df["pivot_low"].ffill()
    return df


# ===================================================================== #
# 4. Fair Value Gap (FVG)
# ===================================================================== #

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Fair Value Gaps and track active (unfilled) gaps.

    Definitions (using 3-candle window centred at bar *i*):

    * **Bullish FVG** at *i*: ``High[i-2] < Low[i]`` — gap-up.
    * **Bearish FVG** at *i*: ``Low[i-2] > High[i]`` — gap-down.

    An FVG is **filled** (closed) when price trades through the gap.

    Columns added
    ~~~~~~~~~~~~~~
    ``fvg_type``, ``fvg_top``, ``fvg_bottom``,
    ``active_bull_fvg_top``, ``active_bull_fvg_bottom``,
    ``active_bear_fvg_top``, ``active_bear_fvg_bottom``.
    """
    _require_columns(df, ["Open", "High", "Low", "Close"])

    n = len(df)
    fvg_type: list[Optional[str]] = [None] * n
    fvg_top = np.full(n, np.nan)
    fvg_bottom = np.full(n, np.nan)

    # Active FVG tracking columns
    act_bull_top = np.full(n, np.nan)
    act_bull_bot = np.full(n, np.nan)
    act_bear_top = np.full(n, np.nan)
    act_bear_bot = np.full(n, np.nan)

    highs = df["High"].to_numpy(dtype=np.float64)
    lows = df["Low"].to_numpy(dtype=np.float64)

    # Running lists of active gaps: (top, bottom)
    active_bull_gaps: list[list[float]] = []
    active_bear_gaps: list[list[float]] = []

    for i in range(n):
        # --- detect new FVGs (need at least 3 bars) ----------------------
        if i >= 2:
            if highs[i - 2] < lows[i]:
                # Bullish FVG
                fvg_type[i] = "BULLISH_FVG"
                fvg_top[i] = lows[i]       # top of gap = low of current candle
                fvg_bottom[i] = highs[i - 2]  # bottom of gap = high of candle i-2
                active_bull_gaps.append([lows[i], highs[i - 2]])

            elif lows[i - 2] > highs[i]:
                # Bearish FVG
                fvg_type[i] = "BEARISH_FVG"
                fvg_top[i] = lows[i - 2]    # top of gap = low of candle i-2
                fvg_bottom[i] = highs[i]     # bottom of gap = high of current candle
                active_bear_gaps.append([lows[i - 2], highs[i]])

        # --- fill check: remove gaps fully traded through ----------------
        # Bullish gap filled when price (low) goes below the gap bottom
        active_bull_gaps = [
            g for g in active_bull_gaps if lows[i] > g[1]
        ]
        # Bearish gap filled when price (high) goes above the gap top
        active_bear_gaps = [
            g for g in active_bear_gaps if highs[i] < g[0]
        ]

        # --- record most recent active gap of each type ------------------
        if active_bull_gaps:
            act_bull_top[i] = active_bull_gaps[-1][0]
            act_bull_bot[i] = active_bull_gaps[-1][1]

        if active_bear_gaps:
            act_bear_top[i] = active_bear_gaps[-1][0]
            act_bear_bot[i] = active_bear_gaps[-1][1]

    df["fvg_type"] = fvg_type
    df["fvg_top"] = fvg_top
    df["fvg_bottom"] = fvg_bottom
    df["active_bull_fvg_top"] = act_bull_top
    df["active_bull_fvg_bottom"] = act_bull_bot
    df["active_bear_fvg_top"] = act_bear_top
    df["active_bear_fvg_bottom"] = act_bear_bot
    return df


# ===================================================================== #
# 5. Inversion FVG (IFVG)
# ===================================================================== #

def detect_ifvg(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Inversion Fair Value Gaps.

    Requires :func:`detect_fvg` to have been called first so that
    ``fvg_type``, ``fvg_top``, ``fvg_bottom`` columns exist.

    An IFVG forms when a previously detected FVG is **fully breached** in
    the opposite direction:

    * A *Bullish FVG* breached to the downside (Close < fvg_bottom) becomes
      a **BEARISH_IFVG** (now acts as resistance).
    * A *Bearish FVG* breached to the upside (Close > fvg_top) becomes a
      **BULLISH_IFVG** (now acts as support).

    Columns added: ``ifvg_type``, ``ifvg_top``, ``ifvg_bottom``.
    """
    _require_columns(df, ["Close", "fvg_type", "fvg_top", "fvg_bottom"])

    n = len(df)
    ifvg_type: list[Optional[str]] = [None] * n
    ifvg_top = np.full(n, np.nan)
    ifvg_bottom = np.full(n, np.nan)

    closes = df["Close"].to_numpy(dtype=np.float64)

    # Track all detected FVGs: list of (bar_index, type, top, bottom, inverted)
    fvg_records: list[list] = []

    for i in range(n):
        ft = df["fvg_type"].iat[i]
        if ft in ("BULLISH_FVG", "BEARISH_FVG"):
            fvg_records.append([i, ft, fvg_top_val := df["fvg_top"].iat[i],
                                fvg_bot_val := df["fvg_bottom"].iat[i], False])

        # Check all not-yet-inverted FVGs
        for rec in fvg_records:
            if rec[4]:  # already inverted
                continue
            _, ftype, ftop, fbot, _ = rec
            if ftype == "BULLISH_FVG" and closes[i] < fbot:
                # Breached downside → BEARISH IFVG
                ifvg_type[i] = "BEARISH_IFVG"
                ifvg_top[i] = ftop
                ifvg_bottom[i] = fbot
                rec[4] = True
            elif ftype == "BEARISH_FVG" and closes[i] > ftop:
                # Breached upside → BULLISH IFVG
                ifvg_type[i] = "BULLISH_IFVG"
                ifvg_top[i] = ftop
                ifvg_bottom[i] = fbot
                rec[4] = True

    df["ifvg_type"] = ifvg_type
    df["ifvg_top"] = ifvg_top
    df["ifvg_bottom"] = ifvg_bottom
    return df


# ===================================================================== #
# 6. RSI (Wilder's Smoothing)
# ===================================================================== #

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ``RSI`` column using Wilder's exponential smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.DataFrame
        *df* with ``RSI`` column added.
    """
    _require_columns(df, ["Close"])

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing ≡ EMA with alpha = 1/period
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    return df


# ===================================================================== #
# 7. RSI Mode Classification
# ===================================================================== #

def classify_rsi_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``rsi_mode`` column based on RSI thresholds.

    Requires :func:`calculate_rsi` to have been called first.

    Classification
    ~~~~~~~~~~~~~~
    * ``REVERSAL_LONG`` — RSI < 30
    * ``REVERSAL_SHORT`` — RSI > 70
    * ``BREAKOUT_OK`` — 45 ≤ RSI ≤ 60
    * ``NEUTRAL`` — everything else
    """
    _require_columns(df, ["RSI"])

    conditions = [
        df["RSI"] < 30,
        df["RSI"] > 70,
        (df["RSI"] >= 45) & (df["RSI"] <= 60),
    ]
    choices = ["REVERSAL_LONG", "REVERSAL_SHORT", "BREAKOUT_OK"]
    df["rsi_mode"] = np.select(conditions, choices, default="NEUTRAL")
    return df


# ===================================================================== #
# 8. Master convenience function
# ===================================================================== #

def add_all_indicators(
    df: pd.DataFrame,
    fast_ma: int = 20,
    slow_ma: int = 50,
    rsi_period: int = 14,
    sr_left: int = 5,
    sr_right: int = 5,
) -> pd.DataFrame:
    """Apply **all** indicators in the correct order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame.
    fast_ma, slow_ma : int
        SMA windows.
    rsi_period : int
        RSI look-back.
    sr_left, sr_right : int
        Pivot-point look-back / look-ahead bars.

    Returns
    -------
    pd.DataFrame
        Fully enriched DataFrame with every indicator column.
    """
    _require_columns(df, ["Open", "High", "Low", "Close", "Volume"])

    df = add_moving_averages(df, fast=fast_ma, slow=slow_ma)
    df = determine_trend_bias(df)
    df = calculate_pivot_sr(df, left_bars=sr_left, right_bars=sr_right)
    df = detect_fvg(df)
    df = detect_ifvg(df)
    df = calculate_rsi(df, period=rsi_period)
    df = classify_rsi_mode(df)

    return df


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick self-test with synthetic data
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    _open = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100, 10_000, size=n).astype(float)

    test_df = pd.DataFrame(
        {"Open": _open, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )

    enriched = add_all_indicators(test_df)
    print(enriched.tail(10).to_string())
    print(f"\nColumns ({len(enriched.columns)}): {list(enriched.columns)}")
