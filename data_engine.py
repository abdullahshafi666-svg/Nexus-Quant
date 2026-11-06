"""
NEXUS-QUANT Data Engine
=======================
Robust market data pipeline with local parquet caching and economic calendar
integration.  Designed for Python 3.10+ on Windows.

Modules
-------
- ASSET_UNIVERSE : canonical symbol registry
- download_data  : single-symbol fetcher with 4-hour cache
- load_all_universe : bulk downloader
- fetch_economic_calendar / get_high_impact_events / is_news_blackout :
  economic event awareness layer
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR: Path = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_MAX_AGE_SECONDS: int = 4 * 3600          # 4 hours for market data
CALENDAR_CACHE_MAX_AGE: int = 12 * 3600         # 12 hours for calendar

ASSET_UNIVERSE: dict[str, dict[str, Any]] = {
    "Crypto": {
        "symbols": ["BTC-USD", "ETH-USD"],
        "interval": "15m",
        "period": "60d",
    },
    "Forex": {
        "symbols": ["EURUSD=X", "GBPUSD=X", "GBPJPY=X", "EURAUD=X"],
        "interval": "30m",
        "period": "60d",
    },
    "Indices & Commodities": {
        "symbols": ["GC=F", "SI=F", "CL=F", "NQ=F", "ES=F"],
        "interval": "5m",
        "period": "60d",
    },
    "Stocks": {
        "symbols": ["NVDA", "TSLA", "AAPL", "MSFT"],
        "interval": "1d",
        "period": "2y",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_filename(symbol: str, interval: str) -> str:
    """Return a filesystem-safe parquet filename for a symbol/interval pair."""
    safe = re.sub(r'[^A-Za-z0-9_\-]', '_', f"{symbol}_{interval}")
    return f"{safe}.parquet"


def _cache_path(symbol: str, interval: str) -> Path:
    """Full path to the cached parquet file."""
    return CACHE_DIR / _safe_filename(symbol, interval)


def _is_cache_stale(path: Path, max_age_seconds: int) -> bool:
    """Return True if *path* does not exist or is older than *max_age_seconds*."""
    if not path.exists():
        return True
    mtime = path.stat().st_mtime
    return (time.time() - mtime) > max_age_seconds


# ---------------------------------------------------------------------------
# Core data download
# ---------------------------------------------------------------------------


def download_data(
    symbol: str,
    interval: str,
    period: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data for *symbol* via yfinance with local parquet cache.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``"BTC-USD"``, ``"EURUSD=X"``).
    interval : str
        Bar interval (``"1m"`` … ``"1d"``).
    period : str
        Look-back period (``"60d"``, ``"2y"``, etc.).
    force_refresh : bool, optional
        If ``True``, bypass the cache entirely.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``DatetimeIndex`` with columns
        ``Open, High, Low, Close, Volume``.

    Raises
    ------
    ValueError
        If yfinance returns an empty frame.
    """
    cache = _cache_path(symbol, interval)

    # ---- try cache first ------------------------------------------------
    if not force_refresh and not _is_cache_stale(cache, CACHE_MAX_AGE_SECONDS):
        try:
            df = pd.read_parquet(cache)
            if not df.empty:
                return df
        except Exception:
            pass  # corrupt file → re-download

    # ---- download --------------------------------------------------------
    try:
        ticker = yf.Ticker(symbol)
        df: pd.DataFrame = ticker.history(period=period, interval=interval)
    except Exception as exc:
        raise RuntimeError(
            f"yfinance download failed for {symbol} "
            f"(interval={interval}, period={period}): {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"No data returned by yfinance for {symbol} "
            f"(interval={interval}, period={period})."
        )

    # ---- clean -----------------------------------------------------------
    # Keep only the canonical OHLCV columns
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {keep_cols} but missing {missing} for {symbol}."
        )

    df = df[keep_cols].copy()
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Datetime"
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df["Volume"] = df["Volume"].fillna(0).astype(float)

    # ---- persist ---------------------------------------------------------
    try:
        df.to_parquet(cache, engine="pyarrow")
    except Exception:
        # Non-fatal: we still have the frame in memory
        pass

    return df


# ---------------------------------------------------------------------------
# Bulk loader
# ---------------------------------------------------------------------------


def load_all_universe() -> dict[str, pd.DataFrame]:
    """Download every symbol declared in ``ASSET_UNIVERSE``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ``symbol`` → OHLCV ``DataFrame``.  Symbols that fail to
        download are omitted (a warning is printed to stdout).
    """
    result: dict[str, pd.DataFrame] = {}
    for category, meta in ASSET_UNIVERSE.items():
        symbols: list[str] = meta["symbols"]
        interval: str = meta["interval"]
        period: str = meta["period"]
        for sym in symbols:
            try:
                df = download_data(sym, interval, period)
                result[sym] = df
                print(f"[OK]  {sym:<12s} | {interval:>4s} | {len(df):>6,} bars")
            except Exception as exc:
                print(f"[ERR] {sym:<12s} | {exc}")
    return result


# ---------------------------------------------------------------------------
# Economic calendar
# ---------------------------------------------------------------------------

_FF_URL = "https://www.forexfactory.com/calendar"
_FF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Known recurring high-impact USD events (month, weekday-rule or day)
_STATIC_HIGH_IMPACT: list[dict[str, str]] = [
    {"event_name": "FOMC Interest Rate Decision", "currency": "USD", "impact": "high"},
    {"event_name": "Non-Farm Payrolls", "currency": "USD", "impact": "high"},
    {"event_name": "CPI m/m", "currency": "USD", "impact": "high"},
    {"event_name": "Core CPI m/m", "currency": "USD", "impact": "high"},
    {"event_name": "PPI m/m", "currency": "USD", "impact": "high"},
    {"event_name": "GDP q/q", "currency": "USD", "impact": "high"},
    {"event_name": "Retail Sales m/m", "currency": "USD", "impact": "high"},
    {"event_name": "ISM Manufacturing PMI", "currency": "USD", "impact": "high"},
    {"event_name": "ECB Interest Rate Decision", "currency": "EUR", "impact": "high"},
    {"event_name": "BOE Interest Rate Decision", "currency": "GBP", "impact": "high"},
    {"event_name": "BOJ Interest Rate Decision", "currency": "JPY", "impact": "high"},
    {"event_name": "RBA Interest Rate Decision", "currency": "AUD", "impact": "high"},
    {"event_name": "Employment Change (AU)", "currency": "AUD", "impact": "high"},
    {"event_name": "UK CPI y/y", "currency": "GBP", "impact": "high"},
]


def _parse_forexfactory_calendar(html: str) -> pd.DataFrame:
    """Best-effort parse of the Forex Factory calendar HTML.

    The page layout changes frequently so this parser is deliberately lenient:
    it uses regex to extract table rows and cells.
    """
    rows: list[dict[str, Any]] = []

    # FF wraps each event in a <tr> with class containing "calendar__row"
    row_pattern = re.compile(
        r'<tr[^>]*class="[^"]*calendar__row[^"]*"[^>]*>(.*?)</tr>',
        re.DOTALL,
    )
    cell_tag = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
    strip_tags = re.compile(r"<[^>]+>")

    current_date_str: str = ""

    for row_match in row_pattern.finditer(html):
        cells_html = cell_tag.findall(row_match.group(1))
        cells = [strip_tags.sub("", c).strip() for c in cells_html]

        if len(cells) < 7:
            continue

        # Column mapping (approximate — FF can shift columns):
        # 0: date, 1: time, 2: currency, 3: impact (span title), 4: event,
        # 5: actual, 6: forecast, 7: previous

        date_cell = cells[0].strip()
        if date_cell:
            current_date_str = date_cell

        time_cell = cells[1].strip() if len(cells) > 1 else ""
        currency = cells[2].strip() if len(cells) > 2 else ""

        # Impact: look for "High Impact", "Medium Impact", "Low Impact"
        impact_raw = cells_html[3] if len(cells_html) > 3 else "" if len(cells_html) > 3 else ""
        impact = "low"
        if "High Impact" in impact_raw or "high" in impact_raw.lower() or "red" in impact_raw.lower():
            impact = "high"
        elif "Medium Impact" in impact_raw or "medium" in impact_raw.lower() or "orange" in impact_raw.lower():
            impact = "medium"

        event_name = cells[4].strip() if len(cells) > 4 else ""
        actual = cells[5].strip() if len(cells) > 5 else ""
        forecast = cells[6].strip() if len(cells) > 6 else ""
        previous = cells[7].strip() if len(cells) > 7 else ""

        if not event_name:
            continue

        # Try to build a datetime
        dt = _try_parse_ff_datetime(current_date_str, time_cell)

        rows.append(
            {
                "datetime": dt,
                "currency": currency,
                "impact": impact,
                "event_name": event_name,
                "forecast": forecast,
                "previous": previous,
                "actual": actual,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["datetime", "currency", "impact", "event_name",
                 "forecast", "previous", "actual"]
    )


def _try_parse_ff_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Attempt to parse a Forex Factory date/time pair into a ``datetime``."""
    if not date_str:
        return None

    # FF uses formats like "Mon Jun 10" or "Jun 10" (no year)
    now = datetime.now(tz=timezone.utc)
    year = now.year

    for fmt in ("%a %b %d", "%b %d", "%A %b %d"):
        try:
            parsed = datetime.strptime(date_str, fmt).replace(year=year)
            break
        except ValueError:
            parsed = None

    if parsed is None:
        return None

    # Time: "8:30am", "2:00pm", "All Day", "Tentative", or empty
    if time_str and re.match(r"\d{1,2}:\d{2}", time_str):
        for tf in ("%I:%M%p", "%I:%M %p", "%H:%M"):
            try:
                t = datetime.strptime(time_str.lower().replace(" ", ""), tf)
                parsed = parsed.replace(hour=t.hour, minute=t.minute)
                break
            except ValueError:
                continue

    return parsed.replace(tzinfo=timezone.utc)


def _fetch_ff_calendar() -> pd.DataFrame:
    """Attempt to fetch and parse the Forex Factory calendar page."""
    resp = requests.get(_FF_URL, headers=_FF_HEADERS, timeout=15)
    resp.raise_for_status()
    return _parse_forexfactory_calendar(resp.text)


def _fetch_nager_holidays(year: int | None = None) -> pd.DataFrame:
    """Fallback: fetch US public holidays from the free Nager.Date API."""
    if year is None:
        year = datetime.now(tz=timezone.utc).year

    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/US"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame(
            columns=["datetime", "currency", "impact", "event_name",
                     "forecast", "previous", "actual"]
        )

    rows = []
    for item in data:
        dt = datetime.strptime(item["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        rows.append(
            {
                "datetime": dt,
                "currency": "USD",
                "impact": "medium",
                "event_name": f"US Holiday — {item.get('localName', item.get('name', ''))}",
                "forecast": "",
                "previous": "",
                "actual": "",
            }
        )
    return pd.DataFrame(rows)


def _generate_static_calendar() -> pd.DataFrame:
    """Last-resort fallback: produce a static list of known high-impact events
    spread across the current year/month for awareness."""
    now = datetime.now(tz=timezone.utc)
    rows: list[dict[str, Any]] = []

    for month in range(1, 13):
        for entry in _STATIC_HIGH_IMPACT:
            # Place the event on the 15th of each month as a placeholder
            try:
                dt = datetime(now.year, month, 15, 14, 30, tzinfo=timezone.utc)
            except ValueError:
                continue
            rows.append(
                {
                    "datetime": dt,
                    "currency": entry["currency"],
                    "impact": entry["impact"],
                    "event_name": entry["event_name"],
                    "forecast": "",
                    "previous": "",
                    "actual": "",
                }
            )

    return pd.DataFrame(rows)


def fetch_economic_calendar(*, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the economic calendar with three-tier fallback.

    1. Forex Factory HTML scrape
    2. Nager.Date public-holidays API
    3. Static recurring high-impact event list

    Results are cached to ``data_cache/calendar.parquet`` for 12 hours.

    Returns
    -------
    pd.DataFrame
        Columns: ``datetime, currency, impact, event_name, forecast,
        previous, actual``.
    """
    cache = CACHE_DIR / "calendar.parquet"

    if not force_refresh and not _is_cache_stale(cache, CALENDAR_CACHE_MAX_AGE):
        try:
            df = pd.read_parquet(cache)
            if not df.empty:
                return df
        except Exception:
            pass

    # --- tier 1: Forex Factory -------------------------------------------
    df = pd.DataFrame()
    try:
        df = _fetch_ff_calendar()
    except Exception:
        pass

    # --- tier 2: Nager.Date holidays -------------------------------------
    if df.empty:
        try:
            df = _fetch_nager_holidays()
        except Exception:
            pass

    # --- tier 3: static list ---------------------------------------------
    if df.empty:
        df = _generate_static_calendar()

    # Ensure correct column set
    for col in ("datetime", "currency", "impact", "event_name",
                "forecast", "previous", "actual"):
        if col not in df.columns:
            df[col] = np.nan

    # Persist cache
    try:
        df.to_parquet(cache, engine="pyarrow")
    except Exception:
        pass

    return df


# ---------------------------------------------------------------------------
# High-impact event helpers
# ---------------------------------------------------------------------------


def get_high_impact_events(
    calendar_df: pd.DataFrame,
    minutes_buffer: int = 30,
) -> pd.DataFrame:
    """Filter *calendar_df* for high-impact (\"Red\") events.

    Parameters
    ----------
    calendar_df : pd.DataFrame
        As returned by :func:`fetch_economic_calendar`.
    minutes_buffer : int, optional
        Unused here but kept for API symmetry; the buffer is applied in
        :func:`is_news_blackout`.

    Returns
    -------
    pd.DataFrame
        Subset of rows where ``impact == "high"`` and ``datetime`` is not null.
    """
    if calendar_df.empty:
        return calendar_df

    mask = (
        calendar_df["impact"].str.lower().eq("high")
        & calendar_df["datetime"].notna()
    )
    return calendar_df.loc[mask].copy().reset_index(drop=True)


def is_news_blackout(
    current_time: datetime,
    high_impact_events: pd.DataFrame,
    buffer_minutes: int = 30,
) -> bool:
    """Return ``True`` if *current_time* falls within ±*buffer_minutes* of any
    high-impact event.

    Parameters
    ----------
    current_time : datetime
        The timestamp to check (should be timezone-aware UTC).
    high_impact_events : pd.DataFrame
        DataFrame with a ``datetime`` column of high-impact events.
    buffer_minutes : int, optional
        Guard window in minutes around each event.  Default 30.
    """
    if high_impact_events.empty:
        return False

    buf = timedelta(minutes=buffer_minutes)

    for evt_time in high_impact_events["datetime"]:
        if evt_time is None or pd.isna(evt_time):
            continue
        evt_dt: datetime = pd.Timestamp(evt_time).to_pydatetime()
        # Make both tz-aware for comparison
        if evt_dt.tzinfo is None:
            evt_dt = evt_dt.replace(tzinfo=timezone.utc)
        ct = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)

        if (evt_dt - buf) <= ct <= (evt_dt + buf):
            return True

    return False


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS-QUANT Data Engine — full universe download")
    print("=" * 60)
    data = load_all_universe()
    print(f"\nLoaded {len(data)} symbols.")

    print("\nFetching economic calendar …")
    cal = fetch_economic_calendar()
    hi = get_high_impact_events(cal)
    print(f"Calendar rows : {len(cal)}")
    print(f"High-impact   : {len(hi)}")
