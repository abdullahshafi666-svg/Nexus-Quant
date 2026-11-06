"""
NEXUS-QUANT Fundamental Analysis Engine
========================================
Production-grade fundamental analysis module for equities using yfinance.
Provides financial statement retrieval, key metric calculation, Piotroski
F-Score, health-card generation, and multi-ticker comparison.

Requires: yfinance, pandas, numpy
Python 3.10+
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator / denominator, or None on any failure."""
    try:
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _pct_fmt(value: float | None) -> str:
    """Format a ratio as a percentage string, e.g. 0.234 -> '23.4%'."""
    if value is None:
        return "N/A"
    try:
        return f"{value * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _ratio_fmt(value: float | None, decimals: int = 2) -> str | float:
    """Format a ratio as a rounded float or 'N/A'."""
    if value is None:
        return "N/A"
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return "N/A"


def _currency_fmt(value: float | None) -> str:
    """Format a large currency value with B/M/K suffix."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if abs(v) >= 1e12:
            return f"${v / 1e12:.2f}T"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.2f}M"
        if abs(v) >= 1e3:
            return f"${v / 1e3:.2f}K"
        return f"${v:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _extract_item(df: pd.DataFrame, labels: list[str], col_idx: int = 0) -> float | None:
    """
    Extract a numeric item from a yfinance financial DataFrame.

    yfinance returns DataFrames where the *index* contains line-item names
    (e.g. 'Total Revenue') and the *columns* are fiscal-period timestamps.
    ``col_idx=0`` is the most recent period, ``col_idx=1`` is the prior one.
    """
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            try:
                val = df.iloc[df.index.get_loc(label), col_idx]
                if pd.notna(val):
                    return float(val)
            except (IndexError, KeyError, TypeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# 1. Financial Statements
# ---------------------------------------------------------------------------

def get_financial_statements(ticker: str) -> dict[str, Any]:
    """
    Fetch balance sheet, income statement, and cash flow statement for
    *ticker* via yfinance.

    Returns
    -------
    dict
        Keys: ``balance_sheet``, ``income_statement``, ``cash_flow`` (each a
        ``pd.DataFrame``), and ``info`` (``dict``).  On failure the
        DataFrames will be empty and ``info`` will be ``{}``.
    """
    result: dict[str, Any] = {
        "balance_sheet": pd.DataFrame(),
        "income_statement": pd.DataFrame(),
        "cash_flow": pd.DataFrame(),
        "info": {},
    }
    try:
        t = yf.Ticker(ticker)

        try:
            bs = t.balance_sheet
            result["balance_sheet"] = bs if bs is not None and not bs.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Balance sheet fetch failed for %s: %s", ticker, exc)

        try:
            inc = t.income_stmt
            result["income_statement"] = inc if inc is not None and not inc.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Income statement fetch failed for %s: %s", ticker, exc)

        try:
            cf = t.cashflow
            result["cash_flow"] = cf if cf is not None and not cf.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Cash flow fetch failed for %s: %s", ticker, exc)

        try:
            info = t.info
            result["info"] = info if isinstance(info, dict) else {}
        except Exception as exc:
            logger.warning("Info fetch failed for %s: %s", ticker, exc)

    except Exception as exc:
        logger.error("Failed to create Ticker object for %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# 2. Key Metrics
# ---------------------------------------------------------------------------

def calculate_key_metrics(ticker: str) -> dict[str, Any]:
    """
    Calculate and return a dictionary of key fundamental metrics for *ticker*.

    Metrics include market cap, P/E ratios, EPS, growth rates, free cash flow,
    leverage ratios, profitability margins, ROIC, and dividend yield.  Values
    that cannot be computed are set to ``"N/A"``.
    """
    metrics: dict[str, Any] = {
        "Market Cap": "N/A",
        "P/E (Trailing)": "N/A",
        "P/E (Forward)": "N/A",
        "EPS (Trailing)": "N/A",
        "EPS Growth (YoY)": "N/A",
        "Revenue Growth (YoY)": "N/A",
        "Free Cash Flow": "N/A",
        "FCF Yield": "N/A",
        "Debt-to-Equity": "N/A",
        "Current Ratio": "N/A",
        "ROE": "N/A",
        "ROIC": "N/A",
        "Gross Margin": "N/A",
        "Operating Margin": "N/A",
        "Net Margin": "N/A",
        "Dividend Yield": "N/A",
    }

    try:
        data = get_financial_statements(ticker)
        info = data["info"]
        bs = data["balance_sheet"]
        inc = data["income_statement"]
        cf = data["cash_flow"]

        # --- Market Cap ---
        market_cap = info.get("marketCap")
        metrics["Market Cap"] = _currency_fmt(market_cap)

        # --- P/E Ratios ---
        metrics["P/E (Trailing)"] = _ratio_fmt(info.get("trailingPE"))
        metrics["P/E (Forward)"] = _ratio_fmt(info.get("forwardPE"))

        # --- EPS ---
        trailing_eps = info.get("trailingEps")
        metrics["EPS (Trailing)"] = _ratio_fmt(trailing_eps)

        # --- EPS Growth YoY ---
        eps_current = _extract_item(inc, ["Diluted EPS", "Basic EPS"], 0)
        eps_prior = _extract_item(inc, ["Diluted EPS", "Basic EPS"], 1)
        eps_growth = _safe_div((eps_current - eps_prior) if eps_current is not None and eps_prior is not None else None,
                               abs(eps_prior) if eps_prior is not None and eps_prior != 0 else None)
        metrics["EPS Growth (YoY)"] = _pct_fmt(eps_growth)

        # --- Revenue Growth YoY ---
        rev_current = _extract_item(inc, ["Total Revenue", "Revenue", "Operating Revenue"], 0)
        rev_prior = _extract_item(inc, ["Total Revenue", "Revenue", "Operating Revenue"], 1)
        rev_growth = _safe_div(
            (rev_current - rev_prior) if rev_current is not None and rev_prior is not None else None,
            abs(rev_prior) if rev_prior is not None and rev_prior != 0 else None,
        )
        metrics["Revenue Growth (YoY)"] = _pct_fmt(rev_growth)

        # --- Free Cash Flow ---
        op_cf = _extract_item(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)
        capex = _extract_item(cf, ["Capital Expenditure", "Capital Expenditures"], 0)
        fcf: float | None = None
        if op_cf is not None and capex is not None:
            # capex from yfinance is typically negative; FCF = OpCF + capex (where capex < 0)
            fcf = op_cf + capex  # capex already negative
        elif op_cf is not None:
            fcf = op_cf  # fallback: treat capex as 0
        metrics["Free Cash Flow"] = _currency_fmt(fcf)

        # --- FCF Yield ---
        fcf_yield = _safe_div(fcf, market_cap)
        metrics["FCF Yield"] = _pct_fmt(fcf_yield)

        # --- Debt-to-Equity ---
        total_debt = _extract_item(bs, ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
        total_equity = _extract_item(bs, ["Total Stockholders Equity", "Stockholders Equity",
                                          "Total Equity Gross Minority Interest", "Common Stock Equity"], 0)
        de_ratio = _safe_div(total_debt, total_equity)
        metrics["Debt-to-Equity"] = _ratio_fmt(de_ratio)

        # --- Current Ratio ---
        current_assets = _extract_item(bs, ["Current Assets", "Total Current Assets"], 0)
        current_liab = _extract_item(bs, ["Current Liabilities", "Total Current Liabilities",
                                          "Current Liabilities And Provisions"], 0)
        cr = _safe_div(current_assets, current_liab)
        metrics["Current Ratio"] = _ratio_fmt(cr)

        # --- ROE ---
        net_income = _extract_item(inc, ["Net Income", "Net Income Common Stockholders",
                                         "Net Income From Continuing Operations"], 0)
        roe = _safe_div(net_income, total_equity)
        metrics["ROE"] = _pct_fmt(roe)

        # --- ROIC ---
        # ROIC = NOPAT / Invested Capital
        # NOPAT = Operating Income * (1 - Tax Rate)
        operating_income = _extract_item(inc, ["Operating Income", "EBIT", "Operating Income Computed"], 0)
        tax_provision = _extract_item(inc, ["Tax Provision", "Income Tax Expense"], 0)
        pretax_income = _extract_item(inc, ["Pretax Income", "Income Before Tax",
                                            "Net Income Before Taxes"], 0)
        tax_rate = _safe_div(tax_provision, pretax_income) if tax_provision is not None and pretax_income is not None else None
        nopat: float | None = None
        if operating_income is not None and tax_rate is not None:
            nopat = operating_income * (1.0 - tax_rate)
        elif operating_income is not None:
            nopat = operating_income * 0.75  # assume ~25% tax

        # Invested Capital = Total Equity + Total Debt - Cash
        cash = _extract_item(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                                   "Cash Financial"], 0)
        invested_capital: float | None = None
        if total_equity is not None and total_debt is not None:
            invested_capital = total_equity + total_debt - (cash or 0.0)
        elif total_equity is not None:
            invested_capital = total_equity - (cash or 0.0)

        roic = _safe_div(nopat, invested_capital)
        metrics["ROIC"] = _pct_fmt(roic)

        # --- Margins ---
        gross_profit = _extract_item(inc, ["Gross Profit"], 0)
        gross_margin = _safe_div(gross_profit, rev_current)
        metrics["Gross Margin"] = _pct_fmt(gross_margin)

        operating_margin = _safe_div(operating_income, rev_current)
        metrics["Operating Margin"] = _pct_fmt(operating_margin)

        net_margin = _safe_div(net_income, rev_current)
        metrics["Net Margin"] = _pct_fmt(net_margin)

        # --- Dividend Yield ---
        div_yield = info.get("dividendYield")
        metrics["Dividend Yield"] = _pct_fmt(div_yield) if div_yield is not None else "N/A"

        # Add snake_case aliases for UI convenience
        aliases = {
            "market_cap": metrics.get("Market Cap"),
            "pe_trailing": metrics.get("P/E (Trailing)"),
            "pe_forward": metrics.get("P/E (Forward)"),
            "eps_trailing": metrics.get("EPS (Trailing)"),
            "eps_growth": metrics.get("EPS Growth (YoY)"),
            "revenue_growth": metrics.get("Revenue Growth (YoY)"),
            "free_cash_flow": metrics.get("Free Cash Flow"),
            "fcf_yield": metrics.get("FCF Yield"),
            "debt_to_equity": metrics.get("Debt-to-Equity"),
            "current_ratio": metrics.get("Current Ratio"),
            "roe": metrics.get("ROE"),
            "roic": metrics.get("ROIC"),
            "gross_margin": metrics.get("Gross Margin"),
            "operating_margin": metrics.get("Operating Margin"),
            "net_margin": metrics.get("Net Margin"),
            "dividend_yield": metrics.get("Dividend Yield"),
        }
        metrics.update(aliases)

    except Exception as exc:
        logger.error("calculate_key_metrics failed for %s: %s", ticker, exc)

    return metrics


# ---------------------------------------------------------------------------
# 3. Piotroski F-Score
# ---------------------------------------------------------------------------

def calculate_piotroski_f_score(ticker: str) -> dict[str, Any]:
    """
    Calculate the Piotroski F-Score (0–9) for *ticker*.

    The score comprises nine binary signals across profitability (4 pts),
    leverage / liquidity (3 pts), and operating efficiency (2 pts).

    Returns
    -------
    dict
        ``total_score`` (int 0-9), ``breakdown`` (dict of bools),
        ``interpretation`` (str).
    """
    breakdown: dict[str, bool] = {
        "positive_roa": False,
        "positive_cfo": False,
        "delta_roa": False,
        "accruals": False,
        "delta_leverage": False,
        "delta_liquidity": False,
        "no_dilution": False,
        "delta_gross_margin": False,
        "delta_asset_turnover": False,
    }

    result: dict[str, Any] = {
        "total_score": 0,
        "breakdown": breakdown,
        "interpretation": "Weak (0-3)",
    }

    try:
        data = get_financial_statements(ticker)
        bs = data["balance_sheet"]
        inc = data["income_statement"]
        cf = data["cash_flow"]

        if inc.empty or bs.empty or cf.empty:
            logger.warning("Insufficient data for Piotroski score: %s", ticker)
            return result

        # ---- helper extractions (current = col 0, prior = col 1) ----
        net_income_curr = _extract_item(inc, ["Net Income", "Net Income Common Stockholders",
                                               "Net Income From Continuing Operations"], 0)
        total_assets_curr = _extract_item(bs, ["Total Assets"], 0)
        total_assets_prior = _extract_item(bs, ["Total Assets"], 1)
        avg_assets_curr = None
        if total_assets_curr is not None and total_assets_prior is not None:
            avg_assets_curr = (total_assets_curr + total_assets_prior) / 2.0
        elif total_assets_curr is not None:
            avg_assets_curr = total_assets_curr

        roa_curr = _safe_div(net_income_curr, avg_assets_curr)

        net_income_prior = _extract_item(inc, ["Net Income", "Net Income Common Stockholders",
                                                "Net Income From Continuing Operations"], 1)
        total_assets_prior2 = _extract_item(bs, ["Total Assets"], 2)
        avg_assets_prior = None
        if total_assets_prior is not None and total_assets_prior2 is not None:
            avg_assets_prior = (total_assets_prior + total_assets_prior2) / 2.0
        elif total_assets_prior is not None:
            avg_assets_prior = total_assets_prior

        roa_prior = _safe_div(net_income_prior, avg_assets_prior)

        op_cf_curr = _extract_item(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"], 0)

        # ----------- PROFITABILITY (4 points) -----------

        # 1. ROA > 0
        if roa_curr is not None and roa_curr > 0:
            breakdown["positive_roa"] = True

        # 2. Operating Cash Flow > 0
        if op_cf_curr is not None and op_cf_curr > 0:
            breakdown["positive_cfo"] = True

        # 3. Delta ROA (improving)
        if roa_curr is not None and roa_prior is not None and roa_curr > roa_prior:
            breakdown["delta_roa"] = True

        # 4. Accruals: CFO > Net Income (quality of earnings)
        if op_cf_curr is not None and net_income_curr is not None and op_cf_curr > net_income_curr:
            breakdown["accruals"] = True

        # ----------- LEVERAGE / LIQUIDITY (3 points) -----------

        # 5. Long-term debt ratio decreased YoY
        lt_debt_curr = _extract_item(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
        lt_debt_prior = _extract_item(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 1)
        lt_debt_ratio_curr = _safe_div(lt_debt_curr, total_assets_curr)
        lt_debt_ratio_prior = _safe_div(lt_debt_prior, total_assets_prior)
        if lt_debt_ratio_curr is not None and lt_debt_ratio_prior is not None:
            if lt_debt_ratio_curr < lt_debt_ratio_prior:
                breakdown["delta_leverage"] = True
        elif lt_debt_curr is None or lt_debt_curr == 0:
            # No debt is a pass
            breakdown["delta_leverage"] = True

        # 6. Current ratio increased YoY
        ca_curr = _extract_item(bs, ["Current Assets", "Total Current Assets"], 0)
        cl_curr = _extract_item(bs, ["Current Liabilities", "Total Current Liabilities",
                                     "Current Liabilities And Provisions"], 0)
        ca_prior = _extract_item(bs, ["Current Assets", "Total Current Assets"], 1)
        cl_prior = _extract_item(bs, ["Current Liabilities", "Total Current Liabilities",
                                      "Current Liabilities And Provisions"], 1)
        cr_curr = _safe_div(ca_curr, cl_curr)
        cr_prior = _safe_div(ca_prior, cl_prior)
        if cr_curr is not None and cr_prior is not None and cr_curr > cr_prior:
            breakdown["delta_liquidity"] = True

        # 7. No dilution (shares outstanding same or decreased)
        shares_curr = _extract_item(bs, ["Share Issued", "Ordinary Shares Number",
                                          "Common Stock Shares Outstanding"], 0)
        shares_prior = _extract_item(bs, ["Share Issued", "Ordinary Shares Number",
                                           "Common Stock Shares Outstanding"], 1)
        if shares_curr is not None and shares_prior is not None:
            if shares_curr <= shares_prior:
                breakdown["no_dilution"] = True
        elif shares_curr is not None:
            # No prior data — assume no dilution
            breakdown["no_dilution"] = True

        # ----------- OPERATING EFFICIENCY (2 points) -----------

        # 8. Gross margin increased YoY
        revenue_curr = _extract_item(inc, ["Total Revenue", "Revenue", "Operating Revenue"], 0)
        revenue_prior = _extract_item(inc, ["Total Revenue", "Revenue", "Operating Revenue"], 1)
        gp_curr = _extract_item(inc, ["Gross Profit"], 0)
        gp_prior = _extract_item(inc, ["Gross Profit"], 1)
        gm_curr = _safe_div(gp_curr, revenue_curr)
        gm_prior = _safe_div(gp_prior, revenue_prior)
        if gm_curr is not None and gm_prior is not None and gm_curr > gm_prior:
            breakdown["delta_gross_margin"] = True

        # 9. Asset turnover ratio increased YoY
        at_curr = _safe_div(revenue_curr, avg_assets_curr)
        at_prior = _safe_div(revenue_prior, avg_assets_prior)
        if at_curr is not None and at_prior is not None and at_curr > at_prior:
            breakdown["delta_asset_turnover"] = True

        # ----------- Totals -----------
        total = sum(1 for v in breakdown.values() if v)
        result["total_score"] = total

        if total >= 7:
            result["interpretation"] = "Strong (7-9)"
        elif total >= 4:
            result["interpretation"] = "Moderate (4-6)"
        else:
            result["interpretation"] = "Weak (0-3)"

    except Exception as exc:
        logger.error("calculate_piotroski_f_score failed for %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# 4. Fundamental Card
# ---------------------------------------------------------------------------

def get_fundamental_card(ticker: str) -> dict[str, Any]:
    """
    Return a comprehensive fundamental-analysis card for *ticker*.

    Combines company metadata, key financial metrics, Piotroski F-Score,
    and an overall health rating.
    """
    card: dict[str, Any] = {
        "ticker": ticker.upper(),
        "company_name": "N/A",
        "sector": "N/A",
        "industry": "N/A",
        "key_metrics": {},
        "piotroski": {},
        "health_rating": "N/A",
    }

    try:
        t = yf.Ticker(ticker)
        info: dict = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        card["company_name"] = info.get("longName") or info.get("shortName") or "N/A"
        card["sector"] = info.get("sector", "N/A")
        card["industry"] = info.get("industry", "N/A")

        # Key metrics
        key_metrics = calculate_key_metrics(ticker)
        card["key_metrics"] = key_metrics

        # Piotroski
        piotroski = calculate_piotroski_f_score(ticker)
        card["piotroski"] = piotroski

        # --- Health rating (composite heuristic) ---
        health_points = 0
        total_checks = 5

        # 1. P/E reasonableness (0 < P/E < 30)
        pe_val = key_metrics.get("P/E (Trailing)")
        try:
            if isinstance(pe_val, (int, float)) and 0 < pe_val < 30:
                health_points += 1
        except (TypeError, ValueError):
            pass

        # 2. Positive FCF
        fcf_str = key_metrics.get("Free Cash Flow", "N/A")
        if isinstance(fcf_str, str) and fcf_str != "N/A" and not fcf_str.startswith("-") and fcf_str.startswith("$"):
            health_points += 1

        # 3. D/E < 1
        de_val = key_metrics.get("Debt-to-Equity")
        try:
            if isinstance(de_val, (int, float)) and 0 <= de_val < 1:
                health_points += 1
        except (TypeError, ValueError):
            pass

        # 4. EPS growing
        eps_g_str = key_metrics.get("EPS Growth (YoY)", "N/A")
        if isinstance(eps_g_str, str) and eps_g_str != "N/A" and not eps_g_str.startswith("-"):
            health_points += 1

        # 5. F-Score >= 6
        if piotroski.get("total_score", 0) >= 6:
            health_points += 1

        # Rating
        ratio = health_points / total_checks
        if ratio >= 0.6:
            card["health_rating"] = "STRONG"
        elif ratio >= 0.4:
            card["health_rating"] = "MODERATE"
        else:
            card["health_rating"] = "WEAK"

    except Exception as exc:
        logger.error("get_fundamental_card failed for %s: %s", ticker, exc)

    return card


# ---------------------------------------------------------------------------
# 5. Compare Fundamentals
# ---------------------------------------------------------------------------

PRIMARY_METRICS = [
    "Market Cap", "P/E (Trailing)", "P/E (Forward)", "EPS (Trailing)",
    "EPS Growth (YoY)", "Revenue Growth (YoY)", "Free Cash Flow",
    "FCF Yield", "Debt-to-Equity", "Current Ratio", "ROE", "ROIC",
    "Gross Margin", "Operating Margin", "Net Margin", "Dividend Yield",
]


def compare_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """
    Compare key fundamental metrics across multiple *tickers*.

    Returns a DataFrame with metric names as the index and each ticker as a
    column.  Returns an empty DataFrame if all tickers fail.
    """
    if not tickers:
        return pd.DataFrame()

    results: dict[str, dict[str, Any]] = {}
    for t in tickers:
        try:
            m = calculate_key_metrics(t)
            results[t.upper()] = {k: m.get(k, "N/A") for k in PRIMARY_METRICS}
        except Exception as exc:
            logger.error("compare_fundamentals: failed for %s: %s", t, exc)
            results[t.upper()] = {k: "N/A" for k in PRIMARY_METRICS}

    df = pd.DataFrame(results)
    df.index.name = "Metric"
    return df
