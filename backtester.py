"""
NEXUS-QUANT Backtesting Engine
==============================
Deterministic, bar-by-bar backtesting engine with support for long/short trades,
breakeven trailing, news blackout filtering, and comprehensive performance metrics.

Only one trade is open at a time. PnL accounts for spread and slippage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TradeParams:
    """Parameters governing trade execution and risk management."""

    sl_buffer_points: float = 0.0002    # breathing room beyond S&R for SL
    tp_rr_ratio: float = 2.5            # take-profit risk:reward (1:2.2–1:3)
    breakeven_rr: float = 1.0           # move SL to entry at this RR multiple
    slippage: float = 0.0001            # per-trade slippage (adverse direction)
    spread: float = 0.0002              # bid-ask spread applied at entry
    risk_per_trade_pct: float = 1.0     # % of capital risked per trade
    initial_capital: float = 10_000.0


@dataclass
class Trade:
    """Represents a single trade from entry to exit."""

    entry_time: pd.Timestamp
    direction: str                       # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    status: str = "OPEN"                 # OPEN | CLOSED_TP | CLOSED_SL | CLOSED_BE
    setup_type: str = "S&R Bounce"       # S&R Bounce | FVG Retest | S&R + FVG Confluence
    trigger_mode: str = "Reversal"       # RSI Reversal | Momentum Breakout
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    rr_achieved: float = 0.0
    be_triggered: bool = False
    bars_held: int = 0


@dataclass
class BacktestResult:
    """Container for backtest output."""

    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict[str, object]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Bar-by-bar, single-position backtesting engine.

    Parameters
    ----------
    params : TradeParams
        Execution / risk parameters for the backtest.
    """

    # Required indicator columns that must be present in the DataFrame.
    _REQUIRED_COLUMNS: List[str] = [
        "trend_bias",
        "support_level",
        "resistance_level",
        "RSI",
        "rsi_mode",
        "active_bull_fvg_top",
        "active_bull_fvg_bottom",
        "active_bear_fvg_top",
        "active_bear_fvg_bottom",
    ]

    def __init__(self, params: TradeParams | None = None) -> None:
        self.params: TradeParams = params or TradeParams()

    # ------------------------------------------------------------------
    # Entry Signal Detection
    # ------------------------------------------------------------------

    def _check_entry_signal(
        self,
        df: pd.DataFrame,
        i: int,
    ) -> Optional[Tuple[str, float, float, float]]:
        """Evaluate whether bar *i* presents a valid entry signal.

        Returns
        -------
        tuple or None
            ``(direction, entry_price, stop_loss, take_profit)`` when a signal
            fires, otherwise ``None``.
        """
        row = df.iloc[i]

        trend_bias: str = str(row.get("trend_bias", "")).upper()
        close: float = float(row["Close"])
        support: float = float(row.get("support_level", np.nan))
        resistance: float = float(row.get("resistance_level", np.nan))
        rsi_mode: str = str(row.get("rsi_mode", "")).upper()

        bull_fvg_top: float = float(row.get("active_bull_fvg_top", np.nan))
        bull_fvg_bot: float = float(row.get("active_bull_fvg_bottom", np.nan))
        bear_fvg_top: float = float(row.get("active_bear_fvg_top", np.nan))
        bear_fvg_bot: float = float(row.get("active_bear_fvg_bottom", np.nan))

        # ----- LONG --------------------------------------------------
        if trend_bias == "LONG" and not np.isnan(support):
            near_support = (
                abs(close - support) / support <= 0.005
                if support != 0.0
                else False
            )
            in_bull_fvg = (
                not np.isnan(bull_fvg_bot)
                and not np.isnan(bull_fvg_top)
                and bull_fvg_bot <= close <= bull_fvg_top
            )
            location_ok = near_support or in_bull_fvg

            rsi_ok = rsi_mode in ("REVERSAL_LONG", "BREAKOUT_OK")

            if location_ok and rsi_ok:
                entry_price = close + self.params.spread  # buy at ask
                sl_price = support - self.params.sl_buffer_points
                risk = entry_price - sl_price
                if risk <= 0:
                    return None
                tp_price = entry_price + risk * self.params.tp_rr_ratio

                if near_support and in_bull_fvg:
                    setup_type = "S&R + FVG Confluence"
                elif near_support:
                    setup_type = "S&R Bounce"
                else:
                    setup_type = "FVG Retest"

                trigger_mode = "RSI Reversal" if rsi_mode == "REVERSAL_LONG" else "Momentum Breakout"

                return ("LONG", entry_price, sl_price, tp_price, setup_type, trigger_mode)

        # ----- SHORT -------------------------------------------------
        if trend_bias == "SHORT" and not np.isnan(resistance):
            near_resistance = (
                abs(close - resistance) / resistance <= 0.005
                if resistance != 0.0
                else False
            )
            in_bear_fvg = (
                not np.isnan(bear_fvg_bot)
                and not np.isnan(bear_fvg_top)
                and bear_fvg_bot <= close <= bear_fvg_top
            )
            location_ok = near_resistance or in_bear_fvg

            rsi_ok = rsi_mode in ("REVERSAL_SHORT", "BREAKOUT_OK")

            if location_ok and rsi_ok:
                entry_price = close - self.params.spread  # sell at bid
                sl_price = resistance + self.params.sl_buffer_points
                risk = sl_price - entry_price
                if risk <= 0:
                    return None
                tp_price = entry_price - risk * self.params.tp_rr_ratio

                if near_resistance and in_bear_fvg:
                    setup_type = "S&R + FVG Confluence"
                elif near_resistance:
                    setup_type = "S&R Bounce"
                else:
                    setup_type = "FVG Retest"

                trigger_mode = "RSI Reversal" if rsi_mode == "REVERSAL_SHORT" else "Momentum Breakout"

                return ("SHORT", entry_price, sl_price, tp_price, setup_type, trigger_mode)

        return None

    # ------------------------------------------------------------------
    # Open-Trade Management
    # ------------------------------------------------------------------

    def _manage_open_trade(self, trade: Trade, bar: pd.Series) -> None:
        """Advance an open trade by one bar, checking SL / TP / BE.

        Mutates *trade* in place.  After the call, inspect ``trade.status``
        to determine whether the trade was closed on this bar.
        """
        trade.bars_held += 1

        bar_low: float = float(bar["Low"])
        bar_high: float = float(bar["High"])
        bar_time: pd.Timestamp = bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp(bar.name)

        risk = abs(trade.entry_price - trade.stop_loss) if not trade.be_triggered else 0.0

        if trade.direction == "LONG":
            # --- 1. Stop-Loss check (worst case first) ----------------
            if bar_low <= trade.stop_loss:
                exit_px = trade.stop_loss - self.params.slippage  # adverse slip
                status = "CLOSED_BE" if trade.be_triggered else "CLOSED_SL"
                self._close_trade(trade, exit_px, bar_time, status)
                return

            # --- 2. Take-Profit check ---------------------------------
            if bar_high >= trade.take_profit:
                exit_px = trade.take_profit - self.params.slippage  # conservative fill
                self._close_trade(trade, exit_px, bar_time, "CLOSED_TP")
                return

            # --- 3. Breakeven trail -----------------------------------
            if not trade.be_triggered:
                original_risk = trade.entry_price - trade.stop_loss
                if original_risk > 0:
                    favorable_excursion = bar_high - trade.entry_price
                    rr_reached = favorable_excursion / original_risk
                    if rr_reached >= self.params.breakeven_rr:
                        trade.stop_loss = trade.entry_price
                        trade.be_triggered = True

        else:  # SHORT
            # --- 1. Stop-Loss check -----------------------------------
            if bar_high >= trade.stop_loss:
                exit_px = trade.stop_loss + self.params.slippage
                status = "CLOSED_BE" if trade.be_triggered else "CLOSED_SL"
                self._close_trade(trade, exit_px, bar_time, status)
                return

            # --- 2. Take-Profit check ---------------------------------
            if bar_low <= trade.take_profit:
                exit_px = trade.take_profit + self.params.slippage
                self._close_trade(trade, exit_px, bar_time, "CLOSED_TP")
                return

            # --- 3. Breakeven trail -----------------------------------
            if not trade.be_triggered:
                original_risk = trade.stop_loss - trade.entry_price
                if original_risk > 0:
                    favorable_excursion = trade.entry_price - bar_low
                    rr_reached = favorable_excursion / original_risk
                    if rr_reached >= self.params.breakeven_rr:
                        trade.stop_loss = trade.entry_price
                        trade.be_triggered = True

    # ------------------------------------------------------------------
    # Trade Closing Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _close_trade(
        trade: Trade,
        exit_price: float,
        exit_time: pd.Timestamp,
        status: str,
    ) -> None:
        """Finalise a trade's exit fields and compute PnL."""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = status

        if trade.direction == "LONG":
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size

        notional = trade.entry_price * trade.position_size
        trade.pnl_pct = (trade.pnl / notional * 100.0) if notional != 0 else 0.0

        initial_risk = abs(trade.entry_price - trade.stop_loss) if not trade.be_triggered else 0.0
        # For RR calculation, always use the *original* risk (entry vs initial SL).
        # Since BE moves SL to entry, we infer original risk from TP & rr_ratio.
        # Fallback: use entry-to-exit magnitude / 1 when risk is 0.
        if initial_risk == 0:
            # BE was triggered before close — derive original risk from TP.
            # TP = entry ± original_risk * rr_ratio  →  original_risk = |TP - entry| / rr_ratio
            original_risk_estimate = abs(trade.take_profit - trade.entry_price)
            # rr_ratio is baked into TP; rr_achieved relative to *that* risk
            if original_risk_estimate != 0:
                if trade.direction == "LONG":
                    trade.rr_achieved = (exit_price - trade.entry_price) / (original_risk_estimate / 2.5)
                else:
                    trade.rr_achieved = (trade.entry_price - exit_price) / (original_risk_estimate / 2.5)
            else:
                trade.rr_achieved = 0.0
        else:
            if trade.direction == "LONG":
                trade.rr_achieved = (exit_price - trade.entry_price) / initial_risk
            else:
                trade.rr_achieved = (trade.entry_price - exit_price) / initial_risk

    # ------------------------------------------------------------------
    # Position Sizing
    # ------------------------------------------------------------------

    def _calculate_position_size(
        self,
        capital: float,
        entry: float,
        sl: float,
        risk_pct: float,
    ) -> float:
        """Kelly-lite position sizing.

        ``position_size = (capital * risk_pct / 100) / |entry − sl|``

        Returns 0.0 when the SL is at or beyond entry (invalid trade).
        """
        risk_per_unit = abs(entry - sl)
        if risk_per_unit == 0:
            return 0.0
        return (capital * risk_pct / 100.0) / risk_per_unit

    # ------------------------------------------------------------------
    # News Blackout Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _is_news_blackout(
        bar_time: pd.Timestamp,
        news_blackout_times: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]],
    ) -> bool:
        """Return True if *bar_time* falls inside any blackout window."""
        if news_blackout_times is None:
            return False
        for start, end in news_blackout_times:
            if start <= bar_time <= end:
                return True
        return False

    # ------------------------------------------------------------------
    # Main Backtest Loop
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        news_blackout_times: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    ) -> BacktestResult:
        """Execute the backtest on *df* (must contain OHLC + indicator columns).

        Parameters
        ----------
        df : pd.DataFrame
            OHLC DataFrame **with indicator columns** already appended
            (see ``_REQUIRED_COLUMNS``).  The index should be a
            ``DatetimeIndex``.
        news_blackout_times : list of (start, end) Timestamp pairs, optional
            Periods during which new entries are suppressed.

        Returns
        -------
        BacktestResult
        """
        if df.empty:
            return BacktestResult(
                trades=[],
                equity_curve=pd.Series(dtype=float),
                metrics=self.calculate_metrics([], pd.Series(dtype=float), self.params.initial_capital),
            )

        # Validate required columns ------------------------------------------
        missing = [c for c in self._REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required indicator columns: {missing}"
            )

        capital: float = self.params.initial_capital
        open_trade: Optional[Trade] = None
        closed_trades: List[Trade] = []
        equity_timestamps: List[pd.Timestamp] = []
        equity_values: List[float] = []

        warmup = min(50, len(df) - 1)

        for i in range(warmup, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i] if isinstance(df.index[i], pd.Timestamp) else pd.Timestamp(df.index[i])

            # --- Manage existing position ------------------------------------
            if open_trade is not None:
                self._manage_open_trade(open_trade, bar)

                if open_trade.status != "OPEN":
                    capital += open_trade.pnl
                    closed_trades.append(open_trade)
                    open_trade = None

            # --- Look for new entry ------------------------------------------
            if open_trade is None:
                if not self._is_news_blackout(bar_time, news_blackout_times):
                    signal = self._check_entry_signal(df, i)
                    if signal is not None:
                        direction, entry_px, sl_px, tp_px, setup_type, trigger_mode = signal
                        size = self._calculate_position_size(
                            capital, entry_px, sl_px, self.params.risk_per_trade_pct
                        )
                        if size > 0:
                            open_trade = Trade(
                                entry_time=bar_time,
                                direction=direction,
                                entry_price=entry_px,
                                stop_loss=sl_px,
                                take_profit=tp_px,
                                position_size=size,
                                setup_type=setup_type,
                                trigger_mode=trigger_mode,
                            )

            # --- Record equity -----------------------------------------------
            mark_to_market = capital
            if open_trade is not None:
                unrealised: float
                if open_trade.direction == "LONG":
                    unrealised = (float(bar["Close"]) - open_trade.entry_price) * open_trade.position_size
                else:
                    unrealised = (open_trade.entry_price - float(bar["Close"])) * open_trade.position_size
                mark_to_market += unrealised

            equity_timestamps.append(bar_time)
            equity_values.append(mark_to_market)

        # If a trade is still open at the end, force-close at last close ------
        if open_trade is not None:
            last_bar = df.iloc[-1]
            last_time = df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.Timestamp(df.index[-1])
            close_px = float(last_bar["Close"])
            if open_trade.direction == "LONG":
                close_px -= self.params.slippage
            else:
                close_px += self.params.slippage
            self._close_trade(open_trade, close_px, last_time, "CLOSED_SL")
            capital += open_trade.pnl
            closed_trades.append(open_trade)
            # Update final equity value
            if equity_values:
                equity_values[-1] = capital

        equity_curve = pd.Series(
            data=equity_values,
            index=pd.DatetimeIndex(equity_timestamps),
            name="equity",
            dtype=float,
        )

        metrics = self.calculate_metrics(closed_trades, equity_curve, self.params.initial_capital)
        return BacktestResult(trades=closed_trades, equity_curve=equity_curve, metrics=metrics)

    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------

    def calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        initial_capital: float,
    ) -> Dict[str, object]:
        """Compute comprehensive performance metrics.

        Returns a flat dictionary suitable for display or serialisation.
        """
        metrics: Dict[str, object] = {}

        total_trades = len(trades)
        metrics["total_trades"] = total_trades

        if total_trades == 0:
            metrics["win_rate_pct"] = 0.0
            metrics["loss_rate_pct"] = 0.0
            metrics["win_rate"] = 0.0
            metrics["loss_rate"] = 0.0
            metrics["total_pnl"] = 0.0
            metrics["total_return_pct"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["max_drawdown_pct"] = 0.0
            metrics["sharpe_ratio"] = 0.0
            metrics["avg_rr_achieved"] = 0.0
            metrics["avg_bars_held"] = 0.0
            metrics["best_trade_pnl"] = 0.0
            metrics["worst_trade_pnl"] = 0.0
            metrics["best_trade"] = 0.0
            metrics["worst_trade"] = 0.0
            metrics["max_consecutive_wins"] = 0
            metrics["max_consecutive_losses"] = 0
            metrics["monthly_returns"] = {}
            return metrics

        # Win / loss counts ---------------------------------------------------
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        metrics["win_rate_pct"] = round(len(wins) / total_trades * 100.0, 2)
        metrics["loss_rate_pct"] = round(len(losses) / total_trades * 100.0, 2)

        # PnL -----------------------------------------------------------------
        total_pnl = sum(t.pnl for t in trades)
        metrics["total_pnl"] = round(total_pnl, 4)
        metrics["total_return_pct"] = round(total_pnl / initial_capital * 100.0, 2) if initial_capital != 0 else 0.0

        # Profit factor -------------------------------------------------------
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        metrics["profit_factor"] = round(gross_profit / gross_loss, 4) if gross_loss != 0 else float("inf")

        # Max Drawdown % (from equity curve peak) -----------------------------
        if len(equity_curve) > 0:
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            metrics["max_drawdown_pct"] = round(float(drawdown.min()) * 100.0, 2)
        else:
            metrics["max_drawdown_pct"] = 0.0

        # Sharpe Ratio (annualised, daily returns approx) ---------------------
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 0 and returns.std() != 0:
                metrics["sharpe_ratio"] = round(
                    float(returns.mean() / returns.std() * np.sqrt(252)), 4
                )
            else:
                metrics["sharpe_ratio"] = 0.0
        else:
            metrics["sharpe_ratio"] = 0.0

        # Average RR & bars held ----------------------------------------------
        metrics["avg_rr_achieved"] = round(float(np.mean([t.rr_achieved for t in trades])), 4)
        metrics["avg_bars_held"] = round(float(np.mean([t.bars_held for t in trades])), 2)

        # Best / worst trade --------------------------------------------------
        metrics["best_trade_pnl"] = round(max(t.pnl for t in trades), 4)
        metrics["worst_trade_pnl"] = round(min(t.pnl for t in trades), 4)

        # Aliases for UI compatibility
        metrics["win_rate"] = metrics["win_rate_pct"]
        metrics["loss_rate"] = metrics["loss_rate_pct"]
        metrics["best_trade"] = metrics["best_trade_pnl"]
        metrics["worst_trade"] = metrics["worst_trade_pnl"]

        # Consecutive wins / losses -------------------------------------------
        max_con_wins = 0
        max_con_losses = 0
        cur_wins = 0
        cur_losses = 0
        for t in trades:
            if t.pnl > 0:
                cur_wins += 1
                cur_losses = 0
            else:
                cur_losses += 1
                cur_wins = 0
            max_con_wins = max(max_con_wins, cur_wins)
            max_con_losses = max(max_con_losses, cur_losses)
        metrics["max_consecutive_wins"] = max_con_wins
        metrics["max_consecutive_losses"] = max_con_losses

        # Monthly returns breakdown -------------------------------------------
        monthly_returns: Dict[str, float] = {}
        for t in trades:
            if t.exit_time is not None:
                key = t.exit_time.strftime("%Y-%m")
            else:
                key = t.entry_time.strftime("%Y-%m")
            monthly_returns[key] = monthly_returns.get(key, 0.0) + t.pnl
        # Round values
        monthly_returns = {k: round(v, 4) for k, v in sorted(monthly_returns.items())}
        metrics["monthly_returns"] = monthly_returns

        # Setup breakdown (S&R vs FVG vs Confluence) -------------------------
        setup_breakdown: Dict[str, Dict[str, object]] = {}
        for t in trades:
            stype = getattr(t, "setup_type", "S&R Bounce")
            if stype not in setup_breakdown:
                setup_breakdown[stype] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            setup_breakdown[stype]["trades"] += 1
            if t.pnl > 0:
                setup_breakdown[stype]["wins"] += 1
            else:
                setup_breakdown[stype]["losses"] += 1
            setup_breakdown[stype]["pnl"] += t.pnl

        for stype, d in setup_breakdown.items():
            cnt = d["trades"]
            d["win_rate_pct"] = round(d["wins"] / cnt * 100.0, 1) if cnt > 0 else 0.0
            d["pnl"] = round(d["pnl"], 2)
        metrics["setup_breakdown"] = setup_breakdown

        # Trigger mode breakdown (RSI Reversal Rejection vs Momentum Breakout)
        trigger_breakdown: Dict[str, Dict[str, object]] = {}
        for t in trades:
            tmode = getattr(t, "trigger_mode", "RSI Reversal")
            if tmode not in trigger_breakdown:
                trigger_breakdown[tmode] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            trigger_breakdown[tmode]["trades"] += 1
            if t.pnl > 0:
                trigger_breakdown[tmode]["wins"] += 1
            else:
                trigger_breakdown[tmode]["losses"] += 1
            trigger_breakdown[tmode]["pnl"] += t.pnl

        for tmode, d in trigger_breakdown.items():
            cnt = d["trades"]
            d["win_rate_pct"] = round(d["wins"] / cnt * 100.0, 1) if cnt > 0 else 0.0
            d["pnl"] = round(d["pnl"], 2)
        metrics["trigger_breakdown"] = trigger_breakdown

        return metrics
