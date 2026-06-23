"""Pure analytics for the trading dashboard.

Indicator computation, signal generation, and backtesting live here with no
Gradio/Plotly/yfinance imports, so the logic can be unit-tested in isolation.
``app.py`` imports these functions for its UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_INITIAL_CAPITAL = 10_000.0


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on an OHLCV DataFrame.

    Adds the following columns:
        SMA_20, SMA_50, EMA_12, EMA_26, RSI,
        MACD, MACD_Signal, MACD_Hist,
        BB_Upper, BB_Middle, BB_Lower
    """
    import ta  # local import: only indicator computation needs it

    close = df["Close"].astype(float)

    # Simple Moving Averages
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)

    # Exponential Moving Averages
    df["EMA_12"] = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"] = ta.trend.ema_indicator(close, window=26)

    # Relative Strength Index
    df["RSI"] = ta.momentum.rsi(close, window=14)

    # MACD
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"] = macd_obj.macd_diff()

    # Bollinger Bands
    bb_obj = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_obj.bollinger_hband()
    df["BB_Middle"] = bb_obj.bollinger_mavg()
    df["BB_Lower"] = bb_obj.bollinger_lband()

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate composite buy/sell signals from indicator crossovers.

    Adds columns: Signal, Signal_Reason.
    Signal values: 1 (buy), -1 (sell), 0 (hold).
    """
    n = len(df)
    signals = np.zeros(n, dtype=int)
    reasons = [""] * n

    sma20 = df["SMA_20"].values
    sma50 = df["SMA_50"].values
    macd = df["MACD"].values
    macd_sig = df["MACD_Signal"].values
    rsi = df["RSI"].values

    for i in range(1, n):
        buy_reasons: list[str] = []
        sell_reasons: list[str] = []

        # --- SMA crossover --------------------------------------------------
        if _crosses_above(sma20, sma50, i):
            buy_reasons.append("SMA 20/50 golden cross")
        elif _crosses_below(sma20, sma50, i):
            sell_reasons.append("SMA 20/50 death cross")

        # --- MACD crossover --------------------------------------------------
        if _crosses_above(macd, macd_sig, i):
            buy_reasons.append("MACD bullish crossover")
        elif _crosses_below(macd, macd_sig, i):
            sell_reasons.append("MACD bearish crossover")

        # --- RSI extremes ----------------------------------------------------
        if not np.isnan(rsi[i]) and not np.isnan(rsi[i - 1]):
            if rsi[i - 1] <= 30 < rsi[i]:
                buy_reasons.append("RSI exits oversold (<30)")
            elif rsi[i - 1] >= 70 > rsi[i]:
                sell_reasons.append("RSI exits overbought (>70)")

        # --- Composite decision (majority vote) ------------------------------
        if len(buy_reasons) > len(sell_reasons) and len(buy_reasons) >= 1:
            signals[i] = 1
            reasons[i] = "; ".join(buy_reasons)
        elif len(sell_reasons) > len(buy_reasons) and len(sell_reasons) >= 1:
            signals[i] = -1
            reasons[i] = "; ".join(sell_reasons)

    df["Signal"] = signals
    df["Signal_Reason"] = reasons
    return df


def _crosses_above(fast: np.ndarray, slow: np.ndarray, i: int) -> bool:
    """Return True if *fast* crosses above *slow* at index *i*."""
    if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(fast[i - 1]) or np.isnan(slow[i - 1]):
        return False
    return fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]


def _crosses_below(fast: np.ndarray, slow: np.ndarray, i: int) -> bool:
    """Return True if *fast* crosses below *slow* at index *i*."""
    if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(fast[i - 1]) or np.isnan(slow[i - 1]):
        return False
    return fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]


def build_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame of only the rows where a signal fired."""
    mask = df["Signal"] != 0
    if mask.sum() == 0:
        return pd.DataFrame(columns=["Date", "Close", "Type", "Reason"])

    out = df.loc[mask, ["Close", "Signal", "Signal_Reason"]].copy()
    out.index.name = "Date"
    out = out.reset_index()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out["Close"] = out["Close"].round(2)
    out["Type"] = out["Signal"].map({1: "BUY", -1: "SELL"})
    out["Reason"] = out["Signal_Reason"]
    return out[["Date", "Close", "Type", "Reason"]]


def run_backtest(
    df: pd.DataFrame, initial_capital: float = DEFAULT_INITIAL_CAPITAL
) -> dict:
    """Run a simple signal-following backtest and compare to buy-and-hold.

    Strategy rules:
        - Start in cash.
        - On a BUY signal, go fully invested (buy at close).
        - On a SELL signal, exit to cash (sell at close).
        - At the end, liquidate any open position.

    Returns a dict with summary statistics.
    """
    close = df["Close"].values.astype(float)
    signals = df["Signal"].values

    # Buy-and-hold
    bh_return = (close[-1] / close[0] - 1) * 100 if close[0] != 0 else 0.0

    # Signal strategy
    cash = float(initial_capital)
    shares = 0.0
    in_position = False
    trades = 0
    winning_trades = 0
    entry_price = 0.0

    equity_curve = np.full(len(close), np.nan)

    for i in range(len(close)):
        if signals[i] == 1 and not in_position and cash > 0:
            shares = cash / close[i]
            entry_price = close[i]
            cash = 0.0
            in_position = True
            trades += 1
        elif signals[i] == -1 and in_position:
            cash = shares * close[i]
            if close[i] > entry_price:
                winning_trades += 1
            shares = 0.0
            in_position = False

        equity_curve[i] = cash + shares * close[i]

    # Liquidate at end if still in position
    if in_position:
        cash = shares * close[-1]
        if close[-1] > entry_price:
            winning_trades += 1
        shares = 0.0
        equity_curve[-1] = cash

    final_value = cash + shares * close[-1]
    strategy_return = (final_value / initial_capital - 1) * 100
    win_rate = (winning_trades / trades * 100) if trades > 0 else 0.0

    # Equity curve for charting (forward-fill NaN gaps)
    eq_series = pd.Series(equity_curve)
    eq_series = eq_series.ffill().bfill()

    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "strategy_return_pct": round(strategy_return, 2),
        "buy_hold_return_pct": round(bh_return, 2),
        "total_trades": trades,
        "winning_trades": winning_trades,
        "win_rate_pct": round(win_rate, 1),
        "equity_curve": eq_series.values,
    }
