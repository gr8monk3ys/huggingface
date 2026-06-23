"""Tests for the trading dashboard's pure analytics (core.py)."""

import numpy as np
import pandas as pd
import pytest

from conftest import load_local_module

core = load_local_module("trading_core", "trading-dashboard-space/core.py")


def test_crosses_above_true():
    fast = np.array([1.0, 3.0])
    slow = np.array([2.0, 2.0])
    assert core._crosses_above(fast, slow, 1)
    assert not core._crosses_below(fast, slow, 1)


def test_crosses_below_true():
    fast = np.array([3.0, 1.0])
    slow = np.array([2.0, 2.0])
    assert core._crosses_below(fast, slow, 1)
    assert not core._crosses_above(fast, slow, 1)


def test_crosses_handle_nan():
    fast = np.array([np.nan, 3.0])
    slow = np.array([2.0, 2.0])
    assert not core._crosses_above(fast, slow, 1)


def test_generate_signals_emits_golden_cross_buy():
    df = pd.DataFrame(
        {
            "SMA_20": [1.0, 1.0, 3.0],
            "SMA_50": [2.0, 2.0, 2.0],
            "MACD": [0.0, 0.0, 0.0],
            "MACD_Signal": [0.0, 0.0, 0.0],
            "RSI": [50.0, 50.0, 50.0],
        }
    )
    out = core.generate_signals(df)
    assert out["Signal"].tolist() == [0, 0, 1]
    assert "golden cross" in out["Signal_Reason"].iloc[2]


def test_run_backtest_full_invest_then_liquidate():
    df = pd.DataFrame({"Close": [10.0, 20.0], "Signal": [1, 0]})
    bt = core.run_backtest(df, initial_capital=10_000.0)
    assert bt["total_trades"] == 1
    assert bt["final_value"] == 20_000.0
    assert bt["strategy_return_pct"] == 100.0
    assert bt["buy_hold_return_pct"] == 100.0
    assert bt["win_rate_pct"] == 100.0
    assert len(bt["equity_curve"]) == 2


def test_run_backtest_no_signals_stays_in_cash():
    df = pd.DataFrame({"Close": [10.0, 12.0, 11.0], "Signal": [0, 0, 0]})
    bt = core.run_backtest(df, initial_capital=5_000.0)
    assert bt["total_trades"] == 0
    assert bt["final_value"] == 5_000.0
    assert bt["strategy_return_pct"] == 0.0


def test_build_signal_table_filters_and_labels():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    df = pd.DataFrame(
        {"Close": [10.0, 20.0], "Signal": [0, 1], "Signal_Reason": ["", "buy reason"]},
        index=idx,
    )
    table = core.build_signal_table(df)
    assert list(table.columns) == ["Date", "Close", "Type", "Reason"]
    assert len(table) == 1
    assert table["Type"].iloc[0] == "BUY"
    assert table["Date"].iloc[0] == "2024-01-02"


def test_build_signal_table_empty_when_no_signals():
    df = pd.DataFrame({"Close": [10.0], "Signal": [0], "Signal_Reason": [""]})
    table = core.build_signal_table(df)
    assert list(table.columns) == ["Date", "Close", "Type", "Reason"]
    assert len(table) == 0


def test_compute_indicators_adds_columns():
    ta = pytest.importorskip("ta")  # noqa: F841 - skip if 'ta' isn't installed
    close = np.linspace(100, 160, 60)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close})
    out = core.compute_indicators(df)
    for col in ("SMA_20", "SMA_50", "RSI", "MACD", "BB_Upper"):
        assert col in out.columns
    assert not np.isnan(out["SMA_20"].iloc[-1])
