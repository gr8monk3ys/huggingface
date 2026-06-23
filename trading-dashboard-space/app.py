"""
Trading Signal Dashboard
========================
A Gradio-based technical analysis dashboard that fetches real stock data,
computes indicators, generates buy/sell signals, and backtests strategies.

Version: 2.0.0 (Gradio 5.x compatible)

Author: Lorenzo Scaturchio (gr8monk3ys)
License: MIT

DISCLAIMER: This tool is for educational purposes only and does NOT
constitute financial advice. Use at your own risk.
"""

import datetime
from typing import Optional

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

from core import (
    build_signal_table,
    compute_indicators,
    generate_signals,
    run_backtest,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA"]

TIMEFRAME_MAP = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
}

COLORS = {
    "bg": "#0e1117",
    "card": "#1a1d29",
    "text": "#e0e0e0",
    "green": "#00d4aa",
    "red": "#ff6b6b",
    "blue": "#4dabf7",
    "purple": "#b197fc",
    "orange": "#ffa94d",
    "yellow": "#ffe066",
    "grid": "#2a2d3a",
    "band_fill": "rgba(77, 171, 247, 0.08)",
}

DISCLAIMER_TEXT = (
    "**Disclaimer:** This dashboard is for educational and informational "
    "purposes only. It does NOT constitute financial advice. Past performance "
    "does not guarantee future results. Always do your own research and consult "
    "a qualified financial advisor before making investment decisions."
)

# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------


def fetch_stock_data(
    ticker: str, timeframe: str
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch historical stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        timeframe: Human-readable timeframe key from TIMEFRAME_MAP.

    Returns:
        Tuple of (DataFrame with OHLCV data, error message or None).
    """
    days = TIMEFRAME_MAP.get(timeframe, 365)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)

    try:
        data = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None, f"No data found for ticker '{ticker}'. Verify the symbol."

        # Flatten MultiIndex columns if present (yfinance >= 0.2.31 quirk)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Ensure expected columns exist
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(data.columns)):
            return None, f"Incomplete data columns for '{ticker}'."

        data = data.copy()
        data.index = pd.to_datetime(data.index)
        return data, None

    except Exception as exc:
        return None, f"Error fetching data for '{ticker}': {exc}"


# ---------------------------------------------------------------------------
# Chart Building
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11),
    ),
    margin=dict(l=60, r=30, t=60, b=40),
)


def build_main_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Build the multi-subplot price / RSI / MACD / Volume chart."""

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.18, 0.18, 0.19],
        subplot_titles=("", "", "", ""),
    )

    dates = df.index

    # --- Row 1: Candlestick + overlays + Bollinger Bands --------------------
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Bollinger Bands (shaded region)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["BB_Upper"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["BB_Lower"],
            fill="tonexty",
            fillcolor=COLORS["band_fill"],
            line=dict(width=0),
            name="Bollinger Bands",
        ),
        row=1,
        col=1,
    )

    # SMA / EMA lines
    for col_name, color, dash in [
        ("SMA_20", COLORS["blue"], "solid"),
        ("SMA_50", COLORS["purple"], "solid"),
        ("EMA_12", COLORS["orange"], "dot"),
        ("EMA_26", COLORS["yellow"], "dot"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[col_name],
                mode="lines",
                line=dict(color=color, width=1.2, dash=dash),
                name=col_name,
            ),
            row=1,
            col=1,
        )

    # Buy / Sell markers
    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Close"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=COLORS["green"],
                    line=dict(color="white", width=1),
                ),
                name="Buy Signal",
                text=buys["Signal_Reason"],
                hovertemplate="%{text}<extra>BUY</extra>",
            ),
            row=1,
            col=1,
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Close"],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=COLORS["red"],
                    line=dict(color="white", width=1),
                ),
                name="Sell Signal",
                text=sells["Signal_Reason"],
                hovertemplate="%{text}<extra>SELL</extra>",
            ),
            row=1,
            col=1,
        )

    # --- Row 2: RSI ---------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["RSI"],
            mode="lines",
            line=dict(color=COLORS["purple"], width=1.3),
            name="RSI (14)",
        ),
        row=2,
        col=1,
    )
    # Overbought / oversold lines
    for level, clr in [(70, COLORS["red"]), (30, COLORS["green"])]:
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color=clr,
            opacity=0.5,
            row=2,
            col=1,
        )
    # Shade the 30-70 zone
    fig.add_hrect(
        y0=30,
        y1=70,
        fillcolor="rgba(255,255,255,0.03)",
        line_width=0,
        row=2,
        col=1,
    )

    # --- Row 3: MACD --------------------------------------------------------
    macd_colors = [
        COLORS["green"] if v >= 0 else COLORS["red"]
        for v in df["MACD_Hist"].fillna(0)
    ]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=df["MACD_Hist"],
            marker_color=macd_colors,
            name="MACD Hist",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["MACD"],
            mode="lines",
            line=dict(color=COLORS["blue"], width=1.2),
            name="MACD",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["MACD_Signal"],
            mode="lines",
            line=dict(color=COLORS["orange"], width=1.2),
            name="Signal Line",
        ),
        row=3,
        col=1,
    )

    # --- Row 4: Volume ------------------------------------------------------
    vol_colors = [
        COLORS["green"] if c >= o else COLORS["red"]
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=df["Volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
        ),
        row=4,
        col=1,
    )

    # --- Layout -------------------------------------------------------------
    fig.update_layout(
        **_PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text=f"{ticker} -- Technical Analysis Dashboard",
            font=dict(size=20),
            x=0.5,
        ),
        height=900,
        xaxis_rangeslider_visible=False,
    )

    # Y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="RSI", row=2, col=1, gridcolor=COLORS["grid"], range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Volume", row=4, col=1, gridcolor=COLORS["grid"])

    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS["grid"], row=i, col=1)

    return fig


def build_backtest_chart(
    df: pd.DataFrame, backtest: dict, ticker: str
) -> go.Figure:
    """Build the equity-curve comparison chart for the backtest tab."""

    close = df["Close"].values.astype(float)
    dates = df.index

    # Normalize buy-and-hold to same starting capital
    bh_equity = (close / close[0]) * backtest["initial_capital"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=backtest["equity_curve"],
            mode="lines",
            name="Signal Strategy",
            line=dict(color=COLORS["green"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.07)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bh_equity,
            mode="lines",
            name="Buy & Hold",
            line=dict(color=COLORS["blue"], width=2, dash="dot"),
        )
    )

    # Buy / sell markers on equity curve
    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]

    if not buys.empty:
        buy_indices = [df.index.get_loc(d) for d in buys.index]
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=[backtest["equity_curve"][i] for i in buy_indices],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=COLORS["green"]),
                name="Buy",
                showlegend=False,
            )
        )

    if not sells.empty:
        sell_indices = [df.index.get_loc(d) for d in sells.index]
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=[backtest["equity_curve"][i] for i in sell_indices],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color=COLORS["red"]),
                name="Sell",
                showlegend=False,
            )
        )

    fig.update_layout(
        **_PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text=f"{ticker} -- Strategy vs Buy & Hold (${backtest['initial_capital']:,.0f} start)",
            font=dict(size=18),
            x=0.5,
        ),
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        height=500,
    )

    fig.update_yaxes(gridcolor=COLORS["grid"])
    fig.update_xaxes(gridcolor=COLORS["grid"])

    return fig


# ---------------------------------------------------------------------------
# Backtest Summary Markdown
# ---------------------------------------------------------------------------


def format_backtest_summary(bt: dict, ticker: str) -> str:
    """Return a Markdown summary of backtest results."""
    outperform = bt["strategy_return_pct"] - bt["buy_hold_return_pct"]

    return f"""
### Backtest Results for {ticker}

| Metric | Value |
|--------|-------|
| Initial Capital | ${bt['initial_capital']:,.2f} |
| Final Portfolio Value | ${bt['final_value']:,.2f} |
| **Strategy Return** | **{bt['strategy_return_pct']:+.2f}%** |
| **Buy & Hold Return** | **{bt['buy_hold_return_pct']:+.2f}%** |
| **Outperformance** | **{outperform:+.2f}%** |
| Total Trades | {bt['total_trades']} |
| Winning Trades | {bt['winning_trades']} |
| Win Rate | {bt['win_rate_pct']:.1f}% |

---

*Starting capital: $10,000. Strategy goes fully invested on BUY signals and exits to cash on SELL signals. No transaction costs or slippage modeled.*
"""


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------


def analyze(ticker: str, timeframe: str):
    """Run the full analysis pipeline and return outputs for all three tabs.

    Returns:
        main_chart: Plotly figure for the Charts tab.
        signal_table: DataFrame for the Signals tab.
        backtest_chart: Plotly figure for the Backtest tab.
        backtest_summary: Markdown string for the Backtest tab.
        status: Status message string.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        empty_fig = go.Figure()
        empty_fig.update_layout(**_PLOTLY_LAYOUT_DEFAULTS, height=400)
        return (
            empty_fig,
            pd.DataFrame(columns=["Date", "Close", "Type", "Reason"]),
            empty_fig,
            "Please enter a valid ticker symbol.",
            "Enter a ticker to begin.",
        )

    # Fetch data
    df, error = fetch_stock_data(ticker, timeframe)
    if error:
        empty_fig = go.Figure()
        empty_fig.update_layout(**_PLOTLY_LAYOUT_DEFAULTS, height=400)
        return (
            empty_fig,
            pd.DataFrame(columns=["Date", "Close", "Type", "Reason"]),
            empty_fig,
            f"**Error:** {error}",
            f"Error: {error}",
        )

    # Compute indicators and signals
    df = compute_indicators(df)
    df = generate_signals(df)

    # Build outputs
    main_chart = build_main_chart(df, ticker)
    signal_table = build_signal_table(df)
    bt = run_backtest(df)
    backtest_chart = build_backtest_chart(df, bt, ticker)
    backtest_summary = format_backtest_summary(bt, ticker)

    n_buys = int((df["Signal"] == 1).sum())
    n_sells = int((df["Signal"] == -1).sum())
    latest_close = df["Close"].iloc[-1]
    latest_rsi = df["RSI"].iloc[-1]

    status = (
        f"**{ticker}** | Last Close: ${latest_close:.2f} | "
        f"RSI: {latest_rsi:.1f} | "
        f"Signals: {n_buys} buys, {n_sells} sells ({timeframe})"
    )

    return main_chart, signal_table, backtest_chart, backtest_summary, status


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    css = """
    .disclaimer {
        background-color: #2a1a1a;
        border-left: 4px solid #ff6b6b;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        font-size: 0.85em;
    }
    .status-bar {
        background-color: #1a1d29;
        padding: 10px 16px;
        border-radius: 6px;
        border: 1px solid #2a2d3a;
        font-size: 0.95em;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="Trading Signal Dashboard",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.purple,
            secondary_hue=gr.themes.colors.pink,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=css,
    ) as app:
        # Header
        gr.Markdown(
            "# Trading Signal Dashboard\n"
            "Real-time technical analysis with automated signal generation & backtesting"
        )
        gr.Markdown(DISCLAIMER_TEXT, elem_classes=["disclaimer"])

        # Controls
        with gr.Row():
            ticker_input = gr.Dropdown(
                choices=DEFAULT_TICKERS,
                value="AAPL",
                label="Stock Ticker",
                allow_custom_value=True,
                info="Select a preset or type any valid ticker symbol",
                scale=2,
            )
            timeframe_input = gr.Dropdown(
                choices=list(TIMEFRAME_MAP.keys()),
                value="6 Months",
                label="Timeframe",
                scale=1,
            )
            analyze_btn = gr.Button(
                "Analyze",
                variant="primary",
                scale=1,
            )

        # Status bar
        status_output = gr.Markdown(
            "Enter a ticker and click **Analyze** to begin.",
            elem_classes=["status-bar"],
        )

        # Tabbed outputs
        with gr.Tabs():
            with gr.TabItem("Charts", id="charts"):
                chart_output = gr.Plot(label="Technical Analysis Chart")

            with gr.TabItem("Signals", id="signals"):
                gr.Markdown("### Recent Trading Signals")
                gr.Markdown(
                    "Signals are generated from SMA crossovers, MACD crossovers, "
                    "and RSI overbought/oversold exits."
                )
                signal_table_output = gr.Dataframe(
                    headers=["Date", "Close", "Type", "Reason"],
                    label="Signal Log",
                    wrap=True,
                )

            with gr.TabItem("Backtest Results", id="backtest"):
                gr.Markdown("### Strategy Backtest")
                gr.Markdown(
                    "Simulates following the generated signals with a $10,000 starting "
                    "portfolio and compares against a simple buy-and-hold strategy."
                )
                backtest_summary_output = gr.Markdown()
                backtest_chart_output = gr.Plot(label="Equity Curve")

        # Wire up the button
        analyze_btn.click(
            fn=analyze,
            inputs=[ticker_input, timeframe_input],
            outputs=[
                chart_output,
                signal_table_output,
                backtest_chart_output,
                backtest_summary_output,
                status_output,
            ],
        )

        # Also trigger on dropdown change for quick exploration
        ticker_input.change(
            fn=analyze,
            inputs=[ticker_input, timeframe_input],
            outputs=[
                chart_output,
                signal_table_output,
                backtest_chart_output,
                backtest_summary_output,
                status_output,
            ],
        )
        timeframe_input.change(
            fn=analyze,
            inputs=[ticker_input, timeframe_input],
            outputs=[
                chart_output,
                signal_table_output,
                backtest_chart_output,
                backtest_summary_output,
                status_output,
            ],
        )

        # Footer
        gr.Markdown(
            "---\n"
            "Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys) | "
            "Data from Yahoo Finance | "
            "Not financial advice"
        )

    return app


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.launch()
