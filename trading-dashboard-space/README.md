---
title: Trading Signal Dashboard
emoji: 📈
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Technical analysis dashboard with trading signals
---

# Trading Signal Dashboard

An interactive technical analysis dashboard that fetches real-time stock data and generates trading signals using classical technical indicators.

## Features

- **Real-time data** via Yahoo Finance for any publicly traded ticker
- **Technical indicators**: SMA (20/50), EMA (12/26), RSI (14), MACD, Bollinger Bands
- **Signal generation**: Buy/sell signals from indicator crossovers (SMA, MACD, RSI)
- **Interactive Plotly charts**: Price with overlays, RSI, MACD, and Volume subplots
- **Signal summary table** with recent buy/sell signals and reasoning
- **Backtesting engine**: Compare signal-based strategy returns against buy-and-hold
- **Multiple timeframes**: 1 Month, 3 Months, 6 Months, 1 Year, 2 Years

## How It Works

1. Enter a stock ticker (e.g., AAPL, GOOGL, MSFT, TSLA)
2. Select a lookback timeframe
3. The dashboard fetches historical price data, computes technical indicators, and identifies trading signals
4. Review charts, signals, and backtest results across three organized tabs

## Indicators & Signals

| Indicator | Signal Logic |
|-----------|-------------|
| SMA Crossover | Buy when SMA-20 crosses above SMA-50; Sell when SMA-20 crosses below SMA-50 |
| MACD Crossover | Buy when MACD line crosses above Signal line; Sell on downward cross |
| RSI Extremes | Buy when RSI crosses above 30 (oversold); Sell when RSI crosses below 70 (overbought) |

## Disclaimer

**This application is for educational and informational purposes only. It does NOT constitute financial advice, investment recommendations, or solicitation to buy or sell any securities.** Past performance and backtesting results do not guarantee future returns. Always consult a qualified financial advisor before making investment decisions. The creator assumes no liability for any financial losses incurred from using this tool.

## Author

Built by [Lorenzo Scaturchio (gr8monk3ys)](https://huggingface.co/gr8monk3ys)
