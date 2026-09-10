"""Seaborn panels plus interactive Plotly charts for each indicator."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")


def _window(frame: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    plot_df = frame.tail(lookback).copy()
    plot_df.index = pd.to_datetime(plot_df.index)
    plot_df.index.name = "Date"
    return plot_df


def _long(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        frame[columns]
        .rename_axis("Date")
        .reset_index()
        .melt(id_vars="Date", var_name="series", value_name="value")
    )


def plot_price_sma(frame: pd.DataFrame, ticker: str = "", lookback: int = 252) -> plt.Figure:
    plot_df = _window(frame, lookback)
    long_df = _long(plot_df, ["Close", "sma_50", "sma_200"])
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.lineplot(data=long_df, x="Date", y="value", hue="series", ax=ax)
    ax.set_title(f"{ticker} close vs SMA 50 / SMA 200".strip())
    ax.set_ylabel("Price")
    fig.tight_layout()
    return fig


def plot_bollinger(frame: pd.DataFrame, ticker: str = "", lookback: int = 252) -> plt.Figure:
    plot_df = _window(frame, lookback)
    long_df = _long(plot_df, ["Close", "bb_upper", "bb_mid", "bb_lower"])
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.lineplot(data=long_df, x="Date", y="value", hue="series", ax=ax)
    ax.set_title(f"{ticker} Bollinger Bands".strip())
    ax.set_ylabel("Price")
    fig.tight_layout()
    return fig


def plot_rsi(frame: pd.DataFrame, ticker: str = "", lookback: int = 252) -> plt.Figure:
    plot_df = _window(frame, lookback)
    long_df = _long(plot_df, ["rsi_14"])
    fig, ax = plt.subplots(figsize=(11, 3.5))
    sns.lineplot(data=long_df, x="Date", y="value", hue="series", ax=ax, legend=False)
    ax.axhline(70, color="crimson", linestyle="--", linewidth=1)
    ax.axhline(30, color="crimson", linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_title(f"{ticker} RSI 14".strip())
    ax.set_ylabel("RSI")
    fig.tight_layout()
    return fig


def plot_macd(frame: pd.DataFrame, ticker: str = "", lookback: int = 252) -> plt.Figure:
    plot_df = _window(frame, lookback)
    long_df = _long(plot_df, ["macd", "macd_signal"])
    fig, ax = plt.subplots(figsize=(11, 3.5))
    colors = ["#4c72b0" if v >= 0 else "#c44e52" for v in plot_df["macd_hist"]]
    ax.bar(plot_df.index, plot_df["macd_hist"], color=colors, width=1.0, alpha=0.45, label="Hist")
    sns.lineplot(data=long_df, x="Date", y="value", hue="series", ax=ax)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_title(f"{ticker} MACD".strip())
    ax.set_ylabel("MACD")
    fig.tight_layout()
    return fig


def plot_all_seaborn(frame: pd.DataFrame, ticker: str = "", lookback: int = 252) -> list[plt.Figure]:
    """One Seaborn figure per indicator."""
    return [
        plot_price_sma(frame, ticker, lookback),
        plot_bollinger(frame, ticker, lookback),
        plot_rsi(frame, ticker, lookback),
        plot_macd(frame, ticker, lookback),
    ]


def plot_interactive(frame: pd.DataFrame, ticker: str = "", lookback: int = 252):
    """Zoom, pan, and hover. Seaborn is static; Plotly provides the interaction."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot_df = _window(frame, lookback)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.34, 0.22, 0.22, 0.22],
        subplot_titles=(
            f"{ticker} price + SMA 50/200",
            "Bollinger Bands",
            "RSI 14",
            "MACD",
        ),
    )

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_50"], name="SMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_200"], name="SMA 200"), row=1, col=1)

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_upper"], name="BB upper", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Close (BB)", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_mid"], name="BB mid", line=dict(width=1, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_lower"], name="BB lower", line=dict(width=1)), row=2, col=1)

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi_14"], name="RSI 14"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="crimson", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="crimson", row=3, col=1)

    hist_colors = ["#4c72b0" if v >= 0 else "#c44e52" for v in plot_df["macd_hist"]]
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df["macd_hist"], name="MACD hist", marker_color=hist_colors, opacity=0.5),
        row=4,
        col=1,
    )
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd"], name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd_signal"], name="Signal"), row=4, col=1)

    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_layout(
        height=980,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=80, b=40),
    )
    fig.update_xaxes(rangeslider_visible=True, row=4, col=1)
    return fig
