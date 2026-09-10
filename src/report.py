"""Render a self-contained one-page HTML research brief."""

from __future__ import annotations

import base64
import io
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import project_root


def render_brief(
    ticker: str,
    frame: pd.DataFrame,
    snapshot: dict[str, Any],
    headlines: list[dict[str, str]],
    signal: dict[str, Any],
) -> Path:
    chart = _chart_base64(frame, ticker)
    html = _html(ticker, snapshot, headlines, signal, chart)
    out_dir = project_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_brief.html"
    path.write_text(html, encoding="utf-8")
    return path


def _chart_base64(frame: pd.DataFrame, ticker: str) -> str:
    plot_df = frame.tail(180)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(plot_df.index, plot_df["Close"], color="#111", label="Close")
    axes[0].plot(plot_df.index, plot_df["bb_upper"], color="#888", linewidth=0.8)
    axes[0].plot(plot_df.index, plot_df["bb_mid"], color="#555", linewidth=0.8)
    axes[0].plot(plot_df.index, plot_df["bb_lower"], color="#888", linewidth=0.8)
    axes[0].set_title(f"{ticker} price and Bollinger Bands")
    axes[0].legend(loc="upper left")

    axes[1].plot(plot_df.index, plot_df["rsi_14"], color="#0b6")
    axes[1].axhline(70, color="#c33", linewidth=0.7)
    axes[1].axhline(30, color="#c33", linewidth=0.7)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("RSI 14")

    axes[2].plot(plot_df.index, plot_df["macd"], label="MACD")
    axes[2].plot(plot_df.index, plot_df["macd_signal"], label="Signal")
    axes[2].bar(plot_df.index, plot_df["macd_hist"], color="#999", width=1.0)
    axes[2].set_title("MACD")
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _html(
    ticker: str,
    snapshot: dict[str, Any],
    headlines: list[dict[str, str]],
    signal: dict[str, Any],
    chart: str,
) -> str:
    news = "".join(
        f"<li>{item.get('date', '')} — {item.get('title', '')} <em>({item.get('publisher', '')})</em></li>"
        for item in headlines
    )
    risks = "".join(f"<li>{risk}</li>" for risk in signal.get("risks", []))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{ticker} equity research brief</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 860px; margin: 32px auto; color: #111; }}
    h1, h2 {{ font-family: Arial, sans-serif; }}
    .meta {{ color: #555; }}
    .signal {{ font-size: 1.4rem; font-weight: 700; }}
    img {{ width: 100%; }}
    .disclaimer {{ font-size: 0.85rem; color: #444; border-top: 1px solid #ddd; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>{ticker} one-page research brief</h1>
  <p class="meta">Generated {date.today().isoformat()} · last bar {snapshot.get("date")}</p>
  <p class="signal">Signal: {signal.get("signal")} · confidence {signal.get("confidence")} · {signal.get("horizon")}</p>
  <h2>Market snapshot</h2>
  <p>Close {snapshot.get("close"):.2f} · SMA20 {snapshot.get("sma_20"):.2f} · SMA50 {snapshot.get("sma_50"):.2f}
     · RSI {snapshot.get("rsi_14"):.1f} · MACD hist {snapshot.get("macd_hist"):.3f}
     · bias {snapshot.get("momentum_bias")}</p>
  <img alt="Technical charts" src="data:image/png;base64,{chart}" />
  <h2>Thesis</h2>
  <p>{signal.get("rationale")}</p>
  <h3>Bull case</h3>
  <p>{signal.get("bull_case")}</p>
  <h3>Bear case</h3>
  <p>{signal.get("bear_case")}</p>
  <h3>Risks</h3>
  <ul>{risks}</ul>
  <h2>Headlines used</h2>
  <ul>{news}</ul>
  <p class="disclaimer">This document is a technical demonstration for an ML engineering
  assessment. It is not investment advice, a solicitation, or a recommendation to buy
  or sell any security. Do not make financial decisions from this output.</p>
</body>
</html>
"""
