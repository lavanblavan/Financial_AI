"""Technical indicators from first principles. No TA-Lib."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    close = out["Close"]

    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["rsi_14"] = rsi_wilder(close, 14)
    macd_line, signal, hist = macd(close, 12, 26, 9)
    out["macd"] = macd_line
    out["macd_signal"] = signal
    out["macd_hist"] = hist
    upper, mid, lower = bollinger(close, 20, 2.0)
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["momentum_bias"] = _momentum_bias(out)
    return out


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI: seed with SMA, then smooth with alpha=1/period."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi.clip(0, 100)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    signal = line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    hist = line - signal
    return line, signal, hist


def bollinger(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    # Population stdev matches common charting defaults.
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def latest_snapshot(frame: pd.DataFrame) -> dict:
    row = frame.dropna(subset=["sma_20", "rsi_14", "macd"]).iloc[-1]
    prev = frame.dropna(subset=["sma_20", "sma_50"]).iloc[-2]

    return {
        "date": str(row.name.date()) if hasattr(row.name, "date") else str(row.name),
        "close": float(row["Close"]),
        "sma_20": float(row["sma_20"]),
        "sma_50": float(row["sma_50"]),
        "sma_cross": "bullish" if row["sma_20"] > row["sma_50"] else "bearish",
        "sma_cross_flipped": bool(
            (prev["sma_20"] > prev["sma_50"]) != (row["sma_20"] > row["sma_50"])
        ),
        "rsi_14": float(row["rsi_14"]),
        "macd": float(row["macd"]),
        "macd_signal": float(row["macd_signal"]),
        "macd_hist": float(row["macd_hist"]),
        "bb_upper": float(row["bb_upper"]),
        "bb_mid": float(row["bb_mid"]),
        "bb_lower": float(row["bb_lower"]),
        "bb_position": _bb_position(float(row["Close"]), float(row["bb_lower"]), float(row["bb_upper"])),
        "momentum_bias": str(row["momentum_bias"]),
    }


def _bb_position(close: float, lower: float, upper: float) -> str:
    width = upper - lower
    if width <= 0:
        return "unknown"
    pct = (close - lower) / width
    if pct > 0.8:
        return "near_upper"
    if pct < 0.2:
        return "near_lower"
    return "mid_band"


def _momentum_bias(frame: pd.DataFrame) -> pd.Series:
    bull = (
        (frame["sma_20"] > frame["sma_50"])
        & (frame["macd_hist"] > 0)
        & (frame["rsi_14"] >= 45)
        & (frame["rsi_14"] <= 70)
    )
    bear = (
        (frame["sma_20"] < frame["sma_50"])
        & (frame["macd_hist"] < 0)
        & (frame["rsi_14"] <= 55)
        & (frame["rsi_14"] >= 30)
    )
    out = pd.Series("mixed", index=frame.index, dtype="object")
    out = out.mask(bull, "bullish")
    out = out.mask(bear, "bearish")
    return out
