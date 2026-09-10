import numpy as np
import pandas as pd

from src.indicators import bollinger, macd, rsi_wilder, sma


def _close(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="Close")


def test_sma_known_window():
    close = _close([1, 2, 3, 4, 5])
    result = sma(close, 3)
    assert np.isclose(result.iloc[-1], 4.0)


def test_rsi_bounds_and_uptrend():
    close = _close([float(i) for i in range(1, 40)])
    rsi = rsi_wilder(close, 14)
    assert rsi.dropna().between(0, 100).all()
    assert rsi.iloc[-1] > 70


def test_macd_positive_in_uptrend():
    close = _close([100 + i * 0.8 for i in range(80)])
    line, signal, hist = macd(close)
    assert line.dropna().iloc[-1] > 0
    assert not np.isnan(hist.dropna().iloc[-1])


def test_bollinger_uses_population_std():
    close = _close([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] * 3)
    upper, mid, lower = bollinger(close, window=5, num_std=2)
    window = close.iloc[-5:]
    expected_mid = window.mean()
    expected_std = window.std(ddof=0)
    assert np.isclose(mid.iloc[-1], expected_mid)
    assert np.isclose(upper.iloc[-1], expected_mid + 2 * expected_std)
    assert np.isclose(lower.iloc[-1], expected_mid - 2 * expected_std)
