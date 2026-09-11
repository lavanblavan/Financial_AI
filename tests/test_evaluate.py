import numpy as np
import pandas as pd

from src.evaluate import add_forward_returns, evaluate_llm_output, walk_forward_table
from src.indicators import add_indicators


def test_walk_forward_buy_positive_in_noisy_uptrend():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(0.08 + rng.normal(0, 0.35, 400)), index=idx)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        }
    )
    scored = add_forward_returns(add_indicators(prices, rule_variant="baseline"), horizons=(5,))
    table = walk_forward_table(scored, horizon=5)
    buy = table.loc[table["signal"] == "BUY"].iloc[0]
    assert int(buy["n"]) > 0
    assert buy["mean_fwd_return"] > 0


def test_llm_schema_and_agreement():
    snapshot = {"momentum_bias": "bullish", "sma_cross": "bullish", "rsi_14": 55}
    result = {
        "signal": "BUY",
        "confidence": 0.7,
        "rationale": "SMA-50 above SMA-200 with constructive RSI.",
        "risks": ["macro", "earnings"],
    }
    report = evaluate_llm_output(snapshot, result)
    assert report["schema_ok"] is True
    assert report["agrees_with_rule"] is True
    assert report["overbought_buy_conflict"] is False
