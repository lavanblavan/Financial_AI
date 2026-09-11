import numpy as np
import pandas as pd

from src.evaluate import apply_rule_variant, compare_rule_variants, select_rule_variant
from src.indicators import RULE_VARIANTS, add_indicators, momentum_bias


def _uptrend_prices(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(0.08 + rng.normal(0, 0.35, n)), index=idx)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        }
    )


def test_all_rule_variants_produce_bias_labels():
    frame = add_indicators(_uptrend_prices(), rule_variant="baseline")
    for variant in RULE_VARIANTS:
        bias = momentum_bias(frame, variant=variant)
        assert set(bias.dropna().unique()).issubset({"bullish", "bearish", "mixed"})


def test_apply_rule_variant_sets_rule_signal():
    frame = add_indicators(_uptrend_prices(), rule_variant="baseline")
    scored = apply_rule_variant(frame, "score")
    assert scored["rule_signal"].isin(["BUY", "HOLD", "SELL"]).all()


def test_select_rule_variant_prefers_higher_spread():
    comparison = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "buy_sell_spread": 0.01,
                "strategy_sharpe": 0.5,
                "excess_return": 0.02,
                "description": "a",
                "buy_n": 10,
                "sell_n": 5,
            },
            {
                "variant": "acceleration",
                "buy_sell_spread": 0.03,
                "strategy_sharpe": 0.4,
                "excess_return": 0.01,
                "description": "b",
                "buy_n": 8,
                "sell_n": 6,
            },
        ]
    )
    winner, _ = select_rule_variant(comparison)
    assert winner == "acceleration"


def test_compare_rule_variants_returns_all_variants():
    frame = add_indicators(_uptrend_prices(), rule_variant="baseline")
    comparison, winner, rationale = compare_rule_variants(frame, holdout_bars=60, horizon=5)
    assert set(comparison["variant"]) == set(RULE_VARIANTS)
    assert winner in RULE_VARIANTS
    assert rationale["selected_variant"] == winner
