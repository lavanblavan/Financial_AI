"""Evaluate the technical rule and the live LLM output.

The walk-forward test scores the deterministic momentum rule, not the LLM.
The LLM is scored for schema validity and agreement with that rule.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.indicators import RULE_VARIANT_LABELS, RULE_VARIANTS, momentum_bias

RULE_TO_SIGNAL = {"bullish": "BUY", "bearish": "SELL", "mixed": "HOLD"}
SIGNAL_POSITION = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}


def apply_rule_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Attach momentum_bias for a rule variant and map to BUY/HOLD/SELL."""
    out = frame.copy()
    out["momentum_bias"] = momentum_bias(out, variant=variant)
    out["rule_signal"] = out["momentum_bias"].map(RULE_TO_SIGNAL)
    return out


def add_forward_returns(frame: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    out = frame.copy()
    close = out["Close"]
    for horizon in horizons:
        out[f"fwd_{horizon}d"] = close.shift(-horizon) / close - 1.0
    if "rule_signal" not in out.columns:
        out["rule_signal"] = out["momentum_bias"].map(RULE_TO_SIGNAL)
    return out


def walk_forward_table(frame: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Mean next-horizon return and hit rate by BUY / HOLD / SELL."""
    col = f"fwd_{horizon}d"
    ready = frame.dropna(subset=["rule_signal", col, "sma_200"])
    rows: list[dict[str, Any]] = []
    for signal in ("BUY", "HOLD", "SELL"):
        sample = ready.loc[ready["rule_signal"] == signal, col]
        n = int(sample.shape[0])
        if n == 0:
            rows.append(
                {
                    "signal": signal,
                    "n": 0,
                    "mean_fwd_return": None,
                    "hit_rate": None,
                }
            )
            continue
        if signal == "BUY":
            hit = float((sample > 0).mean())
        elif signal == "SELL":
            hit = float((sample < 0).mean())
        else:
            hit = None
        rows.append(
            {
                "signal": signal,
                "n": n,
                "mean_fwd_return": float(sample.mean()),
                "hit_rate": hit,
            }
        )
    return pd.DataFrame(rows)


def holdout_split(frame: pd.DataFrame, holdout_bars: int = 126) -> tuple[pd.DataFrame, pd.DataFrame]:
    ready = frame.dropna(subset=["sma_200", "momentum_bias"])
    if len(ready) <= holdout_bars + 20:
        return ready, ready
    return ready.iloc[:-holdout_bars], ready.iloc[-holdout_bars:]


def strategy_equity(frame: pd.DataFrame) -> pd.DataFrame:
    """Next-day long/flat/short from the rule. Signal at close t, return on t+1."""
    ready = frame.dropna(subset=["rule_signal", "Close", "sma_200"]).copy()
    daily = ready["Close"].pct_change()
    position = ready["rule_signal"].map(SIGNAL_POSITION).shift(1)
    strategy = (position * daily).fillna(0.0)
    buy_hold = daily.fillna(0.0)
    out = pd.DataFrame(
        {
            "strategy": (1.0 + strategy).cumprod(),
            "buy_hold": (1.0 + buy_hold).cumprod(),
            "daily_strategy": strategy,
            "daily_buy_hold": buy_hold,
        },
        index=ready.index,
    )
    return out


def strategy_stats(equity: pd.DataFrame) -> dict[str, float]:
    strat = equity["daily_strategy"]
    bh = equity["daily_buy_hold"]
    return {
        "strategy_total_return": float(equity["strategy"].iloc[-1] - 1.0),
        "buy_hold_total_return": float(equity["buy_hold"].iloc[-1] - 1.0),
        "strategy_sharpe": _sharpe(strat),
        "buy_hold_sharpe": _sharpe(bh),
        "excess_return": float(equity["strategy"].iloc[-1] - equity["buy_hold"].iloc[-1]),
    }


def _sharpe(daily: pd.Series) -> float:
    std = float(daily.std())
    if std == 0:
        return 0.0
    return float((daily.mean() / std) * (252 ** 0.5))


def _table_metric(table: pd.DataFrame, signal: str, column: str) -> float | None:
    row = table.loc[table["signal"] == signal]
    if row.empty or int(row.iloc[0]["n"]) == 0:
        return None
    value = row.iloc[0][column]
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)


def compare_rule_variants(
    indicator_frame: pd.DataFrame,
    holdout_bars: int = 126,
    horizon: int = 10,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Score each rule variant on holdout only; return comparison table and winner."""
    rows: list[dict[str, Any]] = []

    for variant in RULE_VARIANTS:
        variant_frame = apply_rule_variant(indicator_frame, variant)
        variant_scored = add_forward_returns(variant_frame)
        _, holdout = holdout_split(variant_scored, holdout_bars=holdout_bars)
        table = walk_forward_table(holdout, horizon=horizon)
        equity = strategy_equity(holdout)
        stats = strategy_stats(equity)

        buy_mean = _table_metric(table, "BUY", "mean_fwd_return")
        sell_mean = _table_metric(table, "SELL", "mean_fwd_return")
        buy_n = int(table.loc[table["signal"] == "BUY", "n"].iloc[0]) if not table.empty else 0
        sell_n = int(table.loc[table["signal"] == "SELL", "n"].iloc[0]) if not table.empty else 0
        buy_hit = _table_metric(table, "BUY", "hit_rate")
        sell_hit = _table_metric(table, "SELL", "hit_rate")

        if buy_mean is not None and sell_mean is not None:
            buy_sell_spread = buy_mean - sell_mean
        elif buy_mean is not None:
            buy_sell_spread = buy_mean
        elif sell_mean is not None:
            buy_sell_spread = -sell_mean
        else:
            buy_sell_spread = float("-inf")

        rows.append(
            {
                "variant": variant,
                "description": RULE_VARIANT_LABELS[variant],
                "buy_n": buy_n,
                "sell_n": sell_n,
                "buy_mean_fwd": buy_mean,
                "sell_mean_fwd": sell_mean,
                "buy_sell_spread": buy_sell_spread if buy_sell_spread != float("-inf") else None,
                "buy_hit_rate": buy_hit,
                "sell_hit_rate": sell_hit,
                "strategy_sharpe": stats["strategy_sharpe"],
                "excess_return": stats["excess_return"],
                "strategy_total_return": stats["strategy_total_return"],
            }
        )

    comparison = pd.DataFrame(rows)
    winner, rationale = select_rule_variant(comparison)
    return comparison, winner, rationale


def select_rule_variant(
    comparison: pd.DataFrame,
    min_directional: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Pick variant with best holdout BUY-SELL spread when both sides have enough samples.

    If no variant has sufficient BUY and SELL counts, fall back to holdout Sharpe, then excess return.
    """
    ranked = comparison.copy()

    def spread_score(row: pd.Series) -> float:
        if int(row["buy_n"]) < min_directional or int(row["sell_n"]) < min_directional:
            return float("-inf")
        spread = row["buy_sell_spread"]
        if spread is None or (isinstance(spread, float) and pd.isna(spread)):
            return float("-inf")
        return float(spread)

    ranked["_spread_sort"] = ranked.apply(spread_score, axis=1)
    ranked = ranked.sort_values(
        by=["_spread_sort", "strategy_sharpe", "excess_return"],
        ascending=[False, False, False],
    )
    winner = str(ranked.iloc[0]["variant"])
    top = ranked.iloc[0]
    used_spread = float(top["_spread_sort"]) > float("-inf")
    rationale = {
        "selected_variant": winner,
        "description": str(top["description"]),
        "buy_sell_spread": top["buy_sell_spread"],
        "strategy_sharpe": float(top["strategy_sharpe"]),
        "excess_return": float(top["excess_return"]),
        "buy_n": int(top["buy_n"]),
        "sell_n": int(top["sell_n"]),
        "used_buy_sell_spread": used_spread,
        "min_directional": min_directional,
        "ranking": ranked[["variant", "buy_sell_spread", "strategy_sharpe", "excess_return"]].to_dict("records"),
    }
    return winner, rationale


def evaluate_llm_output(snapshot: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Structural checks on the live LLM JSON. Not a claim about alpha."""
    signal = str(result.get("signal", "")).upper()
    rule = RULE_TO_SIGNAL.get(str(snapshot.get("momentum_bias", "")), "HOLD")
    confidence = result.get("confidence")
    try:
        conf_ok = 0.0 <= float(confidence) <= 1.0
    except (TypeError, ValueError):
        conf_ok = False

    schema_ok = signal in {"BUY", "HOLD", "SELL"} and conf_ok
    agrees_with_rule = signal == rule
    conflict_hold = (
        snapshot.get("sma_cross") == "bearish"
        and float(snapshot.get("rsi_14", 50)) > 70
        and signal == "BUY"
    )
    return {
        "schema_ok": schema_ok,
        "signal": signal,
        "rule_signal": rule,
        "agrees_with_rule": agrees_with_rule,
        "overbought_buy_conflict": conflict_hold,
        "has_rationale": bool(str(result.get("rationale", "")).strip()),
        "n_risks": len(result.get("risks") or []),
    }
