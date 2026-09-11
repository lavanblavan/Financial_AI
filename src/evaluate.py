"""Evaluate the technical rule and the live LLM output.

The walk-forward test scores the deterministic momentum rule, not the LLM.
The LLM is scored for schema validity and agreement with that rule.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

RULE_TO_SIGNAL = {"bullish": "BUY", "bearish": "SELL", "mixed": "HOLD"}
SIGNAL_POSITION = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}


def add_forward_returns(frame: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    out = frame.copy()
    close = out["Close"]
    for horizon in horizons:
        out[f"fwd_{horizon}d"] = close.shift(-horizon) / close - 1.0
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
