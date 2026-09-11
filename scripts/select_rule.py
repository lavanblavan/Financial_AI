"""Compare momentum rule variants on holdout and write docs/RULE_SELECTION.md."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_RULE_VARIANT, load_settings, project_root
from src.data import fetch_prices
from src.evaluate import compare_rule_variants
from src.indicators import RULE_VARIANT_LABELS, add_indicators


def _comparison_table(comparison: pd.DataFrame) -> str:
    display = comparison.copy()
    for col in ("buy_mean_fwd", "sell_mean_fwd", "buy_sell_spread", "excess_return", "strategy_total_return"):
        if col in display.columns:
            display[col] = display[col].map(_pct)
    for col in ("buy_hit_rate", "sell_hit_rate"):
        if col in display.columns:
            display[col] = display[col].map(lambda v: "n/a" if v is None or pd.isna(v) else f"{v:.1%}")
    if "strategy_sharpe" in display.columns:
        display["strategy_sharpe"] = display["strategy_sharpe"].map(lambda v: f"{v:.3f}")
    cols = display.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in display.iterrows()]
    return "\n".join([header, sep, *body])


def _pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.2%}"


def render_markdown(
    ticker: str,
    comparison: pd.DataFrame,
    winner: str,
    rationale: dict,
    holdout_bars: int,
    horizon: int,
) -> str:
    lines = [
        "# Technical rule selection",
        "",
        f"Generated {date.today().isoformat()} for **{ticker}**.",
        "",
        "## Method",
        "",
        "Three interpretable momentum rules were compared on the **last "
        f"{holdout_bars} trading sessions (holdout)** only. In-sample data was not used "
        "to pick thresholds. Ranking criteria:",
        "",
        f"1. **BUY-SELL spread** — mean {horizon}d forward return after BUY minus after SELL (only when both sides have n>={5})",
        "2. **Strategy Sharpe** on holdout (daily long/flat/short from rule)",
        "3. **Excess return** vs buy-and-hold",
        "",
        "## Variants",
        "",
    ]
    for name, label in RULE_VARIANT_LABELS.items():
        marker = " **(selected)**" if name == winner else ""
        lines.append(f"- `{name}`{marker}: {label}")
    lines.extend(["", "## Holdout comparison", ""])
    lines.append(_comparison_table(comparison))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**Selected variant:** `{winner}`",
            "",
            f"- Description: {rationale['description']}",
            f"- BUY–SELL spread (10d): {_pct(rationale.get('buy_sell_spread'))}",
            f"- Holdout Sharpe: {rationale['strategy_sharpe']:.3f}",
            f"- Excess vs buy-and-hold: {_pct(rationale.get('excess_return'))}",
            f"- Directional sample sizes: BUY n={rationale['buy_n']}, SELL n={rationale['sell_n']}",
            "",
            "### Why this variant",
            "",
            _why_paragraph(winner, rationale, comparison),
            "",
            "### Caveats",
            "",
            "- Single ticker and one holdout window; treat as **diagnostic**, not proof of edge.",
            "- Small SELL counts make spread and hit rates unstable.",
            "- The LLM agent is still evaluated on output quality; this choice affects the deterministic rule baseline only.",
            "",
            f"Active default in `src/config.py`: `{DEFAULT_RULE_VARIANT}`",
            "",
        ]
    )
    return "\n".join(lines)


def _why_paragraph(winner: str, rationale: dict, comparison: pd.DataFrame) -> str:
    spread = rationale.get("buy_sell_spread")
    sharpe = rationale["strategy_sharpe"]
    others = comparison.loc[comparison["variant"] != winner]
    min_n = rationale.get("min_directional", 5)
    if rationale.get("used_buy_sell_spread") and spread is not None and not pd.isna(spread):
        lead = (
            f"`{winner}` ranked first on holdout because it showed the strongest BUY-SELL forward-return "
            f"separation (spread {_pct(spread)}) with at least {min_n} samples on each side"
        )
    else:
        lead = (
            f"`{winner}` ranked first on holdout Sharpe ({sharpe:.3f}) because no variant had "
            f"enough BUY and SELL samples (>={min_n}) for a reliable spread comparison"
        )
    if not others.empty:
        credible = others[
            (others["buy_n"] >= rationale.get("min_directional", 5))
            & (others["sell_n"] >= rationale.get("min_directional", 5))
        ]
        if not credible.empty:
            alt = credible.sort_values("buy_sell_spread", ascending=False).iloc[0]
            lead += (
                f", ahead of `{alt['variant']}` (spread {_pct(alt.get('buy_sell_spread'))}, "
                f"Sharpe {alt['strategy_sharpe']:.3f})."
            )
        else:
            lead += ". Other variants lacked enough SELL samples for a two-sided spread comparison."
    else:
        lead += "."
    return lead


def main() -> None:
    settings = load_settings()
    ticker = settings.ticker
    prices = fetch_prices(ticker, period=settings.lookback)
    indicators = add_indicators(prices, rule_variant="baseline")

    comparison, winner, rationale = compare_rule_variants(indicators, holdout_bars=126, horizon=10)

    print(f"Rule variant comparison — {ticker} (holdout last 126 sessions, 10d horizon)\n")
    print(comparison.to_string(index=False))
    print(f"\nSelected variant: {winner}")
    print(json.dumps(rationale, indent=2, default=str))

    doc_path = project_root() / "docs" / "RULE_SELECTION.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        render_markdown(ticker, comparison, winner, rationale, holdout_bars=126, horizon=10),
        encoding="utf-8",
    )
    print(f"\nWrote {doc_path}")

    if winner != DEFAULT_RULE_VARIANT:
        print(
            f"\nNote: config DEFAULT_RULE_VARIANT is '{DEFAULT_RULE_VARIANT}' "
            f"but holdout winner is '{winner}'. Update src/config.py if you want them aligned."
        )


if __name__ == "__main__":
    main()
