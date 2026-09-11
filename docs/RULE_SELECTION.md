# Technical rule selection

Generated 2026-09-11 for **AAPL**.

## Method

Three interpretable momentum rules were compared on the **last 126 trading sessions (holdout)** only. In-sample data was not used to pick thresholds. Ranking criteria:

1. **BUY-SELL spread** — mean 10d forward return after BUY minus after SELL (only when both sides have n>=5)
2. **Strategy Sharpe** on holdout (daily long/flat/short from rule)
3. **Excess return** vs buy-and-hold

## Variants

- `baseline`: Strict AND filter (SMA + MACD>0 + RSI 45-70)
- `acceleration`: Trend + rising MACD histogram + RSI 50-65
- `score` **(selected)**: Confluence score >=4/5 (SMA, MACD, MACD rising, RSI, vs BB mid)

## Holdout comparison

| variant | description | buy_n | sell_n | buy_mean_fwd | sell_mean_fwd | buy_sell_spread | buy_hit_rate | sell_hit_rate | strategy_sharpe | excess_return | strategy_total_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Strict AND filter (SMA + MACD>0 + RSI 45-70) | 48 | 0 | 2.97% | n/a | 2.97% | 79.2% | n/a | 2.630 | -4.34% | 29.50% |
| acceleration | Trend + rising MACD histogram + RSI 50-65 | 15 | 0 | 4.68% | n/a | 4.68% | 100.0% | n/a | 1.385 | -28.02% | 5.82% |
| score | Confluence score >=4/5 (SMA, MACD, MACD rising, RSI, vs BB mid) | 52 | 14 | 2.87% | 0.81% | 2.07% | 78.8% | 28.6% | 0.689 | -27.50% | 6.35% |

## Decision

**Selected variant:** `score`

- Description: Confluence score >=4/5 (SMA, MACD, MACD rising, RSI, vs BB mid)
- BUY–SELL spread (10d): 2.07%
- Holdout Sharpe: 0.689
- Excess vs buy-and-hold: -27.50%
- Directional sample sizes: BUY n=52, SELL n=14

### Why this variant

`score` ranked first on holdout because it showed the strongest BUY-SELL forward-return separation (spread 2.07%) with at least 5 samples on each side. Other variants lacked enough SELL samples for a two-sided spread comparison.

### Caveats

- Single ticker and one holdout window; treat as **diagnostic**, not proof of edge.
- Small SELL counts make spread and hit rates unstable.
- The LLM agent is still evaluated on output quality; this choice affects the deterministic rule baseline only.

Active default in `src/config.py`: `score`
