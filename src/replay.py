"""Replay past dates: decide with data known then, score vs what price did next."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.config import Settings, project_root
from src.evaluate import RULE_TO_SIGNAL
from src.indicators import add_indicators, latest_snapshot
from src.news import fetch_news_asof, fetch_news_window
from src.signal import GroqAPIError, LLMNotConfiguredError, build_signal


def replay_history(
    prices: pd.DataFrame,
    ticker: str,
    settings: Settings | None = None,
    step: int = 21,
    horizon: int = 10,
    use_llm: bool = True,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """On each replay date, use only prices up to that day, plus dated headlines.

    Then compare the BUY/SELL call to the actual next `horizon` trading-day return.
    Pass start/end (YYYY-MM-DD) to focus on one month, e.g. May 2026.
    """
    frame = prices.copy()
    idx = pd.to_datetime(frame.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    frame.index = idx
    frame = frame.sort_index()

    positions = _decision_positions(frame, step=step, horizon=horizon, start=start, end=end)
    corpus: list[dict[str, str]] = []
    if start and end:
        news_start = pd.Timestamp(start) - pd.Timedelta(days=14)
        corpus = fetch_news_window(ticker, news_start, end, max_records=75)

    rows: list[dict[str, Any]] = []
    for pos in positions:
        asof = frame.index[pos]
        hist = frame.iloc[: pos + 1]
        scored = add_indicators(hist)
        snapshot = latest_snapshot(scored)
        headlines = fetch_news_asof(ticker, asof, corpus=corpus if start and end else None)
        decision = _decide(snapshot, headlines, settings, ticker, use_llm)
        future_close = float(frame.iloc[pos + horizon]["Close"])
        asof_close = float(frame.iloc[pos]["Close"])
        fwd = future_close / asof_close - 1.0
        signal = decision["signal"]
        rows.append(
            {
                "date": asof.date().isoformat(),
                "close": asof_close,
                "signal": signal,
                "rule_signal": RULE_TO_SIGNAL.get(str(snapshot.get("momentum_bias")), "HOLD"),
                "rsi_14": snapshot.get("rsi_14"),
                "sma_cross": snapshot.get("sma_cross"),
                "n_headlines": len(headlines),
                "headline": headlines[0]["title"] if headlines else "",
                "fwd_return": fwd,
                "actual_move": "UP" if fwd > 0 else "DOWN",
                "match": _is_match(signal, fwd),
                "source": decision["source"],
                "rationale": decision.get("rationale", ""),
            }
        )
    return pd.DataFrame(rows)


def _decision_positions(
    frame: pd.DataFrame,
    step: int,
    horizon: int,
    start: str | None,
    end: str | None,
) -> list[int]:
    last = len(frame) - horizon
    if start and end:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        picks = [
            i
            for i in range(len(frame))
            if i >= 200 and i < last and start_ts <= frame.index[i] <= end_ts
        ]
        if not picks:
            raise ValueError(f"No replay dates between {start} and {end}. Check that prices cover that month plus 10 later days.")
        return picks[:: max(step, 1)]

    first = step + 200
    if last <= first:
        raise ValueError("Not enough history. Need ~2y of prices for SMA-200 plus a 10-day future window.")
    return list(range(first, last, step))


def replay_summary(replay: pd.DataFrame) -> dict[str, Any]:
    directional = replay[replay["signal"].isin(["BUY", "SELL"])]
    n_dir = int(len(directional))
    return {
        "n_dates": int(len(replay)),
        "n_buy": int((replay["signal"] == "BUY").sum()),
        "n_sell": int((replay["signal"] == "SELL").sum()),
        "n_hold": int((replay["signal"] == "HOLD").sum()),
        "n_directional": n_dir,
        "directional_accuracy": float(directional["match"].mean()) if n_dir else None,
        "mean_return_after_buy": _mean_if(replay, "BUY"),
        "mean_return_after_sell": _mean_if(replay, "SELL"),
        "n_with_headlines": int((replay["n_headlines"] > 0).sum()),
    }


def render_replay_report(
    ticker: str,
    replay: pd.DataFrame,
    summary: dict[str, Any],
    horizon: int = 10,
) -> str:
    acc = summary.get("directional_accuracy")
    acc_txt = f"{acc:.0%}" if acc is not None else "n/a"
    rows = "".join(
        f"<tr class='{'ok' if row.match else 'miss'}'>"
        f"<td>{row.date}</td><td>{row.close:.2f}</td><td>{row.signal}</td>"
        f"<td>{row.rule_signal}</td><td>{row.n_headlines}</td>"
        f"<td>{row.fwd_return:.2%}</td><td>{row.actual_move}</td>"
        f"<td>{'Yes' if row.match else 'No'}</td>"
        f"<td>{_short(row.headline)}</td></tr>"
        for row in replay.itertuples()
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{ticker} May 2026 signal replay</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 32px auto; color: #111; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    tr.ok td:nth-child(8) {{ color: #0a7; font-weight: 700; }}
    tr.miss td:nth-child(8) {{ color: #c33; font-weight: 700; }}
    .note {{ color: #444; font-size: 0.9rem; }}
    .expect {{ background: #f7f7f7; padding: 12px 16px; }}
  </style>
</head>
<body>
  <h1>{ticker} out-of-sample replay (May 2026)</h1>
  <p>Each row is a decision date. Indicators are computed on prices available through that close.
  News is restricted to GDELT items with a seen-date in the prior 14 days. The realised column is the
  actual close {horizon} trading sessions later. There is no look-ahead in the feature set.</p>
  <div class="expect">
    <p><strong>How to interpret</strong></p>
    <ul>
      <li><strong>Directional accuracy</strong> uses BUY and SELL only. HOLD is a non-position and is omitted from this rate.</li>
      <li>A useful rule tends to show <em>higher mean next-{horizon}d return after BUY than after SELL</em>.</li>
      <li>If BUY+SELL counts are small, treat the percentage as descriptive, not a robust edge.</li>
      <li>A large HOLD share is consistent with a conservative filter, not automatically an error.</li>
    </ul>
  </div>
  <p><strong>Directional accuracy (BUY/SELL): {acc_txt}</strong>
     · decision dates {summary['n_dates']}
     · BUY {summary['n_buy']} · SELL {summary['n_sell']} · HOLD {summary['n_hold']}
     · dates with headlines {summary['n_with_headlines']}</p>
  <p>Mean realised return over the next {horizon} sessions — after BUY: {_pct(summary.get('mean_return_after_buy'))}
     · after SELL: {_pct(summary.get('mean_return_after_sell'))}</p>
  <table>
    <thead>
      <tr>
        <th>Date</th><th>Close</th><th>Signal</th><th>Technical rule</th>
        <th>Headlines</th><th>Next {horizon}d</th><th>Realised</th><th>Correct direction</th><th>Lead headline</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <p class="note">News coverage is GDELT (public dated index), not a licensed newswire. Respect a several-second
  pause between API re-runs. This report is a research diagnostic for an engineering assessment and is not
  investment advice. Generated {date.today().isoformat()}.</p>
</body>
</html>
"""
    path = project_root() / "outputs" / f"{ticker}_replay.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path)


def _decide(
    snapshot: dict[str, Any],
    headlines: list[dict[str, str]],
    settings: Settings | None,
    ticker: str,
    use_llm: bool,
) -> dict[str, str]:
    if use_llm and settings is not None and settings.llm_ready:
        try:
            result = build_signal(snapshot, headlines, settings, ticker)
            return {
                "signal": str(result["signal"]),
                "rationale": str(result.get("rationale", "")),
                "source": "llm",
            }
        except (LLMNotConfiguredError, GroqAPIError, Exception):
            pass
    return {
        "signal": RULE_TO_SIGNAL.get(str(snapshot.get("momentum_bias")), "HOLD"),
        "rationale": "Technical rule applied; language-model call skipped or unavailable.",
        "source": "rule",
    }


def _is_match(signal: str, fwd: float) -> bool:
    if signal == "BUY":
        return fwd > 0
    if signal == "SELL":
        return fwd < 0
    return False


def _mean_if(replay: pd.DataFrame, signal: str) -> float | None:
    sample = replay.loc[replay["signal"] == signal, "fwd_return"]
    if sample.empty:
        return None
    return float(sample.mean())


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _short(text: str, limit: int = 90) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"
