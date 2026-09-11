"""Fetch recent headlines without a paid news API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests


def fetch_news_asof(
    ticker: str,
    asof,
    lookback_days: int = 14,
    min_items: int = 8,
) -> list[dict[str, str]]:
    """Headlines dated in a window ending on asof. Not a paid news archive."""
    asof_ts = pd.Timestamp(asof)
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_convert(None)
    start = (asof_ts - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = asof_ts.strftime("%Y-%m-%d")
    query = f"{ticker} stock after:{start} before:{end}"
    return _dedupe(_from_google_news(ticker, query=query), min_items)


def fetch_news(ticker: str, min_items: int = 10) -> list[dict[str, str]]:
    headlines = _from_yfinance(ticker)
    if len(headlines) < min_items:
        headlines.extend(_from_google_news(ticker))
    return _dedupe(headlines, min_items)


def _dedupe(headlines: list[dict[str, str]], min_items: int) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in headlines:
        title = item["title"].strip()
        key = title.lower()
        if not title or key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= min_items:
            break
    return unique


def _from_yfinance(ticker: str) -> list[dict[str, str]]:
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items: list[dict[str, str]] = []
    for row in raw:
        title = _headline_title(row)
        if not title:
            continue
        items.append(
            {
                "title": title,
                "publisher": _nested(row, "publisher") or _nested(row, "content", "provider", "displayName") or "Yahoo",
                "date": _format_ts(row),
                "source": "yfinance",
            }
        )
    return items


def _from_google_news(ticker: str, query: str | None = None) -> list[dict[str, str]]:
    query = quote_plus(query or f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return []

    parsed = feedparser.parse(response.content)
    items: list[dict[str, str]] = []
    for entry in parsed.entries:
        title = str(getattr(entry, "title", "")).strip()
        if not title:
            continue
        items.append(
            {
                "title": title,
                "publisher": str(getattr(entry, "source", {}).get("title", "Google News"))
                if isinstance(getattr(entry, "source", None), dict)
                else "Google News",
                "date": str(getattr(entry, "published", ""))[:16],
                "source": "google-news-rss",
            }
        )
    return items


def _headline_title(row: dict[str, Any]) -> str:
    return (
        _nested(row, "title")
        or _nested(row, "content", "title")
        or _nested(row, "content", "contentTitle")
        or ""
    )


def _nested(row: dict[str, Any], *keys: str) -> str | None:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    text = str(current).strip()
    return text or None


def _format_ts(row: dict[str, Any]) -> str:
    ts = row.get("providerPublishTime") or _nested(row, "content", "pubDate")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    if ts:
        return str(ts)[:16]
    return ""
