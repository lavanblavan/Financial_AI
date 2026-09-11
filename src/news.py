"""Fetch recent headlines without a paid news API."""

from __future__ import annotations

import time
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
    corpus: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Headlines whose seen-date is in (asof - lookback, asof]. Prefer a preloaded corpus."""
    asof_ts = _naive_ts(asof)
    start = asof_ts - pd.Timedelta(days=lookback_days)
    pool = corpus if corpus is not None else fetch_news_window(ticker, start, asof_ts)
    in_window: list[dict[str, str]] = []
    for item in pool:
        seen = _parse_seen(item.get("date", ""))
        if seen is None or seen < start or seen > asof_ts:
            continue
        in_window.append(item)
    if in_window:
        return _dedupe(in_window, min_items)
    return _dedupe(_from_google_news(ticker, query=f"{ticker} stock after:{start.date()} before:{asof_ts.date()}"), min_items)


def fetch_news_window(ticker: str, start, end, max_records: int = 75) -> list[dict[str, str]]:
    """Dated articles from GDELT for [start, end]. One call, better than Google after/before."""
    start_ts = _naive_ts(start)
    end_ts = _naive_ts(end)
    articles = _from_gdelt(ticker, start_ts, end_ts, max_records=max_records)
    if articles:
        return articles
    query = f"{ticker} stock after:{start_ts.date()} before:{end_ts.date()}"
    return _from_google_news(ticker, query=query)


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


def _naive_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _parse_seen(raw: str) -> pd.Timestamp | None:
    if not raw:
        return None
    try:
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Timestamp(ts)


def _from_gdelt(ticker: str, start: pd.Timestamp, end: pd.Timestamp, max_records: int = 75) -> list[dict[str, str]]:
    name = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon", "NVDA": "Nvidia"}.get(
        ticker.upper(), ticker
    )
    params = {
        "query": f"({name} OR {ticker}) sourcelang:english",
        "mode": "ArtList",
        "maxrecords": str(max_records),
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    headers = {"User-Agent": "FinancialAI-assessment/1.0"}
    try:
        response = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params,
            headers=headers,
            timeout=40,
        )
        if response.status_code == 429:
            time.sleep(6)
            response = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params=params,
                headers=headers,
                timeout=40,
            )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    items: list[dict[str, str]] = []
    for row in payload.get("articles") or []:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        items.append(
            {
                "title": title,
                "publisher": str(row.get("domain") or "GDELT"),
                "date": str(row.get("seendate") or ""),
                "source": "gdelt",
            }
        )
    return items
