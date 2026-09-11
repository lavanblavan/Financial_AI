"""Turn indicator snapshot + headlines into a structured LLM signal."""

from __future__ import annotations

import json
from typing import Any

import requests

from src.config import DEFAULT_RULE_VARIANT, Settings, groq_key_shape, project_root
from src.evaluate import RULE_TO_SIGNAL

GROQ_MODEL = "openai/gpt-oss-20b"


class LLMNotConfiguredError(RuntimeError):
    pass


class GroqAPIError(RuntimeError):
    pass


def build_signal(
    snapshot: dict[str, Any],
    headlines: list[dict[str, str]],
    settings: Settings,
    ticker: str,
) -> dict[str, Any]:
    if not settings.llm_ready:
        raise LLMNotConfiguredError(
            "No LLM key found. Local: set GROQ_API_KEY in .env. "
            "Colab: add GROQ_API_KEY in Secrets and grant access."
        )

    prompt = _read_prompt()
    context = _context_pack(ticker, snapshot, headlines)
    raw = _call_llm(settings, prompt, context)
    return _parse_signal(raw)


def _read_prompt() -> str:
    path = project_root() / "prompts" / "analyst.md"
    return path.read_text(encoding="utf-8")


def _context_pack(ticker: str, snapshot: dict[str, Any], headlines: list[dict[str, str]]) -> str:
    rule_signal = RULE_TO_SIGNAL.get(str(snapshot.get("momentum_bias", "")), "HOLD")
    lines = [
        f"TICKER: {ticker}",
        f"RULE_VARIANT: {snapshot.get('rule_variant', DEFAULT_RULE_VARIANT)}",
        f"RULE_SIGNAL: {rule_signal}",
        "TECHNICALS:",
    ]
    for key, value in snapshot.items():
        if key in {"rule_variant"}:
            continue
        lines.append(f"- {key}: {value}")
    lines.append("HEADLINES:")
    if not headlines:
        lines.append("- none")
    for item in headlines:
        lines.append(f"- [{item.get('date', '')}] {item.get('title', '')} ({item.get('publisher', '')})")
    return "\n".join(lines)


def _call_llm(settings: Settings, prompt: str, context: str) -> str:
    return _call_groq(settings.groq_api_key or "", prompt, context)


def _call_groq(api_key: str, prompt: str, context: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
            ],
        },
        timeout=60,
    )
    if response.status_code >= 400:
        raise GroqAPIError(_groq_error_message(response, api_key))
    return response.json()["choices"][0]["message"]["content"]


def _groq_error_message(response: requests.Response, api_key: str) -> str:
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or str(payload)
    except ValueError:
        detail = response.text[:300]
    shape = groq_key_shape(api_key)
    if response.status_code == 401:
        return (
            f"Groq 401 Unauthorized (key shape {shape}). "
            "The secret is present but Groq rejected it. "
            "Create a new key at https://console.groq.com/keys, "
            "paste only the gsk_... value (no quotes, no Bearer), "
            "update the Colab Secret GROQ_API_KEY, then Runtime → Restart session."
        )
    return f"Groq HTTP {response.status_code}: {detail}"


def _parse_signal(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    signal = str(payload.get("signal", "HOLD")).upper()
    if signal not in {"BUY", "HOLD", "SELL"}:
        signal = "HOLD"
    payload["signal"] = signal
    try:
        payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.5))))
    except (TypeError, ValueError):
        payload["confidence"] = 0.5
    payload.setdefault("horizon", "1-4 weeks")
    payload.setdefault("rationale", "")
    payload.setdefault("bull_case", "")
    payload.setdefault("bear_case", "")
    payload.setdefault("risks", [])
    return payload
