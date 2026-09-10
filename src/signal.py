"""Turn indicator snapshot + headlines into a structured LLM signal."""

from __future__ import annotations

import json
from typing import Any

import requests

from src.config import Settings, project_root


class LLMNotConfiguredError(RuntimeError):
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
    lines = [f"TICKER: {ticker}", "TECHNICALS:"]
    for key, value in snapshot.items():
        lines.append(f"- {key}: {value}")
    lines.append("HEADLINES:")
    if not headlines:
        lines.append("- none")
    for item in headlines:
        lines.append(f"- [{item.get('date', '')}] {item.get('title', '')} ({item.get('publisher', '')})")
    return "\n".join(lines)


def _call_llm(settings: Settings, prompt: str, context: str) -> str:
    if settings.llm_provider == "gemini":
        return _call_gemini(settings.gemini_api_key or "", prompt, context)
    return _call_groq(settings.groq_api_key or "", prompt, context)


def _call_groq(api_key: str, prompt: str, context: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.1-8b-instant",
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _call_gemini(api_key: str, prompt: str, context: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    response = requests.post(
        url,
        json={
            "system_instruction": {"parts": [{"text": prompt}]},
            "contents": [{"parts": [{"text": context}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


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
