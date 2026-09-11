"""Load settings from Colab Secrets or a local .env file.

Colab and your laptop do not share files. Secrets must be read from
the place that matches where the notebook is running.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def running_in_colab() -> bool:
    return "google.colab" in sys.modules


def project_root() -> Path:
    """Repo root: folder that contains src/ and requirements.txt."""
    here = Path(__file__).resolve().parent
    return here.parent


def probe_colab_secret(name: str) -> str:
    """Why a Colab secret is missing. Never returns the secret value."""
    try:
        from google.colab import userdata
    except ImportError:
        return "not_colab"
    try:
        value = userdata.get(name)
    except Exception as exc:
        kind = type(exc).__name__
        message = str(exc).lower()
        if "Access" in kind or "access" in message or "grant" in message:
            return "access_denied"
        if "NotFound" in kind or "not found" in message:
            return "not_found"
        return f"error:{kind}"
    if value and str(value).strip():
        return "present"
    return "empty"


def _secret_from_colab(name: str) -> str | None:
    if probe_colab_secret(name) != "present":
        return None
    from google.colab import userdata

    return str(userdata.get(name)).strip()


def normalize_secret(value: str | None) -> str | None:
    """Strip quotes, whitespace, and a leading Bearer prefix."""
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if text.lower().startswith("bearer "):
        text = text[7:].strip()
    return text or None


def groq_key_shape(api_key: str | None) -> str:
    if not api_key:
        return "missing"
    if api_key.startswith("gsk_"):
        return "gsk_*"
    return "unexpected"


def _secret_from_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_secret(name: str) -> str | None:
    """Colab Secrets first, then process env / .env."""
    if running_in_colab():
        value = _secret_from_colab(name)
        if value:
            return normalize_secret(value)
    return normalize_secret(_secret_from_env(name))


# Selected on 126-session holdout — see docs/RULE_SELECTION.md and scripts/select_rule.py
DEFAULT_RULE_VARIANT = "score"


@dataclass(frozen=True)
class Settings:
    ticker: str
    lookback: str
    groq_api_key: str | None

    @property
    def llm_ready(self) -> bool:
        return bool(self.groq_api_key)


def load_settings() -> Settings:
    """Load .env locally. In Colab, Secrets already live in userdata."""
    if not running_in_colab():
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None
        if load_dotenv:
            load_dotenv(project_root() / ".env", override=False)

    return Settings(
        ticker=(get_secret("TICKER") or "AAPL").upper(),
        lookback=get_secret("LOOKBACK") or "2y",
        groq_api_key=get_secret("GROQ_API_KEY"),
    )


def describe_env(settings: Settings) -> dict[str, str]:
    """Safe status for printing. Never includes key values."""
    source = "colab-secrets" if running_in_colab() else "local-dotenv"
    status = {
        "runtime": "colab" if running_in_colab() else "local",
        "secret_source": source,
        "ticker": settings.ticker,
        "lookback": settings.lookback,
        "llm_provider": "groq",
        "llm_key_present": "yes" if settings.llm_ready else "no",
        "groq_key_shape": groq_key_shape(settings.groq_api_key),
    }
    if running_in_colab():
        status["groq_secret_status"] = probe_colab_secret("GROQ_API_KEY")
    return status


def missing_key_help(settings: Settings) -> str:
    if settings.llm_ready:
        return ""
    if running_in_colab():
        status = probe_colab_secret("GROQ_API_KEY")
        if status == "access_denied":
            return (
                "Colab has GROQ_API_KEY but this notebook cannot read it. "
                "Open the key icon (Secrets), find GROQ_API_KEY, and turn Notebook access ON. "
                "Then Runtime → Restart session and Run all."
            )
        if status == "not_found":
            return (
                "Colab Secret GROQ_API_KEY is missing. "
                "Do not use Windows 'copy' in Colab. "
                "Click the key icon → Add secret → name exactly GROQ_API_KEY → paste the key → "
                "enable Notebook access → re-run this cell."
            )
        if status == "empty":
            return "Colab Secret GROQ_API_KEY exists but is empty. Paste the key, save, and re-run."
        return (
            f"Could not read GROQ_API_KEY ({status}). "
            "Add it under the Colab key icon and enable Notebook access."
        )
    return "Local: copy .env.example to .env and set GROQ_API_KEY."
