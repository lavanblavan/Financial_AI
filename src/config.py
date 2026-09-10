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


def _secret_from_colab(name: str) -> str | None:
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        value = userdata.get(name)
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


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
            return value
    return _secret_from_env(name)


@dataclass(frozen=True)
class Settings:
    ticker: str
    lookback: str
    llm_provider: str
    groq_api_key: str | None
    gemini_api_key: str | None

    @property
    def llm_api_key(self) -> str | None:
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        return self.groq_api_key

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_key)


def load_settings() -> Settings:
    """Load .env locally. In Colab, Secrets already live in userdata."""
    if not running_in_colab():
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None
        if load_dotenv:
            load_dotenv(project_root() / ".env", override=False)

    provider = (get_secret("LLM_PROVIDER") or "groq").strip().lower()
    if provider not in {"groq", "gemini"}:
        provider = "groq"

    return Settings(
        ticker=(get_secret("TICKER") or "AAPL").upper(),
        lookback=get_secret("LOOKBACK") or "2y",
        llm_provider=provider,
        groq_api_key=get_secret("GROQ_API_KEY"),
        gemini_api_key=get_secret("GEMINI_API_KEY"),
    )


def describe_env(settings: Settings) -> dict[str, str]:
    """Safe status for printing. Never includes key values."""
    source = "colab-secrets" if running_in_colab() else "local-dotenv"
    return {
        "runtime": "colab" if running_in_colab() else "local",
        "secret_source": source,
        "ticker": settings.ticker,
        "lookback": settings.lookback,
        "llm_provider": settings.llm_provider,
        "llm_key_present": "yes" if settings.llm_ready else "no",
    }
