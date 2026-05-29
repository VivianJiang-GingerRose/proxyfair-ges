"""Configuration helpers for LLM provider credentials and defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv is optional

_DEFAULT_OPENAI_MODEL = "gpt-4o"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
_DEFAULT_GOOGLE_MODEL = "gemini-3-pro-preview"
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_TOKENS = 1024


@dataclass
class ProviderSettings:
    """Holds API credentials and generation defaults for the supported providers."""

    openai_api_key: Optional[str] = None
    openai_model: str = _DEFAULT_OPENAI_MODEL
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = _DEFAULT_ANTHROPIC_MODEL
    google_api_key: Optional[str] = None
    google_model: str = _DEFAULT_GOOGLE_MODEL
    temperature: float = _DEFAULT_TEMPERATURE
    max_output_tokens: int = _DEFAULT_MAX_TOKENS

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ProviderSettings":
        """Create settings from a JSON file and/or environment variables."""

        data: Dict[str, Any] = {}
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if not isinstance(loaded, dict):
                    raise ValueError("LLM config file must contain a JSON object.")
                data = loaded

        def _from_env(key: str) -> Optional[str]:
            value = os.getenv(key)
            return value if value not in (None, "") else None

        def _from_data(keys: tuple[str, ...]) -> Optional[Any]:
            current: Any = data
            for part in keys:
                if not isinstance(current, dict) or part not in current:
                    return None
                current = current[part]
            return current

        def _resolve(
            data_keys: tuple[str, ...],
            env_key: str,
            default: Optional[Any],
        ) -> Optional[Any]:
            if data:
                value = _from_data(data_keys)
                if value not in (None, ""):
                    return value
            env_value = _from_env(env_key)
            if env_value is not None:
                return env_value
            return default

        generation_defaults = data.get("generation", {}) if isinstance(data.get("generation"), dict) else {}

        settings = cls(
            openai_api_key=_resolve(("openai", "api_key"), "OPENAI_API_KEY", data.get("openai_api_key")),
            openai_model=_resolve(("openai", "model"), "OPENAI_MODEL", data.get("openai_model", _DEFAULT_OPENAI_MODEL)) or _DEFAULT_OPENAI_MODEL,
            anthropic_api_key=_resolve(("anthropic", "api_key"), "ANTHROPIC_API_KEY", data.get("anthropic_api_key")),
            anthropic_model=_resolve(("anthropic", "model"), "ANTHROPIC_MODEL", data.get("anthropic_model", _DEFAULT_ANTHROPIC_MODEL)) or _DEFAULT_ANTHROPIC_MODEL,
            google_api_key=_resolve(("google", "api_key"), "GOOGLE_API_KEY", data.get("google_api_key")),
            google_model=_resolve(("google", "model"), "GOOGLE_MODEL", data.get("google_model", _DEFAULT_GOOGLE_MODEL)) or _DEFAULT_GOOGLE_MODEL,
            temperature=float(
                _resolve(("generation", "temperature"), "LLM_TEMPERATURE", generation_defaults.get("temperature", _DEFAULT_TEMPERATURE))
            ),
            max_output_tokens=int(
                _resolve(("generation", "max_output_tokens"), "LLM_MAX_OUTPUT_TOKENS", generation_defaults.get("max_output_tokens", _DEFAULT_MAX_TOKENS))
            ),
        )
        return settings

    def enabled_providers(self) -> Dict[str, bool]:
        """Return a mapping that indicates which providers are ready to use."""

        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "google": bool(self.google_api_key),
        }

    def as_dict(self) -> Dict[str, Any]:
        """Expose a serializable representation without secrets."""

        return {
            "openai_model": self.openai_model,
            "anthropic_model": self.anthropic_model,
            "google_model": self.google_model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
