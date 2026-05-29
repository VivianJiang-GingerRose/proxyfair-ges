"""Wrapper classes for invoking external LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .config import ProviderSettings


@dataclass
class BaseLLMClient:
    """Stores shared configuration for a provider client."""

    name: str
    api_key: Optional[str]
    model: str
    temperature: float
    max_output_tokens: int

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def __post_init__(self) -> None:  # Needed so subclasses can call super().__post_init__
        return None

    def generate(self, prompt: str) -> str:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


@dataclass
class OpenAIClient(BaseLLMClient):
    """Thin wrapper over the OpenAI chat-completions endpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.is_configured():
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        else:
            self._client = None

    def generate(self, prompt: str) -> str:
        if not self._client:
            raise RuntimeError("OpenAI client is not configured. Provide OPENAI_API_KEY.")
        
        # Use max_completion_tokens for newer models like gpt-4o
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        
        # Check if using newer models that require max_completion_tokens
        if self.model.startswith(("gpt-4o", "gpt-5", "o1", "o3")):
            params["max_completion_tokens"] = self.max_output_tokens
        else:
            params["max_tokens"] = self.max_output_tokens
            
        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()


@dataclass
class AnthropicClient(BaseLLMClient):
    """Wrapper around the Anthropic Claude messages API."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.is_configured():
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        else:
            self._client = None

    def generate(self, prompt: str) -> str:
        if not self._client:
            raise RuntimeError("Anthropic client is not configured. Provide ANTHROPIC_API_KEY.")
        response = self._client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        chunks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "".join(chunks).strip()


@dataclass
class GoogleAIClient(BaseLLMClient):
    """Wrapper around the Google Generative AI SDK (Gemini)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.is_configured():
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self.api_key)
            self._types = types
        else:
            self._client = None
            self._types = None

    def generate(self, prompt: str) -> str:
        if not self._client:
            raise RuntimeError("Google AI client is not configured. Provide GOOGLE_API_KEY.")
        
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        
        if response.text:
            return response.text.strip()
        
        raise RuntimeError("Google AI response did not contain any text content.")


def build_clients(settings: ProviderSettings) -> Dict[str, BaseLLMClient]:
    """Instantiate clients for all configured providers."""

    clients: Dict[str, BaseLLMClient] = {}

    openai_client = OpenAIClient(
        name="openai",
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
    )
    if openai_client.is_configured():
        clients["openai"] = openai_client

    anthropic_client = AnthropicClient(
        name="anthropic",
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
    )
    if anthropic_client.is_configured():
        clients["anthropic"] = anthropic_client

    google_client = GoogleAIClient(
        name="google",
        api_key=settings.google_api_key,
        model=settings.google_model,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
    )
    if google_client.is_configured():
        clients["google"] = google_client

    return clients
