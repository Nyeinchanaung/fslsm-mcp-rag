"""Unified LLM interface for OpenAI, Anthropic, and Ollama (local Llama 3)."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from enum import Enum

from config.logging_config import logger


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # local Llama 3 via `ollama serve`


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Single interface over all three providers. Swap models without changing call sites."""

    def __init__(
        self,
        model: str,
        provider: LLMProvider,
        temperature: float = 0.3,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self._ollama_base = ollama_base_url
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == LLMProvider.OPENAI:
            import openai
            from config.settings import settings
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in .env")
            return openai.OpenAI(api_key=settings.openai_api_key)

        if self.provider == LLMProvider.ANTHROPIC:
            import anthropic
            from config.settings import settings
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in .env")
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # OLLAMA: no persistent client needed — raw HTTP
        return None

    def chat(self, system: str, user: str, max_tokens: int = 512) -> LLMResponse:
        """Send a system + user message and return a structured response."""
        logger.debug("LLM call [%s|%s] user_len=%d", self.provider, self.model, len(user))

        if self.provider == LLMProvider.OPENAI:
            return self._openai_chat(system, user, max_tokens)
        if self.provider == LLMProvider.ANTHROPIC:
            return self._anthropic_chat(system, user, max_tokens)
        if self.provider == LLMProvider.OLLAMA:
            return self._ollama_chat(system, user, max_tokens)

        raise ValueError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------ #
    #  Provider implementations                                           #
    # ------------------------------------------------------------------ #

    def _openai_chat(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            model=self.model,
        )

    def _anthropic_chat(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(
            content=resp.content[0].text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            model=self.model,
        )

    def _ollama_chat(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self._ollama_base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())

        return LLMResponse(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            model=self.model,
        )
