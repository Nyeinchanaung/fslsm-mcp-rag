"""
Unified LLM client backed by LiteLLM.

All provider-specific formatting (OpenAI, Anthropic, Ollama) is handled
internally by LiteLLM, ensuring identical prompt delivery across models.
This design choice is documented in the thesis methodology to eliminate
provider-adapter confounds in cross-model comparisons.

Reference: https://github.com/BerriAI/litellm
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import litellm

from config.logging_config import logger

# Suppress LiteLLM's verbose logs; keep warnings/errors
litellm.suppress_debug_info = True
import logging as _logging
_logging.getLogger("LiteLLM").setLevel(_logging.WARNING)


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost: float = 0.0
    raw_response: Optional[dict] = field(default=None, repr=False)


# LiteLLM requires provider-prefixed model names.
# This dict maps the short names used in config.yaml to LiteLLM model strings.
MODEL_REGISTRY: dict[str, str] = {
    # API models
    "gpt-4o":                       "openai/gpt-4o",
    "gpt-4.1-mini":                  "openai/gpt-4.1-mini",
    "claude-sonnet-4-20250514":     "anthropic/claude-sonnet-4-20250514",
    # Local via Ollama
    "llama3.1:8b":                  "ollama/llama3.1:8b",
    "llama3.1:70b":                 "ollama/llama3.1:70b",
}


def get_litellm_model(model_name: str) -> str:
    """Resolve short config name → LiteLLM prefixed model string."""
    return MODEL_REGISTRY.get(model_name, model_name)


class LLMClient:
    """
    Provider-agnostic LLM client for the FSLSM-RAG-MCP thesis.

    Usage:
        client = LLMClient("gpt-4.1-mini", temperature=0.3)
        resp = client.chat(system="You are...", user="Question?")
        print(resp.content, resp.prompt_tokens, resp.cost)
    """

    def __init__(self, model: str, temperature: float = 0.3):
        self.model_name = model
        self.litellm_model = get_litellm_model(model)
        self.temperature = temperature

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a single system+user turn to the model."""
        logger.debug(
            "LLM call [%s] user_len=%d", self.litellm_model, len(user)
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        response = litellm.completion(
            model=self.litellm_model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
            num_retries=3,
        )

        # Extract cost (LiteLLM tracks this per-call)
        cost = response._hidden_params.get("response_cost", 0.0) or 0.0

        return LLMResponse(
            content=response.choices[0].message.content or "",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            model=response.model,
            cost=cost,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )
