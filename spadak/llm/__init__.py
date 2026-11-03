"""spadak.llm — provider-agnostic LLM layer.

Includes a deterministic `MockLLM` (scripted responses) so the entire
harness is testable offline — a key advantage over harnesses that force
live API calls in CI.
"""
from spadak.llm.base import LLMProvider, LLMResponse, CompletionParams
from spadak.llm.mock import MockLLM, ScriptedBehavior
from spadak.llm.openai import OpenAIProvider
from spadak.llm.anthropic import AnthropicProvider
from spadak.llm.retry import with_retries, RateLimiter

__all__ = [
    "LLMProvider", "LLMResponse", "CompletionParams",
    "MockLLM", "ScriptedBehavior",
    "OpenAIProvider", "AnthropicProvider",
    "with_retries", "RateLimiter",
]
