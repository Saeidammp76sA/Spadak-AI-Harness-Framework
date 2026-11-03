"""LLM provider protocol shared by every backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from spadak.core.messages import Message, ToolCall


@dataclass(frozen=True)
class CompletionParams:
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.2
    model: str | None = None
    max_tokens: int = 2048
    stop: list[str] | None = None


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = "unknown"
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Everything the harness needs from a model backend."""

    name: str

    def complete(self, *, messages: list[Message], tools: list[dict[str, Any]],
                 temperature: float, model: str | None) -> LLMResponse:
        ...
