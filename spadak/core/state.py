"""Conversation state & step outcome — passed through the runtime loop."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from spadak.core.messages import Message


@dataclass
class State:
    """Mutable running state of one agent turn-loop."""
    messages: list[Message] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)   # per-run working data
    step: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    finished: bool = False
    output: str | None = None                                  # final answer

    def add(self, msg: Message | list[Message]) -> None:
        if isinstance(msg, list):
            self.messages.extend(msg)
        else:
            self.messages.append(msg)

    def token_count(self) -> int:
        return sum(m.token_estimate() for m in self.messages)

    def last(self, n: int = 1) -> list[Message]:
        return self.messages[-n:]

    def copy(self) -> "State":
        return replace(self, messages=list(self.messages),
                       scratchpad=dict(self.scratchpad))


@dataclass
class StepOutcome:
    """What one loop iteration produced; drives control flow."""
    continue_run: bool = True
    stop_reason: str | None = None        # 'final' | 'max_steps' | 'budget' | 'guardrail' | 'error'
    delta_messages: list[Message] = field(default_factory=list)
