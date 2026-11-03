"""Guardrail contract.

Each guardrail can veto or rewrite at three checkpoints:
  * check_input   — before the run starts (user message),
  * check_tool_call — before a tool executes,
  * check_output  — after the model responds, before it is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Action(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REWRITE = "rewrite"


@dataclass
class Verdict:
    action: Action = Action.ALLOW
    reason: str = ""
    replaced_text: str | None = None      # for output rewrites
    replaced_args: dict[str, Any] | None = None  # for tool-arg sanitization

    @property
    def blocked(self) -> bool:
        return self.action == Action.BLOCK

    @staticmethod
    def allow() -> "Verdict":
        return Verdict(Action.ALLOW)


class Guardrail:
    """Base class; override only the checkpoints you care about."""
    name = "guardrail"

    def check_input(self, agent: Any, user_input: str) -> Verdict:
        return Verdict.allow()

    def check_tool_call(self, agent: Any, state: Any, call: Any) -> Verdict:
        return Verdict.allow()

    def check_output(self, agent: Any, state: Any, message: Any) -> Verdict:
        return Verdict.allow()
