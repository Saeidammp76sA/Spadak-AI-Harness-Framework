"""Agent abstraction — the contract every agent in Spadak implements.

An Agent is *not* a monolith: it declares capabilities (tools, memory
scope) and exposes two hooks (`prepare` / `step`) that the runtime loop
drives.  This inversion-of-control is what lets evaluation/safety/memory
plug in without agents knowing about them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from spadak.core.messages import Message
from spadak.core.state import State, StepOutcome


class FinishReason(str, Enum):
    FINAL = "final"
    MAX_STEPS = "max_steps"
    BUDGET_EXCEEDED = "budget_exceeded"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    ERROR = "error"
    HANDOFF = "handoff"


@dataclass
class AgentConfig:
    name: str = "agent"
    system_prompt: str = ""
    max_steps: int = 12
    token_budget: int | None = None           # hard cap on tokens for one run
    cost_budget_usd: float | None = None      # hard cap on spend for one run
    tools: list[str] = field(default_factory=list)   # allow-list of tool names
    temperature: float = 0.2
    model: str | None = None                  # provider-specific override
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    answer: str
    finish_reason: FinishReason
    state: State
    handoff_to: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


class Agent:
    """Base class. Subclasses usually only override `system_prompt_fn`."""

    def __init__(self, config: AgentConfig | None = None,
                 llm: Any = None, tools: Any = None,
                 memory: Any = None, bus: Any = None) -> None:
        self.config = config or AgentConfig()
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.bus = bus

    # ---------- overridable hooks ----------
    def system_prompt_fn(self, state: State) -> str:
        """Build the system prompt; can pull from memory."""
        base = self.config.system_prompt
        if self.memory is not None:
            recall = self.memory.recall_query(base or "general")
            if recall:
                base += "\n\n# Relevant long-term memory\n" + recall
        return base

    def prepare(self, state: State, user_input: str) -> None:
        """Called once before the loop; default appends the user message."""
        if self.config.system_prompt or True:
            sys_p = self.system_prompt_fn(state)
            if sys_p:
                state.add(Message.system(sys_p, sender=self.config.name))
        state.add(Message.user(user_input, sender="user"))

    def step(self, state: State) -> StepOutcome:
        """One reasoning iteration: LLM call + optional tool execution."""
        from spadak.core.loop import default_step
        return default_step(self, state)

    # ---------- helpers used by loop ----------
    def allowed_tools(self) -> list[str] | None:
        return self.config.tools or None

    def budget_check(self, state: State) -> str | None:
        if self.config.token_budget and state.tokens_used >= self.config.token_budget:
            return "token_budget"
        if self.config.cost_budget_usd and state.cost_usd >= self.config.cost_budget_usd:
            return "cost_budget"
        return None

    def __repr__(self) -> str:
        return f"<Agent {self.config.name!r} model={self.config.model}>"
