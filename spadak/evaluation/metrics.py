"""Run-level metrics computed from state + events."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spadak.core.events import EventType
from spadak.core.state import State


@dataclass
class RunMetrics:
    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    guardrail_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extras"}
        d["extras"] = self.extras
        return d


def compute_metrics(state: State, events: list[Any]) -> RunMetrics:
    m = RunMetrics(steps=state.step, tokens_in=0, tokens_out=0,
                   cost_usd=state.cost_usd)
    for ev in events:
        p = ev.payload
        if ev.type == EventType.LLM_RESPONSE:
            m.llm_calls += 1
            m.latency_ms += p.get("latency_ms", 0.0)
        elif ev.type == EventType.TOOL_CALL:
            m.tool_calls += 1
        elif ev.type == EventType.TOOL_RESULT and not p.get("ok", True):
            m.tool_failures += 1
        elif ev.type == EventType.GUARDRAIL_TRIGGERED:
            m.guardrail_hits += 1
        elif ev.type == EventType.RUN_FINISHED:
            m.finish_reason = p.get("finish_reason")
    # token split approximation from message metadata
    for msg in state.messages:
        t = msg.meta.get("tokens")
        if isinstance(t, int):
            m.tokens_out += t
    m.tokens_in = max(0, state.tokens_used - m.tokens_out)
    return m
