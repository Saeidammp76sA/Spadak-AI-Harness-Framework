"""Supervisor pattern: a router agent decides which specialist handles a turn.

`supervisor_policy(harness)` returns a callable usable as Harness.on_route —
it asks the supervisor LLM to pick among registered agents (or FINISH).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from spadak.core.messages import Message


@dataclass
class RoutingDecision:
    next_agent: str | None        # None => finish
    reason: str = ""


FINISH = "FINISH"


def supervisor_policy(supervisor_llm: Any, worker_names: list[str]) -> Callable[[str, Any], RoutingDecision]:
    """Build a routing function: ask `supervisor_llm` who goes next."""
    choices = worker_names + [FINISH]

    def route(user_input: str, context: Any) -> RoutingDecision:
        prompt = (
            "You are a supervisor routing work to specialists.\n"
            f"Available agents: {', '.join(choices)}.\n"
            "Decide the NEXT agent for this request. Reply ONLY JSON "
            '{"agent": "<name or FINISH>", "reason": "<short>"}\n\n'
            f"Conversation so far:\n{_render(context)}\n\nRequest: {user_input}"
        )
        resp = supervisor_llm.complete(messages=[Message.user(prompt, sender="supervisor")],
                                       tools=[], temperature=0.0, model=None)
        data = _extract_json(resp.text or "")
        agent = (data or {}).get("agent", FINISH)
        if agent not in choices:
            agent = FINISH
        return RoutingDecision(next_agent=None if agent == FINISH else agent,
                               reason=(data or {}).get("reason", ""))

    return route


def _render(messages: list[Message] | None) -> str:
    if not messages:
        return "(none)"
    lines = []
    for m in messages[-6:]:
        lines.append(f"{m.role.value}/{m.sender}: {m.render()[:200]}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class Supervisor:
    """Convenience wrapper binding policy + harness for fluent use."""

    def __init__(self, harness: Any, llm: Any, workers: list[str]) -> None:
        self.harness = harness
        self.llm = llm
        self.workers = workers
        harness.router = supervisor_policy(llm, workers)

    def run(self, user_input: str) -> Any:
        return self.harness.run_routed(user_input)
