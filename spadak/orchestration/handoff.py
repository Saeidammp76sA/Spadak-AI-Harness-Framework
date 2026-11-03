"""Handoffs: first-class transfers between agents (OpenAI-swarm style).

A handoff is just a tool call that returns control to the runtime with a
`target` set; the harness then continues the SAME conversation with the
new agent — preserving full transcript for grounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spadak.tools.tool import Tool, tool


@dataclass(frozen=True)
class Handoff:
    target: str
    reason: str = ""

    def as_result(self) -> dict[str, Any]:
        return {"handoff": self.target, "reason": self.reason}


def make_transfer_tool(target_agent: str, description: str = "") -> Tool:
    """Build a `transfer_to_<agent>` tool that triggers a handoff."""
    name = f"transfer_to_{target_agent}"

    def _transfer(reason: str = "") -> dict[str, Any]:
        """Transfer this conversation to another agent. Provide a short reason."""
        return Handoff(target=target_agent, reason=reason).as_result()

    _transfer.__name__ = name
    return tool(_transfer, name=name,
                description=description or f"Hand the current task over to {target_agent}.")
