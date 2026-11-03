"""Swarm pattern: decentralized handoffs between peers.

Each agent gets `transfer_to_<peer>` tools; the runtime follows the
handoff chain (with loop protection) until an agent answers without
transferring.
"""
from __future__ import annotations

from typing import Any

from spadak.orchestration.handoff import make_transfer_tool


class Swarm:
    def __init__(self, harness: Any, max_handoffs: int = 6) -> None:
        self.harness = harness
        self.max_handoffs = max_handoffs

    def wire(self) -> "Swarm":
        """Attach transfer tools to every peer of every agent."""
        names = list(self.harness.agents)
        for name, agent in self.harness.agents.items():
            for peer in names:
                if peer == name:
                    continue
                t = make_transfer_tool(peer)
                agent.tools.register(t)
                if peer not in agent.config.tools:
                    agent.config.tools.append(peer and t.name)
        self.harness.swarm_enabled = True
        self.harness.max_handoffs = self.max_handoffs
        return self

    def run(self, user_input: str, first_agent: str) -> Any:
        return self.harness.run_swarm(user_input, entry=first_agent)
