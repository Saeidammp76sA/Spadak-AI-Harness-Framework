"""Spadak — a modular harness for AI agents.

Spadak is deliberately built as SEVERAL independent sub-frameworks, each
usable in isolation, then unified by a thin runtime layer:

    spadak.core           message/event model, agent protocol, runtime loop
    spadak.llm            provider abstraction (OpenAI/Anthropic/local/mock)
    spadak.memory         short-term + long-term memory (episodic/semantic)
    spadak.tools          typed tool registry, schema generation, sandboxing
    spadak.orchestration  multi-agent patterns: supervisor, swarm, pipeline
    spadak.evaluation     traces, assertions, judges, regression suites
    spadak.safety         guardrails: policies, injection defense, budgets

The integration point is ``spadak.runtime.Harness`` which wires all
sub-frameworks together with dependency injection — swap any component
without touching the others.
"""

from spadak.core.messages import Message, Role, ToolCall, ToolResult
from spadak.core.events import Event, EventType
from spadak.core.agent import Agent, AgentConfig
from spadak.runtime import Harness, HarnessBuilder

__version__ = "0.1.0"

__all__ = [
    "Message", "Role", "ToolCall", "ToolResult",
    "Event", "EventType",
    "Agent", "AgentConfig",
    "Harness", "HarnessBuilder",
    "__version__",
]
