"""spadak.core — framework-independent primitives: messages, events, agents."""

from spadak.core.messages import Message, Role, ToolCall, ToolResult, TextContent
from spadak.core.events import Event, EventType, EventBus
from spadak.core.agent import Agent, AgentConfig, AgentOutput, FinishReason
from spadak.core.state import State, StepOutcome

__all__ = [
    "Message", "Role", "ToolCall", "ToolResult", "TextContent",
    "Event", "EventType", "EventBus",
    "Agent", "AgentConfig", "AgentOutput", "FinishReason",
    "State", "StepOutcome",
]
