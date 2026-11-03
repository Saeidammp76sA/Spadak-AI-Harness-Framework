"""MockLLM — deterministic, scriptable model backend for offline tests/evals.

Three ways to drive it:
  1. `MockLLM(script=[...])`            — return responses in order
  2. `MockLLM(handler=fn)`              — fn(messages, tools) -> LLMResponse
  3. `ScriptedBehavior.tool_then_final` — convenience: call a tool, then answer
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from spadak.core.messages import Message, ToolCall
from spadak.llm.base import LLMResponse


class MockLLM:
    name = "mock"

    def __init__(self, script: list[LLMResponse] | None = None,
                 handler: Callable[..., LLMResponse] | None = None,
                 model: str = "mock-1") -> None:
        self.script = list(script or [])
        self.handler = handler
        self.model = model
        self.calls: list[dict[str, Any]] = []   # full request log (for assertions)

    def complete(self, *, messages: list[Message], tools: list[dict[str, Any]],
                 temperature: float, model: str | None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools,
                           "temperature": temperature, "model": model})
        if self.script:
            resp = self.script.pop(0)
        elif self.handler is not None:
            resp = self.handler(messages=messages, tools=tools)
        else:
            resp = LLMResponse(text="(mock default reply)")
        resp.model = resp.model if resp.model != "unknown" else self.model
        if not resp.tokens_in:
            resp.tokens_in = sum(m.token_estimate() for m in messages)
        if not resp.tokens_out:
            resp.tokens_out = max(1, len(resp.text or "") // 4)
        return resp


class ScriptedBehavior:
    """Builders for common mock flows."""

    @staticmethod
    def tool_then_final(tool_name: str, args: dict[str, Any],
                        final_text: str) -> "MockLLM":
        call = LLMResponse(text="", tool_calls=[ToolCall(name=tool_name, arguments=args)])
        final = LLMResponse(text=final_text)
        return MockLLM(script=[call, final])

    @staticmethod
    def echo_with_keyword_tool(final_text: str) -> "MockLLM":
        """If the user asks about 'weather', call get_weather; else answer directly."""
        def handler(messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
            last_user = next((m.text for m in reversed(messages)
                              if m.role.value == "user"), "")
            has_tool_result = any(m.role.value == "tool" for m in messages)
            if "weather" in last_user.lower() and not has_tool_result:
                city = _extract_city(last_user)
                return LLMResponse(text="", tool_calls=[
                    ToolCall(name="get_weather", arguments={"city": city})])
            return LLMResponse(text=final_text)
        return MockLLM(handler=handler)


def _extract_city(text: str) -> str:
    m = re.search(r"weather(?: in| for)?\s+(\w+)", text, re.IGNORECASE)
    return m.group(1) if m else "Tehran"
