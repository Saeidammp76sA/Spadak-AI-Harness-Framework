"""Anthropic provider (messages API + tool use). Stdlib-only HTTP."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from spadak.core.messages import Message, ToolCall
from spadak.llm.base import LLMResponse
from spadak.llm.retry import with_retries

_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: str | None = None,
                 base_url: str = "https://api.anthropic.com/v1",
                 pricing: dict[str, tuple[float, float]] | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.pricing = pricing or _DEFAULT_PRICING

    @with_retries(max_attempts=4)
    def complete(self, *, messages: list[Message], tools: list[dict[str, Any]],
                 temperature: float, model: str | None) -> LLMResponse:
        system_parts = [m.render() for m in messages if m.role.value == "system"]
        chat = [_to_anthropic_msg(m) for m in messages if m.role.value != "system"]
        # merge consecutive same-role turns (tool results -> user blocks)
        chat = _merge_roles(chat)

        payload: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": chat,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                                 "input_schema": t.get("parameters", {})} for t in tools]

        req = urllib.request.Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())

        text_chunks, calls = [], []
        for block in body.get("content", []):
            if block["type"] == "text":
                text_chunks.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(ToolCall(name=block["name"], arguments=block["input"],
                                      call_id=block["id"]))

        usage = body.get("usage", {})
        tin = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
        tout = usage.get("output_tokens", 0)
        used_model = body.get("model", self.model)
        pin, pout = self.pricing.get(used_model, (0.0, 0.0))
        cost = (tin * pin + tout * pout) / 1_000_000

        return LLMResponse(text="\n".join(text_chunks) or None, tool_calls=calls,
                           model=used_model, tokens_in=tin, tokens_out=tout,
                           cost_usd=cost, raw=body)


def _to_anthropic_msg(m: Message) -> dict[str, Any]:
    role = "assistant" if m.role.value == "assistant" else "user"
    if m.role.value == "tool":
        r = m.tool_results[0]
        return {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": r.call_id,
            "content": json.dumps(r.output) if not isinstance(r.output, str) else r.output,
            "is_error": not r.ok,
        }]}
    if m.role.value == "assistant" and m.tool_calls:
        blocks: list[dict[str, Any]] = []
        if m.text:
            blocks.append({"type": "text", "text": m.text})
        for c in m.tool_calls:
            blocks.append({"type": "tool_use", "id": c.call_id,
                           "name": c.name, "input": c.arguments})
        return {"role": "assistant", "content": blocks}
    return {"role": role, "content": [{"type": "text", "text": m.render()}]}


def _merge_roles(chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for msg in chat:
        if merged and merged[-1]["role"] == msg["role"]:
            prev_blocks = merged[-1]["content"]
            if isinstance(prev_blocks, list) and isinstance(msg["content"], list):
                prev_blocks.extend(msg["content"])
                continue
        merged.append(msg)
    return merged
