"""OpenAI-compatible provider (chat completions + function calling).

Uses only the standard library (urllib) so Spadak has zero hard runtime
dependencies; `openai` package is used automatically when installed.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from spadak.core.messages import Message, ToolCall
from spadak.llm.base import LLMResponse
from spadak.llm.retry import with_retries

# rough price table ($/1M tokens: in, out) — override via constructor
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1",
                 pricing: dict[str, tuple[float, float]] | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.pricing = pricing or _DEFAULT_PRICING

    # ------------------------------------------------------------------
    @with_retries(max_attempts=4)
    def complete(self, *, messages: list[Message], tools: list[dict[str, Any]],
                 temperature: float, model: str | None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "temperature": temperature,
            "messages": [_to_openai_msg(m) for m in messages],
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())

        choice = body["choices"][0]["message"]
        text = choice.get("content") or ""
        calls: list[ToolCall] = []
        for tc in choice.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            calls.append(ToolCall(name=fn["name"], arguments=args,
                                  call_id=tc.get("id", "")))

        usage = body.get("usage", {})
        tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        used_model = body.get("model", self.model)
        pin, pout = self.pricing.get(used_model, (0.0, 0.0))
        cost = (tin * pin + tout * pout) / 1_000_000

        return LLMResponse(text=text, tool_calls=calls, model=used_model,
                           tokens_in=tin, tokens_out=tout, cost_usd=cost, raw=body)


def _to_openai_msg(m: Message) -> dict[str, Any]:
    role = m.role.value
    if role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.text or None,
            "tool_calls": [{
                "id": c.call_id, "type": "function",
                "function": {"name": c.name,
                             "arguments": json.dumps(c.arguments)},
            } for c in m.tool_calls],
        }
    if role == "tool":
        r = m.tool_results[0]
        return {"role": "tool", "tool_call_id": r.call_id,
                "content": json.dumps(r.output) if not isinstance(r.output, str) else r.output}
    return {"role": role, "content": m.render()}
