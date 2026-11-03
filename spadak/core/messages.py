"""Message model for Spadak.

Design goals (why this beats typical chat-string frameworks):
  * provider-neutral content blocks (text / tool_call / tool_result),
  * every message carries metadata (token estimates, provenance, tags)
    so memory & evaluation layers can reason about it,
  * immutable value objects -> safe to share across concurrent agents.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class TextContent:
    text: str

    def render(self) -> str:
        return self.text


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to execute a tool."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def render(self) -> str:
        return f"<tool_call {self.name} id={self.call_id} args={self.arguments!r}>"


@dataclass(frozen=True)
class ToolResult:
    """The outcome of a ToolCall, fed back into the conversation."""
    call_id: str
    name: str
    output: Any
    ok: bool = True
    latency_ms: float = 0.0
    error: str | None = None

    def render(self) -> str:
        if not self.ok:
            return f"<tool_error {self.name}: {self.error}>"
        return f"<tool_result {self.name}: {self.output!r}>"


ContentBlock = TextContent | ToolCall | ToolResult


def _uid() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Message:
    role: Role
    content: tuple[ContentBlock, ...] = ()
    sender: str = ""                       # agent/provider name that produced it
    msg_id: str = field(default_factory=_uid)
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------- constructors ----------
    @classmethod
    def system(cls, text: str, **meta: Any) -> "Message":
        return cls(Role.SYSTEM, (TextContent(text),), meta=meta)

    @classmethod
    def user(cls, text: str, sender: str = "user", **meta: Any) -> "Message":
        return cls(Role.USER, (TextContent(text),), sender=sender, meta=meta)

    @classmethod
    def assistant(cls, text: str = "", calls: tuple[ToolCall, ...] = (),
                  sender: str = "assistant", **meta: Any) -> "Message":
        blocks: list[ContentBlock] = []
        if text:
            blocks.append(TextContent(text))
        blocks.extend(calls)
        return cls(Role.ASSISTANT, tuple(blocks), sender=sender, meta=meta)

    @classmethod
    def tool(cls, result: ToolResult, sender: str = "tool") -> "Message":
        return cls(Role.TOOL, (result,), sender=sender)

    # ---------- accessors ----------
    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextContent))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolCall))

    @property
    def tool_results(self) -> tuple[ToolResult, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolResult))

    def token_estimate(self) -> int:
        """Cheap heuristic; providers override with real counts in meta['tokens']."""
        cached = self.meta.get("tokens")
        if isinstance(cached, int):
            return cached
        return max(1, len(self.render()) // 4)

    def render(self) -> str:
        parts = [b.render() for b in self.content]
        return "\n".join(p for p in parts if p)

    def with_meta(self, **kv: Any) -> "Message":
        from dataclasses import replace
        return replace(self, meta={**self.meta, **kv})

    # ---------- serialization (for storage / replay / eval) ----------
    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "sender": self.sender,
            "msg_id": self.msg_id,
            "ts": self.ts,
            "meta": self.meta,
            "content": [_block_to_dict(b) for b in self.content],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=Role(d["role"]),
            content=tuple(_block_from_dict(b) for b in d["content"]),
            sender=d.get("sender", ""),
            msg_id=d.get("msg_id", _uid()),
            ts=d.get("ts", time.time()),
            meta=d.get("meta", {}),
        )


def _block_to_dict(b: ContentBlock) -> dict[str, Any]:
    if isinstance(b, TextContent):
        return {"kind": "text", "text": b.text}
    if isinstance(b, ToolCall):
        return {"kind": "tool_call", "name": b.name,
                "arguments": b.arguments, "call_id": b.call_id}
    r: ToolResult = b
    return {"kind": "tool_result", "call_id": r.call_id, "name": r.name,
            "output": r.output, "ok": r.ok, "error": r.error,
            "latency_ms": r.latency_ms}


def _block_from_dict(d: dict[str, Any]) -> ContentBlock:
    kind = d["kind"]
    if kind == "text":
        return TextContent(d["text"])
    if kind == "tool_call":
        return ToolCall(name=d["name"], arguments=d["arguments"],
                        call_id=d.get("call_id", _uid()))
    return ToolResult(call_id=d["call_id"], name=d["name"], output=d["output"],
                      ok=d.get("ok", True), error=d.get("error"),
                      latency_ms=d.get("latency_ms", 0.0))
