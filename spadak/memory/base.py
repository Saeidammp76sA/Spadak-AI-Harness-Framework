"""Memory contracts.

Two-tier design used across the harness:
  * Short-term: shapes the *live* message list sent to the LLM
    (sliding window / token budget / summarization).
  * Long-term: persists facts & episodes across runs; retrieval is
    pluggable (BM25-ish inverted index by default, vector store optional).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from spadak.core.messages import Message

Scope = Literal["episodic", "semantic", "procedural"]


@dataclass
class MemoryRecord:
    content: str
    scope: Scope = "semantic"
    tags: list[str] = field(default_factory=list)
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    importance: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "scope": self.scope, "tags": self.tags,
                "record_id": self.record_id, "ts": self.ts,
                "importance": self.importance, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryRecord":
        return cls(content=d["content"], scope=d.get("scope", "semantic"),
                   tags=d.get("tags", []), record_id=d.get("record_id"),
                   ts=d.get("ts", time.time()),
                   importance=d.get("importance", 0.5), meta=d.get("meta", {}))


@runtime_checkable
class Memory(Protocol):
    """Unified interface the runtime loop expects from any memory backend."""

    def context_window(self, messages: list[Message],
                       max_tokens: int | None = None) -> list[Message]: ...

    def remember(self, record: MemoryRecord) -> None: ...

    def recall_query(self, query: str, k: int = 4) -> str: ...
