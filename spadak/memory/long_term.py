"""Long-term memory: storage backends + retrieval.

`InvertedIndexStore` is a dependency-free BM25-flavoured text index —
surprisingly effective and fully deterministic (great for evals).
`EpisodicMemory` stores run transcripts; `LongTermMemory` composes a
short-term window strategy with a long-term store behind one interface,
which is what the runtime injects into agents.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from spadak.core.messages import Message
from spadak.memory.base import MemoryRecord
from spadak.memory.short_term import TokenBudgetWindow

_WORD_RE = re.compile(r"[a-z0-9\u0600-\u06FF]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class InvertedIndexStore:
    """Tiny BM25 index. Swap for pgvector/Qdrant by implementing same API."""

    def __init__(self, persist_path: str | os.PathLike | None = None) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self.index: dict[str, set[str]] = defaultdict(set)
        self.doc_len: dict[str, int] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()

    # ---- CRUD ----
    def add(self, record: MemoryRecord) -> None:
        self.records[record.record_id] = record
        for tok in set(_tokenize(record.content + " " + " ".join(record.tags))):
            self.index[tok].add(record.record_id)
        self.doc_len[record.record_id] = max(1, len(_tokenize(record.content)))
        self._save()

    def search(self, query: str, k: int = 4,
               scope: str | None = None) -> list[tuple[MemoryRecord, float]]:
        if not self.records:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        n_docs = len(self.records)
        avg_len = sum(self.doc_len.values()) / n_docs
        scores: dict[str, float] = defaultdict(float)
        k1, b = 1.5, 0.75
        for term in q_terms:
            posting = self.index.get(term, set())
            if not posting:
                continue
            idf = math.log(1 + (n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for rid in posting:
                rec = self.records[rid]
                tf = _tokenize(rec.content).count(term)
                dl = self.doc_len[rid]
                scores[rid] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_len))
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        out: list[tuple[MemoryRecord, float]] = []
        for rid, score in ranked[:k]:
            rec = self.records[rid]
            if scope and rec.scope != scope:
                continue
            # importance acts as a gentle prior
            out.append((rec, score * (0.5 + rec.importance)))
        out.sort(key=lambda x: -x[1])
        return out

    def all(self) -> Iterable[MemoryRecord]:
        return self.records.values()

    # ---- persistence ----
    def _save(self) -> None:
        if not self.persist_path:
            return
        data = {rid: r.to_dict() for rid, r in self.records.items()}
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps(data), encoding="utf-8")

    def _load(self) -> None:
        assert self.persist_path
        data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        for d in data.values():
            self.add(MemoryRecord.from_dict(d))


class EpisodicMemory:
    """Append-only log of past runs; each episode summarized & indexed."""

    def __init__(self, store: InvertedIndexStore | None = None) -> None:
        self.store = store or InvertedIndexStore()

    def log_episode(self, task: str, outcome: str, tags: list[str] | None = None,
                    importance: float = 0.7) -> MemoryRecord:
        rec = MemoryRecord(
            content=f"Task: {task}\nOutcome: {outcome}",
            scope="episodic", tags=tags or [], importance=importance,
            meta={"logged_at": time.time()},
        )
        self.store.add(rec)
        return rec

    def recall(self, query: str, k: int = 3) -> list[MemoryRecord]:
        return [r for r, _ in self.store.search(query, k=k, scope="episodic")]


class LongTermMemory:
    """Facade injected into agents: short-term windowing + long-term recall.

    Implements the `Memory` protocol used by spadak.core.loop.
    """

    def __init__(self, store: InvertedIndexStore | None = None,
                 window: TokenBudgetWindow | None = None,
                 auto_capture: bool = True) -> None:
        self.store = store or InvertedIndexStore()
        self.window = window or TokenBudgetWindow()
        self.auto_capture = auto_capture   # learn from user messages post-run

    # ---- Memory protocol ----
    def context_window(self, messages: list[Message],
                       max_tokens: int | None = None) -> list[Message]:
        return self.window.context_window(messages, max_tokens)

    def remember(self, record: MemoryRecord) -> None:
        self.store.add(record)

    def recall_query(self, query: str, k: int = 4) -> str:
        hits = self.store.search(query, k=k)
        if not hits:
            return ""
        lines = [f"- ({r.scope}, imp={r.importance:.1f}) {r.content}" for r, _ in hits]
        return "\n".join(lines)

    # ---- learning hooks ----
    def capture_from_run(self, user_input: str, final_answer: str,
                         tags: list[str] | None = None) -> None:
        """Called by the harness at RUN_FINISHED when auto_capture is on."""
        if not self.auto_capture or not final_answer:
            return
        self.store.add(MemoryRecord(
            content=f"Q: {user_input}\nA: {final_answer}",
            scope="episodic", tags=tags or [], importance=0.6,
        ))

    def save_fact(self, fact: str, tags: list[str] | None = None,
                  importance: float = 0.8) -> MemoryRecord:
        rec = MemoryRecord(content=fact, scope="semantic",
                           tags=tags or [], importance=importance)
        self.store.add(rec)
        return rec
