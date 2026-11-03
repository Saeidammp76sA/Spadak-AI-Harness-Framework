"""Short-term memory: window strategies for the live conversation.

`SlidingWindow` keeps the last N messages; `TokenBudgetWindow` keeps as
many recent turns as fit a token budget (always preserving the system
prompt and complete tool-call/tool-result pairs — dropping half a pair
confuses every provider).
"""
from __future__ import annotations

from typing import Callable

from spadak.core.messages import Message


class SlidingWindow:
    def __init__(self, keep_last: int = 20) -> None:
        self.keep_last = keep_last

    def context_window(self, messages: list[Message],
                       max_tokens: int | None = None) -> list[Message]:
        head = [m for m in messages if m.role.value == "system"][:1]
        rest = [m for m in messages if m.role.value != "system"]
        return head + rest[-self.keep_last:]


class TokenBudgetWindow:
    def __init__(self, budget_tokens: int = 8000,
                 summarizer: Callable[[list[Message]], str] | None = None) -> None:
        self.budget = budget_tokens
        self.summarizer = summarizer   # optional fn(messages)->str to compress evicted turns

    def context_window(self, messages: list[Message],
                       max_tokens: int | None = None) -> list[Message]:
        budget = min(self.budget, max_tokens or self.budget)
        head = [m for m in messages if m.role.value == "system"][:1]
        body = [m for m in messages if m.role.value != "system"]

        used = sum(m.token_estimate() for m in head)
        kept_rev: list[Message] = []
        evicted_rev: list[Message] = []
        for m in reversed(body):
            cost = m.token_estimate()
            if used + cost <= budget:
                used += cost
                kept_rev.append(m)
            else:
                evicted_rev.append(m)

        kept = list(reversed(kept_rev))
        evicted = list(reversed(evicted_rev))

        # keep tool_call/tool_result pairs intact: if first kept message is a
        # tool result whose call was evicted, also keep the call's assistant msg
        while kept and kept[0].role.value == "tool":
            call_id = kept[0].tool_results[0].call_id
            owner = next((m for m in evicted
                          if any(c.call_id == call_id for c in m.tool_calls)), None)
            if owner is None:
                break
            evicted.remove(owner)
            kept.insert(0, owner)

        out = list(head)
        if evicted and self.summarizer is not None:
            summary = self.summarizer(evicted)
            if summary:
                out.append(Message.system("[Summary of earlier turns]\n" + summary))
        elif evicted:
            note = Message.system(f"[{len(evicted)} earlier messages elided by token budget]")
            out.append(note)
        out.extend(kept)
        return out
