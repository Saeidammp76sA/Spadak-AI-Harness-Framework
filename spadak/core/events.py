"""Event system — the observability spine of the harness.

Every interesting thing that happens (LLM call, tool exec, guardrail hit,
handoff...) is published on an EventBus.  Tracing, evaluation and logging
are *consumers* of events, never intrusions into agent code.
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, DefaultDict


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    HANDOFF = "handoff"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    ERROR = "error"
    CUSTOM = "custom"


Handler = Callable[["Event"], None]


@dataclass(frozen=True)
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    run_id: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"[{self.ts:.3f}] {self.type.value} from {self.source}: {self.payload}"


class EventBus:
    """Synchronous pub/sub with wildcard support ('*' subscribes to all)."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[EventType | str, list[Handler]] = defaultdict(list)
        self._history: list[Event] = []
        self._record = True

    def subscribe(self, event_type: EventType | str, handler: Handler) -> Callable[[], None]:
        key: EventType | str = event_type if isinstance(event_type, EventType) else str(event_type)
        self._handlers[key].append(handler)

        def unsubscribe() -> None:
            try:
                self._handlers[key].remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def publish(self, event: Event) -> None:
        if self._record:
            self._history.append(event)
        for key in (event.type, "*"):
            for h in list(self._handlers.get(key, ())):
                h(event)

    def emit(self, event_type: EventType, source: str = "", run_id: str = "",
             **payload: Any) -> Event:
        ev = Event(type=event_type, payload=payload, source=source, run_id=run_id)
        self.publish(ev)
        return ev

    def history(self, event_type: EventType | None = None) -> list[Event]:
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.type == event_type]

    def clear_history(self) -> None:
        self._history.clear()

    def set_recording(self, enabled: bool) -> None:
        self._record = enabled
