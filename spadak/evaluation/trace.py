"""Traces: reconstruct a full run from the event bus (zero-intrusion)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spadak.core.events import Event, EventBus, EventType


@dataclass
class TraceStep:
    index: int
    kind: str                      # 'llm' | 'tool' | 'guardrail' | ...
    agent: str
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0


@dataclass
class Trace:
    run_id: str
    input: str = ""
    output: str | None = None
    steps: list[TraceStep] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    finish_reason: str | None = None

    def tool_calls(self) -> list[str]:
        return [s.detail.get("tool", "") for s in self.steps if s.kind == "tool_call"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "input": self.input,
            "output": self.output,
            "finish_reason": self.finish_reason,
            "steps": [{"index": s.index, "kind": s.kind, "agent": s.agent,
                       "detail": s.detail, "ts": s.ts} for s in self.steps],
        }


def trace_from_events(events: list[Event]) -> Trace:
    tr = Trace(run_id=events[0].run_id if events else "", events=events)
    idx = 0
    for ev in events:
        t = ev.type
        step: TraceStep | None = None
        if t == EventType.RUN_STARTED:
            tr.input = ev.payload.get("input", "")
            step = TraceStep(idx, "run_start", ev.source, dict(ev.payload), ev.ts)
        elif t == EventType.LLM_RESPONSE:
            step = TraceStep(idx, "llm", ev.source, dict(ev.payload), ev.ts)
        elif t == EventType.TOOL_CALL:
            step = TraceStep(idx, "tool_call", ev.source, dict(ev.payload), ev.ts)
        elif t == EventType.TOOL_RESULT:
            step = TraceStep(idx, "tool_result", ev.source, dict(ev.payload), ev.ts)
        elif t == EventType.GUARDRAIL_TRIGGERED:
            step = TraceStep(idx, "guardrail", ev.source, dict(ev.payload), ev.ts)
        elif t == EventType.HANDOFF:
            step = TraceStep(idx, "handoff", ev.source, dict(ev.payload), ev.ts)
        elif t == EventType.RUN_FINISHED:
            tr.output = ev.payload.get("output")
            tr.finish_reason = ev.payload.get("finish_reason")
            step = TraceStep(idx, "run_finish", ev.source, dict(ev.payload), ev.ts)
        if step is not None:
            tr.steps.append(step)
            idx += 1
    return tr


def capture_traces(bus: EventBus) -> dict[str, Trace]:
    """Group recorded events by run_id and build one Trace per run.

    Returns a mapping {run_id: Trace} — stable to iterate with .values().
    """
    by_run: dict[str, list[Event]] = {}
    for ev in bus.history():
        by_run.setdefault(ev.run_id or "-", []).append(ev)
    return {rid: trace_from_events(evs) for rid, evs in by_run.items()}
