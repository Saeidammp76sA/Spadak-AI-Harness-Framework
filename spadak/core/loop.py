"""The default agent step: LLM call -> guardrails -> tool execution -> repeat.

Kept as a free function so custom agents can replace it wholesale while the
runtime stays identical.  Cross-cutting concerns (safety, events, tracing)
are invoked through injected collaborators, never hard-coded here.
"""
from __future__ import annotations

import time
from typing import Any

from spadak.core.events import EventBus, EventType
from spadak.core.messages import Message, ToolCall, ToolResult
from spadak.core.state import State, StepOutcome


def default_step(agent: Any, state: State) -> StepOutcome:
    llm = agent.llm
    if llm is None:
        raise RuntimeError(f"Agent {agent.config.name!r} has no LLM attached")

    bus: EventBus | None = agent.bus
    run_id = state.scratchpad.get("run_id", "")

    # ---- context assembly: memory window + prompt compression hook ----
    ctx_messages = _build_context(agent, state)

    if bus:
        bus.emit(EventType.LLM_REQUEST, source=agent.config.name, run_id=run_id,
                 step=state.step, n_messages=len(ctx_messages))

    t0 = time.perf_counter()
    response = llm.complete(
        messages=ctx_messages,
        tools=_tool_specs_for(agent),
        temperature=agent.config.temperature,
        model=agent.config.model,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.tokens_used += getattr(response, "tokens_in", 0) + getattr(response, "tokens_out", 0)
    state.cost_usd += getattr(response, "cost_usd", 0.0) or 0.0

    assistant_msg = Message.assistant(
        text=response.text or "",
        calls=tuple(response.tool_calls),
        sender=agent.config.name,
        tokens=getattr(response, "tokens_out", 0),
        latency_ms=round(latency_ms, 2),
        model=getattr(response, "model", "?"),
    )
    state.add(assistant_msg)

    if bus:
        bus.emit(EventType.LLM_RESPONSE, source=agent.config.name, run_id=run_id,
                 step=state.step, text_len=len(response.text or ""),
                 n_tool_calls=len(response.tool_calls), latency_ms=round(latency_ms, 1))

    # ---- output guardrails (pre-tool / pre-final) ----
    guardrails = getattr(agent, "_guardrails", None) or []
    for g in guardrails:
        verdict = g.check_output(agent, state, assistant_msg)
        if verdict.blocked and bus:
            bus.emit(EventType.GUARDRAIL_TRIGGERED, source=g.__class__.__name__,
                     run_id=run_id, stage="output", reason=verdict.reason)
        if verdict.blocked:
            return StepOutcome(continue_run=False, stop_reason="guardrail",
                               delta_messages=[assistant_msg])
        if verdict.replaced_text is not None:
            assistant_msg = Message.assistant(text=verdict.replaced_text,
                                              calls=assistant_msg.tool_calls,
                                              sender=agent.config.name)
            state.messages[-1] = assistant_msg

    # ---- no tool calls => final answer ----
    if not response.tool_calls:
        return StepOutcome(continue_run=False, stop_reason="final",
                           delta_messages=[assistant_msg])

    # ---- execute tools (through safety checks + sandbox) ----
    deltas: list[Message] = [assistant_msg]
    tool_manager = agent.tools
    for call in response.tool_calls:
        result = _execute_tool(agent, state, call, tool_manager, guardrails, bus, run_id)
        msg = Message.tool(result, sender=call.name)
        state.add(msg)
        deltas.append(msg)

    return StepOutcome(continue_run=True, stop_reason=None, delta_messages=deltas)


def _execute_tool(agent: Any, state: State, call: ToolCall, manager: Any,
                  guardrails: list, bus: EventBus | None, run_id: str) -> ToolResult:
    if bus:
        bus.emit(EventType.TOOL_CALL, source=agent.config.name, run_id=run_id,
                 tool=call.name, args=call.arguments)

    t0 = time.perf_counter()
    error: str | None = None
    output: Any = None
    ok = True

    if manager is None:
        error = f"no tool registry attached; cannot run {call.name!r}"
        ok = False
    else:
        # input guardrails may veto or sanitize the call
        for g in guardrails:
            verdict = g.check_tool_call(agent, state, call)
            if verdict.blocked:
                error = f"blocked by guardrail: {verdict.reason}"
                ok = False
                break
            if verdict.replaced_args is not None:
                call = ToolCall(name=call.name, arguments=verdict.replaced_args,
                                call_id=call.call_id)
        if ok:
            try:
                output = manager.execute(call)
            except Exception as exc:  # noqa: BLE001 — harness must survive tool crashes
                ok = False
                error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - t0) * 1000.0
    result = ToolResult(call_id=call.call_id, name=call.name, output=output,
                        ok=ok, error=error, latency_ms=round(latency_ms, 2))

    if bus:
        bus.emit(EventType.TOOL_RESULT, source=agent.config.name, run_id=run_id,
                 tool=call.name, ok=ok, latency_ms=result.latency_ms,
                 error=error)
    return result


def _build_context(agent: Any, state: State) -> list[Message]:
    """Assemble the LLM context: system + memory-managed window."""
    messages = state.messages
    memory = getattr(agent, "memory", None)
    if memory is not None:
        messages = memory.context_window(messages,
                                         max_tokens=getattr(agent.config, "token_budget", None))
    return list(messages)


def _tool_specs_for(agent: Any) -> list[dict]:
    manager = getattr(agent, "tools", None)
    if manager is None:
        return []
    return manager.specs(allowed=agent.allowed_tools())
