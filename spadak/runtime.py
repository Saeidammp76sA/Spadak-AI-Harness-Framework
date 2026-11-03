"""Spadak runtime — the integration layer that wires all sub-frameworks.

`Harness` owns: event bus, LLM provider(s), tool registry, memory, agents,
guardrails.  `HarnessBuilder` is the fluent composition API.  This is where
the separate frameworks (core/llm/memory/tools/orchestration/evaluation/
safety) become ONE coherent product — while each remains independently
importable and testable.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from spadak.core.agent import Agent, AgentConfig, AgentOutput, FinishReason
from spadak.core.events import EventBus, EventType
from spadak.core.messages import Message
from spadak.core.state import State
from spadak.llm.base import LLMProvider
from spadak.memory.long_term import LongTermMemory
from spadak.orchestration.handoff import Handoff
from spadak.safety.guardrail import Guardrail, Verdict
from spadak.tools.registry import ToolRegistry


class Harness:
    """Single entry point for running agents with every framework attached."""

    def __init__(self, llm: LLMProvider, tools: ToolRegistry | None = None,
                 memory: LongTermMemory | None = None,
                 guardrails: list[Guardrail] | None = None,
                 bus: EventBus | None = None) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory
        self.guardrails = guardrails or []
        self.bus = bus or EventBus()
        self.agents: dict[str, Agent] = {}
        self.default_agent: str | None = None
        # orchestration hooks (set by Supervisor / Swarm helpers)
        self.router: Callable[[str, Any], Any] | None = None
        self.swarm_enabled = False
        self.max_handoffs = 6

    # ------------------------------------------------------------------
    def add_agent(self, config: AgentConfig, tools: ToolRegistry | None = None,
                  llm: LLMProvider | None = None,
                  guardrails: list[Guardrail] | None = None) -> Agent:
        agent = Agent(config=config,
                      llm=llm or self.llm,
                      tools=tools or self.tools,
                      memory=self.memory,
                      bus=self.bus)
        # per-agent guardrails override global ones when provided
        agent._guardrails = guardrails if guardrails is not None else self.guardrails
        self.agents[config.name] = agent
        if self.default_agent is None:
            self.default_agent = config.name
        return agent

    # ------------------------------------------------------------------
    def run(self, user_input: str, agent: str | None = None,
            state: State | None = None) -> AgentOutput:
        """Run one full turn-loop for a single agent."""
        ag = self._agent_or_default(agent)
        run_id = uuid.uuid4().hex[:10]
        st = state or State()
        st.scratchpad["run_id"] = run_id
        st.scratchpad.setdefault("handoffs", 0)

        self.bus.emit(EventType.RUN_STARTED, source=ag.config.name,
                      run_id=run_id, input=user_input)

        # ---- input guardrails ----
        for g in getattr(ag, "_guardrails", []):
            verdict = g.check_input(ag, user_input)
            if verdict.blocked:
                self.bus.emit(EventType.GUARDRAIL_TRIGGERED, source=g.name,
                              run_id=run_id, stage="input", reason=verdict.reason)
                self.bus.emit(EventType.RUN_FINISHED, source=ag.config.name,
                              run_id=run_id, output=None, finish_reason="guardrail_blocked")
                return AgentOutput(answer="", finish_reason=FinishReason.GUARDRAIL_BLOCKED,
                                   state=st,
                                   artifacts={"reason": verdict.reason})
            if verdict.replaced_text is not None:
                user_input = verdict.replaced_text

        st.scratchpad["user_input"] = user_input
        ag.prepare(st, user_input)

        # ---- main loop ----
        while not st.finished and st.step < ag.config.max_steps:
            st.step += 1
            self.bus.emit(EventType.STEP_STARTED, source=ag.config.name,
                          run_id=run_id, step=st.step)
            try:
                outcome = ag.step(st)
            except Exception as exc:  # noqa: BLE001 — harness survives everything
                self.bus.emit(EventType.ERROR, source=ag.config.name,
                              run_id=run_id, error=f"{type(exc).__name__}: {exc}")
                return self._finish(ag, st, run_id, "", FinishReason.ERROR,
                                   error=str(exc))

            budget_hit = ag.budget_check(st)
            if budget_hit:
                return self._finish(ag, st, run_id, st.output or "",
                                    FinishReason.BUDGET_EXCEEDED, budget=budget_hit)

            # handoff detection from tool results
            target = _detect_handoff(outcome.delta_messages)
            if target:
                if st.scratchpad["handoffs"] >= self.max_handoffs:
                    return self._finish(ag, st, run_id, st.output or "(handoff loop limit)",
                                        FinishReason.MAX_STEPS)
                st.scratchpad["handoffs"] += 1
                nxt = self._agent_or_default(target)
                self.bus.emit(EventType.HANDOFF, source=ag.config.name,
                              run_id=run_id, to=target)
                ag = nxt
                continue

            if not outcome.continue_run:
                st.finished = True
                st.output = outcome.delta_messages[-1].text if outcome.delta_messages else ""
                reason = _map_stop(outcome.stop_reason)
                return self._finish(ag, st, run_id, st.output or "", reason)

            self.bus.emit(EventType.STEP_FINISHED, source=ag.config.name,
                          run_id=run_id, step=st.step)

        return self._finish(ag, st, run_id, st.output or "", FinishReason.MAX_STEPS)

    # ------------------------------------------------------------------
    def run_routed(self, user_input: str) -> AgentOutput:
        """Supervisor pattern: router picks agents until it says FINISH."""
        if self.router is None:
            raise RuntimeError("no router configured; use Supervisor(...) or set harness.router")
        shared = State()
        current: str | None = None
        last_output: AgentOutput | None = None
        for _ in range(8):  # safety bound on routing cycles
            decision = self.router(user_input, shared.messages)
            if decision.next_agent is None:
                break
            current = decision.next_agent
            last_output = self.run(user_input, agent=current, state=shared)
            if last_output.finish_reason == FinishReason.FINAL:
                break
        if last_output is None:
            last_output = self.run(user_input, agent=current, state=shared)
        return last_output

    def run_swarm(self, user_input: str, entry: str) -> AgentOutput:
        return self.run(user_input, agent=entry)   # handoffs handled inside run()

    # ------------------------------------------------------------------
    def _agent_or_default(self, name: str | None) -> Agent:
        key = name or self.default_agent
        if key is None or key not in self.agents:
            raise KeyError(f"unknown agent {key!r}; registered: {list(self.agents)}")
        return self.agents[key]

    def _finish(self, ag: Agent, st: State, run_id: str, answer: str,
                reason: FinishReason, **artifacts: Any) -> AgentOutput:
        self.bus.emit(EventType.RUN_FINISHED, source=ag.config.name, run_id=run_id,
                      output=answer, finish_reason=reason.value,
                      steps=st.step, tokens=st.tokens_used, cost_usd=round(st.cost_usd, 6))
        if self.memory is not None and reason == FinishReason.FINAL:
            self.memory.capture_from_run(st.scratchpad.get("user_input", ""), answer)
        return AgentOutput(answer=answer, finish_reason=reason, state=st,
                           artifacts=dict(artifacts))


def _detect_handoff(messages: list[Message]) -> str | None:
    for m in messages:
        for r in m.tool_results:
            if isinstance(r.output, dict) and "handoff" in r.output:
                return str(r.output["handoff"])
    return None


def _map_stop(stop: str | None) -> FinishReason:
    return {
        "final": FinishReason.FINAL,
        "guardrail": FinishReason.GUARDRAIL_BLOCKED,
        "max_steps": FinishReason.MAX_STEPS,
        "budget": FinishReason.BUDGET_EXCEEDED,
        "error": FinishReason.ERROR,
    }.get(stop or "final", FinishReason.FINAL)


# =============================================================================
class HarnessBuilder:
    """Fluent assembly of the five sub-frameworks into one harness."""

    def __init__(self) -> None:
        self._llm: LLMProvider | None = None
        self._tools = ToolRegistry()
        self._memory: LongTermMemory | None = None
        self._guardrails: list[Guardrail] = []
        self._bus = EventBus()
        self._agents: list[tuple[AgentConfig, dict[str, Any]]] = []

    def llm(self, provider: LLMProvider) -> "HarnessBuilder":
        self._llm = provider
        return self

    def tools(self, *objs: Any) -> "HarnessBuilder":
        self._tools.register_all(objs)
        return self

    def memory(self, mem: LongTermMemory) -> "HarnessBuilder":
        self._memory = mem
        return self

    def guardrails(self, *guards: Guardrail) -> "HarnessBuilder":
        self._guardrails.extend(guards)
        return self

    def bus(self, bus: EventBus) -> "HarnessBuilder":
        self._bus = bus
        return self

    def agent(self, config: AgentConfig, **overrides: Any) -> "HarnessBuilder":
        self._agents.append((config, overrides))
        return self

    def build(self) -> Harness:
        if self._llm is None:
            raise ValueError("HarnessBuilder: an LLM provider is required (.llm(...))")
        h = Harness(llm=self._llm, tools=self._tools, memory=self._memory,
                    guardrails=self._guardrails, bus=self._bus)
        for cfg, overrides in self._agents:
            h.add_agent(cfg, **overrides)
        return h
