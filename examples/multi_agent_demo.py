"""Multi-agent demos: supervisor routing + swarm handoffs + pipeline (offline)."""
from __future__ import annotations

from spadak import AgentConfig, HarnessBuilder
from spadak.core.messages import ToolCall
from spadak.llm import LLMResponse, MockLLM
from spadak.orchestration import Pipeline, Supervisor, Swarm
from spadak.orchestration.handoff import make_transfer_tool
from spadak.tools.registry import ToolRegistry


def _supervisor_demo(verbose: bool = True) -> bool:
    # router script: first send to "math", then FINISH
    router_llm = MockLLM(script=[
        LLMResponse(text='{"agent": "math", "reason": "arithmetic task"}'),
        LLMResponse(text='{"agent": "FINISH", "reason": "done"}'),
    ])

    h = HarnessBuilder().llm(MockLLM(script=[LLMResponse(text="42")])).build()
    h.add_agent(AgentConfig(name="general", system_prompt="General helper"))
    h.add_agent(AgentConfig(name="math", system_prompt="Math specialist"))

    Supervisor(h, llm=router_llm, workers=["general", "math"])
    out = h.run_routed("What is 6 times 7?")
    if verbose:
        print("[supervisor] answer:", out.answer, "| via:", out.state.last()[0].sender)
    return out.answer == "42"


def _swarm_demo(verbose: bool = True) -> bool:
    transfer = make_transfer_tool("expert")
    triage_llm = MockLLM(script=[
        LLMResponse(text="", tool_calls=[
            ToolCall(name="transfer_to_expert", arguments={"reason": "needs depth"})]),
    ])
    expert_llm = MockLLM(script=[LLMResponse(text="Expert says: refactor the module.")])

    h = HarnessBuilder().llm(triage_llm).build()
    reg_triage = ToolRegistry().register_all([transfer])
    reg_expert = ToolRegistry()
    h.add_agent(AgentConfig(name="triage", system_prompt="Route issues.",
                            tools=["transfer_to_expert"]), tools=reg_triage)
    h.add_agent(AgentConfig(name="expert", system_prompt="Deep analysis."),
                tools=reg_expert, llm=expert_llm)

    Swarm(h).wire()
    out = h.run_swarm("Our API is slow under load.", entry="triage")
    if verbose:
        print("[swarm] answer:", out.answer, "| finish:", out.finish_reason.value)
    return "Expert" in out.answer


def _pipeline_demo(verbose: bool = True) -> bool:
    writer = MockLLM(script=[LLMResponse(text="draft: agents need evals")])
    editor = MockLLM(script=[LLMResponse(text="polished: AI agents require rigorous evaluation")])

    h = HarnessBuilder().llm(writer).build()
    h.add_agent(AgentConfig(name="writer"))
    h.add_agent(AgentConfig(name="editor"), llm=editor)

    result = Pipeline(h, ["writer", "editor"]).run("Write about agent harnesses")
    if verbose:
        print("[pipeline] final:", result.final)
    return "polished" in result.final


def run(verbose: bool = True) -> bool:
    return all(f(verbose) for f in (_supervisor_demo, _swarm_demo, _pipeline_demo))


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
