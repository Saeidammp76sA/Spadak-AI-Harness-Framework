"""Regression eval suite demo — runs fully offline via MockLLM."""
from __future__ import annotations

from spadak import AgentConfig, HarnessBuilder
from spadak.evaluation.assertions import (assert_contains, assert_finish_reason,
                                          assert_tool_used)
from spadak.evaluation.judge import RuleJudge
from spadak.evaluation.suite import EvalCase, EvalReport, EvalSuite
from spadak.llm import LLMResponse, MockLLM, ScriptedBehavior
from spadak.memory import LongTermMemory
from spadak.safety import PromptInjectionGuard
from spadak.tools.builtin import register_builtins


def weather_factory():
    llm = ScriptedBehavior.echo_with_keyword_tool(
        "Shiraz today: rainy, 12°C. Bring an umbrella.")
    return (HarnessBuilder()
            .llm(llm)
            .memory(LongTermMemory())
            .guardrails(PromptInjectionGuard())
            .agent(AgentConfig(name="assistant",
                               system_prompt="Weather assistant.",
                               tools=["get_weather"]))
            .build())


def calculator_factory():
    llm = ScriptedBehavior.tool_then_final("calculator", {"expression": "(3+4)*5"},
                                           "The result is 35.")
    return (HarnessBuilder()
            .llm(llm)
            .agent(AgentConfig(name="assistant", tools=["calculator"]))
            .build())


def build_suite() -> EvalSuite:
    suite = EvalSuite(weather_factory, agent_name="assistant")
    suite.add(EvalCase(
        name="weather-uses-tool",
        input="What is the weather in Shiraz?",
        assertions=[assert_tool_used("get_weather"),
                    assert_contains("shiraz"),
                    assert_finish_reason("final")],
        judge=RuleJudge(must_contain=["rain"], must_not_contain=["unknown"]),
    ))
    calc_suite_case = EvalCase(
        name="calculator-grounded",
        input="Compute (3+4)*5",
        assertions=[assert_tool_used("calculator"), assert_contains("35")],
        factory=calculator_factory,
    )
    suite.add(calc_suite_case)
    return suite


def run(verbose: bool = True) -> EvalReport:
    report = build_suite().run_all()
    if verbose:
        for r in report.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"[{mark}] {r.case}: steps={r.metrics.steps} "
                  f"tools={r.trace.tool_calls()} judge={getattr(r.verdict, 'score', None)}")
            for a in r.assertion_results:
                if not a.ok:
                    print("   assertion failed:", a.name, "-", a.detail)
    return report


if __name__ == "__main__":
    rep = run()
    print(rep.summary())
    raise SystemExit(0 if rep.pass_rate == 1.0 else 1)
