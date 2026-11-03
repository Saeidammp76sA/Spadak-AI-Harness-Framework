"""EvalSuite: run many cases against a harness factory, aggregate reports.

This is the regression-testing backbone — CI-friendly because everything
can run offline with MockLLM (hermetic evals).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from spadak.evaluation.assertions import Assertion, AssertionResult
from spadak.evaluation.judge import JudgeVerdict, RuleJudge
from spadak.evaluation.metrics import RunMetrics, compute_metrics
from spadak.evaluation.trace import Trace, trace_from_events


@dataclass
class EvalCase:
    name: str
    input: str
    assertions: list[Assertion] = field(default_factory=list)
    judge: RuleJudge | None = None
    # optional per-case harness override, e.g. lambda: HarnessBuilder()...build()
    factory: Callable[[], Any] | None = None


@dataclass
class CaseResult:
    case: str
    passed: bool
    trace: Trace
    metrics: RunMetrics
    assertion_results: list[AssertionResult] = field(default_factory=list)
    verdict: JudgeVerdict | None = None


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def summary(self) -> dict[str, Any]:
        lat = [r.metrics.latency_ms for r in self.results]
        steps = [r.metrics.steps for r in self.results]
        return {
            "cases": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "pass_rate": round(self.pass_rate, 4),
            "avg_steps": round(statistics.mean(steps), 2) if steps else 0,
            "avg_llm_latency_ms": round(statistics.mean(lat), 1) if lat else 0,
            "total_cost_usd": round(sum(r.metrics.cost_usd for r in self.results), 6),
        }

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]


class EvalSuite:
    """`harness_factory()` must return an object with `.run(agent_name, text)`."""

    def __init__(self, harness_factory: Callable[[], Any],
                 agent_name: str = "assistant") -> None:
        self.harness_factory = harness_factory
        self.agent_name = agent_name
        self.cases: list[EvalCase] = []

    def add(self, case: EvalCase) -> "EvalSuite":
        self.cases.append(case)
        return self

    def run_case(self, case: EvalCase) -> CaseResult:
        harness = case.factory() if case.factory else self.harness_factory()
        before = len(harness.bus.history())
        output = harness.run(case.input, agent=case.name and self.agent_name or self.agent_name)
        events = harness.bus.history()[before:]
        trace = trace_from_events(events)
        metrics = compute_metrics(output.state, events)

        a_results = [a.evaluate(trace) for a in case.assertions]
        verdict = case.judge.judge(case.input, trace.output or "") if case.judge else None

        passed = all(r.ok for r in a_results) and (verdict is None or verdict.passed)
        return CaseResult(case=case.name, passed=passed, trace=trace, metrics=metrics,
                          assertion_results=a_results, verdict=verdict)

    def run_all(self) -> EvalReport:
        report = EvalReport()
        for case in self.cases:
            report.results.append(self.run_case(case))
        return report
