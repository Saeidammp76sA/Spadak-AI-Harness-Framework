"""spadak.evaluation — traces, metrics, assertions, LLM-as-judge, suites."""
from spadak.evaluation.trace import Trace, TraceStep, capture_traces
from spadak.evaluation.metrics import RunMetrics, compute_metrics
from spadak.evaluation.assertions import Assertion, assert_contains, assert_regex, assert_tool_used, assert_finish_reason
from spadak.evaluation.judge import RuleJudge, LLMJudge, JudgeVerdict
from spadak.evaluation.suite import EvalCase, EvalSuite, EvalReport

__all__ = [
    "Trace", "TraceStep", "capture_traces",
    "RunMetrics", "compute_metrics",
    "Assertion", "assert_contains", "assert_regex", "assert_tool_used", "assert_finish_reason",
    "RuleJudge", "LLMJudge", "JudgeVerdict",
    "EvalCase", "EvalSuite", "EvalReport",
]
