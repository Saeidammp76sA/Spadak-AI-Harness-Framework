"""Assertions over a Trace — deterministic pass/fail checks for eval suites."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from spadak.evaluation.trace import Trace


@dataclass
class AssertionResult:
    ok: bool
    name: str
    detail: str = ""


@dataclass
class Assertion:
    name: str
    fn: Callable[[Trace], tuple[bool, str]]

    def evaluate(self, trace: Trace) -> AssertionResult:
        ok, detail = self.fn(trace)
        return AssertionResult(ok=ok, name=self.name, detail=detail)


def assert_contains(substring: str, case: bool = False) -> Assertion:
    def check(tr: Trace) -> tuple[bool, str]:
        out = tr.output or ""
        hay, needle = (out, substring) if case else (out.lower(), substring.lower())
        return (needle in hay), f"expected {substring!r} in output"
    return Assertion(f"contains:{substring}", check)


def assert_regex(pattern: str) -> Assertion:
    rx = re.compile(pattern)

    def check(tr: Trace) -> tuple[bool, str]:
        return bool(rx.search(tr.output or "")), f"output must match /{pattern}/"
    return Assertion(f"regex:{pattern}", check)


def assert_tool_used(tool_name: str) -> Assertion:
    def check(tr: Trace) -> tuple[bool, str]:
        used = tool_name in tr.tool_calls()
        return used, f"tool {tool_name!r} should have been called; got {tr.tool_calls()}"
    return Assertion(f"tool_used:{tool_name}", check)


def assert_tool_not_used(tool_name: str) -> Assertion:
    def check(tr: Trace) -> tuple[bool, str]:
        return tool_name not in tr.tool_calls(), f"tool {tool_name!r} must not be called"
    return Assertion(f"tool_not_used:{tool_name}", check)


def assert_finish_reason(reason: str) -> Assertion:
    def check(tr: Trace) -> tuple[bool, str]:
        return tr.finish_reason == reason, f"finish_reason={tr.finish_reason!r} != {reason!r}"
    return Assertion(f"finish:{reason}", check)


def assert_max_steps(limit: int) -> Assertion:
    def check(tr: Trace) -> tuple[bool, str]:
        n_llm = sum(1 for s in tr.steps if s.kind == "llm")
        return n_llm <= limit, f"{n_llm} LLM steps > {limit}"
    return Assertion(f"max_steps:{limit}", check)
