"""Judges: score agent answers. RuleJudge is deterministic; LLMJudge uses a
second model (any spadak LLMProvider) with a rubric prompt."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from spadak.core.messages import Message


@dataclass
class JudgeVerdict:
    passed: bool
    score: float = 0.0            # 0..1
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class RuleJudge:
    """Deterministic judge: required/forbidden substrings & regexes."""

    def __init__(self, must_contain: list[str] | None = None,
                 must_not_contain: list[str] | None = None,
                 must_match: list[str] | None = None) -> None:
        self.must_contain = must_contain or []
        self.must_not_contain = must_not_contain or []
        self.must_match = [re.compile(p, re.IGNORECASE) for p in (must_match or [])]

    def judge(self, question: str, answer: str) -> JudgeVerdict:
        fails: list[str] = []
        low = (answer or "").lower()
        for s in self.must_contain:
            if s.lower() not in low:
                fails.append(f"missing {s!r}")
        for s in self.must_not_contain:
            if s.lower() in low:
                fails.append(f"forbidden {s!r} present")
        for rx in self.must_match:
            if not rx.search(answer or ""):
                fails.append(f"no match /{rx.pattern}/")
        total = len(self.must_contain) + len(self.must_not_contain) + len(self.must_match)
        score = 1.0 if total == 0 else (total - len(fails)) / total
        return JudgeVerdict(passed=not fails, score=score,
                            rationale="; ".join(fails) or "all rules satisfied")


_JUDGE_PROMPT = """You are a strict evaluator. Grade the AGENT ANSWER against the QUESTION and RUBRIC.

QUESTION:
{question}

RUBRIC:
{rubric}

AGENT ANSWER:
{answer}

Reply ONLY with JSON: {{"pass": true|false, "score": 0.0-1.0, "reason": "one sentence"}}"""


class LLMJudge:
    """Model-based grading — inject any LLMProvider (incl. MockLLM in tests)."""

    def __init__(self, llm: Any, rubric: str = "Correctness, groundedness, completeness.") -> None:
        self.llm = llm
        self.rubric = rubric

    def judge(self, question: str, answer: str) -> JudgeVerdict:
        prompt = _JUDGE_PROMPT.format(question=question, rubric=self.rubric, answer=answer)
        resp = self.llm.complete(
            messages=[Message.user(prompt, sender="evaluator")],
            tools=[], temperature=0.0, model=None)
        text = resp.text or ""
        data = _extract_json(text)
        if data is None:
            return JudgeVerdict(passed=False, score=0.0,
                                rationale=f"judge output unparseable: {text[:120]}")
        return JudgeVerdict(passed=bool(data.get("pass")),
                            score=float(data.get("score", 0.0)),
                            rationale=str(data.get("reason", "")), raw=data)


def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
