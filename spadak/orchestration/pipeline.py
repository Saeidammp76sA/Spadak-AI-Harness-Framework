"""Pipeline pattern: agents chained sequentially; each sees prior outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PipelineStage:
    agent: str                                   # registered agent name
    transform: Callable[[str], str] | None = None  # optional input munging


class Pipeline:
    def __init__(self, harness: Any, stages: list[PipelineStage | str]) -> None:
        self.harness = harness
        self.stages: list[PipelineStage] = [
            s if isinstance(s, PipelineStage) else PipelineStage(agent=s) for s in stages
        ]

    def run(self, user_input: str) -> "PipelineResult":
        current = user_input
        outputs: dict[str, str] = {}
        for stage in self.stages:
            inp = stage.transform(current) if stage.transform else current
            out = self.harness.run(inp, agent=stage.agent)
            answer = out.answer or ""
            outputs[stage.agent] = answer
            current = answer          # next stage works on previous output
        return PipelineResult(final=current, by_stage=outputs)


@dataclass
class PipelineResult:
    final: str
    by_stage: dict[str, str] = field(default_factory=dict)


SequentialPipeline = Pipeline   # explicit alias for readability
