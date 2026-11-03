"""spadak.orchestration — multi-agent patterns sub-framework."""
from spadak.orchestration.handoff import Handoff, make_transfer_tool
from spadak.orchestration.supervisor import Supervisor, supervisor_policy
from spadak.orchestration.pipeline import Pipeline, SequentialPipeline
from spadak.orchestration.swarm import Swarm

__all__ = [
    "Handoff", "make_transfer_tool",
    "Supervisor", "supervisor_policy",
    "Pipeline", "SequentialPipeline",
    "Swarm",
]
