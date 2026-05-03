from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Condition(str, Enum):
    S0 = "S0"
    S1A = "S1a"
    S1B = "S1b"


@dataclass
class ToolExecutionResult:
    tool_id: int
    tool_name: str
    tool_output: str
    evidence: list[dict[str, Any]]
    sources: list[str]
    latency_ms: float
    token_cost_estimate: int
    execution_success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionRecord:
    session_id: str
    benchmark: str
    condition: str
    question_id: str
    question_type: str
    query: str
    profile: dict[str, Any]
    profile_label: str
    selected_tool_id: int
    selected_tool_name: str
    task_optimal_tool_id: int
    task_tsa_hit: bool
    profile_optimal_tool_id: int
    profile_tsa_hit: bool
    profile_eval_eligible: bool
    optimal_tool_id: int
    tsa_hit: bool
    pts_delta: float
    input_tokens: int
    latency_ms: float
    execution_success: bool
    retrieved_evidence: list[dict[str, Any]]
    final_response: str
    tool_result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
