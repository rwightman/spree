"""Evaluator-owned metrics for terminal activities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from .actions import Action
from .terminal import Observation

EvaluationSource = Literal["observation", "agent_action", "final_probe"]
EvaluationStatus = Literal["ok", "no_match", "error"]
ProbeTurnCost = Literal["none", "unknown", "game_turn"]
MetricExtractor = Callable[[Observation], Mapping[str, Any] | None]
ProbeReadiness = Callable[[Observation], bool]


@dataclass(frozen=True)
class EvaluationProbe:
    """One evaluator-owned terminal query.

    Probes do not consume agent decision ticks. ``turn_cost`` describes their
    effect on the game itself, which is separate from the runner's budget.
    """

    name: str
    action: Action
    ready: ProbeReadiness | None = field(default=None, repr=False, compare=False)
    turn_cost: ProbeTurnCost = "unknown"
    observe_timeout: float | None = None
    stable_ms: int | None = None
    byte_quiet_ms: int | None = None
    poll_interval: float | None = None
    prompt_fast_path: bool | None = None

    def is_ready(self, observation: Observation) -> bool:
        return self.ready is None or self.ready(observation)


@dataclass(frozen=True)
class EvaluationProfile:
    """Game-specific metric extraction and optional final status query."""

    name: str
    extractor: MetricExtractor = field(repr=False, compare=False)
    final_probe: EvaluationProbe | None = None


@dataclass
class EvaluationRecord:
    """One metric sample or evaluator error, independent of agent steps."""

    agent_id: str
    evaluator: str
    source: EvaluationSource
    status: EvaluationStatus
    decision_tick: int
    metrics: dict[str, Any] = field(default_factory=dict)
    final: bool = False
    visible_to_model: bool = False
    observation: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] | None = None
    execution: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    probe_turn_cost: ProbeTurnCost | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "agent_id": self.agent_id,
            "evaluator": self.evaluator,
            "source": self.source,
            "status": self.status,
            "decision_tick": self.decision_tick,
            "metrics": self.metrics,
            "final": self.final,
            "visible_to_model": self.visible_to_model,
            "observation": self.observation,
            "action": self.action,
            "execution": self.execution,
            "error": self.error,
            "probe_turn_cost": self.probe_turn_cost,
            "timestamp": self.timestamp,
        }


@dataclass
class EvaluationResult:
    """Evaluation records and convenient latest/final metric snapshots."""

    records: list[EvaluationRecord] = field(default_factory=list)

    @property
    def latest_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for record in self.records:
            if record.status == "ok":
                metrics.update(record.metrics)
        return metrics

    @property
    def final_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for record in self.records:
            if record.final and record.status == "ok":
                metrics.update(record.metrics)
        return metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "latest_metrics": self.latest_metrics,
            "final_metrics": self.final_metrics,
            "records": [record.to_dict() for record in self.records],
        }
