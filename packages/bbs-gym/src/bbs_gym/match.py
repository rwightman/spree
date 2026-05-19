"""Scheduled multi-agent match orchestration."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tty_agent.runner import ActivityBudget, ActivityResult, ActivityRunner, PreparedActivityStep

from .accounts import AccountConfigError
from .env import BbsGym

MatchSchedulerMode = Literal["sequential", "parallel_race", "parallel_barrier", "continuous"]
MatchOrder = Literal["fixed", "shuffle", "rotate"]
DisconnectPolicy = Literal["stop", "reconnect"]


@dataclass(frozen=True)
class MatchParticipantSpec:
    agent_id: str
    provider: str | None = None
    model: str | None = None
    config: dict[str, Any] | None = None


@dataclass
class MatchParticipantRuntime:
    spec: MatchParticipantSpec
    args: Any
    model: object
    model_metadata: dict[str, object]
    runner: ActivityRunner
    log_path: Path
    reconnects: int = 0


@dataclass(frozen=True)
class MatchSchedulerConfig:
    mode: MatchSchedulerMode = "sequential"
    order: MatchOrder = "fixed"
    seed: int | None = None
    disconnect_policy: DisconnectPolicy = "stop"
    max_reconnects: int = 3
    reconnect_delay: float = 2.0
    max_rounds: int = 50
    max_decision_ticks: int = 50
    max_wall_seconds: float = 600.0
    max_workers: int | None = None


@dataclass(frozen=True)
class MatchRunResult:
    rounds: int
    results: list[tuple[MatchParticipantRuntime, ActivityResult]]


def run_scheduled_match(
        gym: BbsGym,
        participants: list[MatchParticipantRuntime],
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
) -> MatchRunResult:
    states: list[tuple[MatchParticipantRuntime, Any]] = []
    for participant in participants:
        agent = gym.connect(participant.spec.agent_id, model_metadata=participant.model_metadata)
        state = participant.runner.start_state(
            agent,
            participant.model,
            ActivityBudget(
                max_decision_ticks=scheduler.max_decision_ticks,
                max_wall_seconds=scheduler.max_wall_seconds,
            ),
        )
        states.append((participant, state))

    if scheduler.mode == "sequential":
        round_number = _run_sequential_match(gym, states, scheduler, match_log_path)
    elif scheduler.mode == "parallel_barrier":
        round_number = _run_parallel_barrier_match(gym, states, scheduler, match_log_path)
    elif scheduler.mode == "parallel_race":
        round_number = _run_parallel_race_match(gym, states, scheduler, match_log_path)
    elif scheduler.mode == "continuous":
        raise ValueError("continuous requires the runner phase split planned for the next scheduler pass")
    else:
        raise ValueError(f"unknown match scheduler mode: {scheduler.mode}")

    if round_number >= scheduler.max_rounds:
        for _, state in states:
            if not state.completed:
                state.stop_reason = "match_rounds"
                state.completed = True

    return MatchRunResult(
        rounds=round_number,
        results=[(participant, participant.runner.finish_state(state)) for participant, state in states],
    )


def match_round_order(
        states: list[tuple[MatchParticipantRuntime, Any]],
        order: str,
        rng: random.Random,
        round_number: int,
) -> list[tuple[MatchParticipantRuntime, Any]]:
    active = [(participant, state) for participant, state in states if not state.completed]
    if order == "fixed":
        return active
    if order == "shuffle":
        shuffled = list(active)
        rng.shuffle(shuffled)
        return shuffled
    if order == "rotate":
        if not active:
            return []
        offset = (round_number - 1) % len(active)
        return active[offset:] + active[:offset]
    raise ValueError(f"unknown match order: {order}")


def handle_match_disconnect(
        gym: BbsGym,
        participant: MatchParticipantRuntime,
        state: Any,
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
        round_number: int,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "participant_disconnected",
            "round": round_number,
            "agent_id": participant.spec.agent_id,
            "reconnects": participant.reconnects,
            "disconnect_policy": scheduler.disconnect_policy,
            "timestamp": time.time(),
        },
    )
    if scheduler.disconnect_policy == "stop":
        return

    _close_agent(state.agent)
    for attempt in range(participant.reconnects + 1, scheduler.max_reconnects + 1):
        if scheduler.reconnect_delay:
            time.sleep(scheduler.reconnect_delay)
        try:
            state.agent = gym.connect(participant.spec.agent_id, model_metadata=participant.model_metadata)
        except (OSError, AccountConfigError, ValueError) as exc:
            participant.reconnects = attempt
            _write_match_event(
                match_log_path,
                {
                    "type": "participant_reconnect_failed",
                    "round": round_number,
                    "agent_id": participant.spec.agent_id,
                    "attempt": attempt,
                    "error": str(exc),
                    "timestamp": time.time(),
                },
            )
            continue

        participant.reconnects = attempt
        state.completed = False
        state.stop_reason = ""
        _write_match_event(
            match_log_path,
            {
                "type": "participant_reconnected",
                "round": round_number,
                "agent_id": participant.spec.agent_id,
                "attempt": attempt,
                "timestamp": time.time(),
            },
        )
        return

    state.stop_reason = "disconnect_reconnect_failed"
    state.completed = True


def write_match_event(path: Path, event: dict[str, Any]) -> None:
    _write_match_event(path, event)


def _run_sequential_match(
        gym: BbsGym,
        states: list[tuple[MatchParticipantRuntime, Any]],
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
) -> int:
    rng = random.Random(scheduler.seed)
    round_number = 0
    while round_number < scheduler.max_rounds and any(not state.completed for _, state in states):
        round_number += 1
        scheduled_states = match_round_order(states, scheduler.order, rng, round_number)
        _write_round_started(match_log_path, round_number, scheduled_states, scheduler)
        _write_match_event(
            match_log_path,
            {
                "type": "commit_order",
                "round": round_number,
                "order": [participant.spec.agent_id for participant, _ in scheduled_states],
                "match_order": scheduler.order,
                "match_seed": scheduler.seed,
                "timestamp": time.time(),
            },
        )
        for participant, state in scheduled_states:
            started_at = time.monotonic()
            _write_match_event(
                match_log_path,
                {
                    "type": "agent_step_started",
                    "round": round_number,
                    "agent_id": participant.spec.agent_id,
                    "timestamp": time.time(),
                },
            )
            step = participant.runner.run_step(state)
            _write_match_event(
                match_log_path,
                {
                    "type": "agent_step_completed",
                    "round": round_number,
                    "agent_id": participant.spec.agent_id,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "timestamp": time.time(),
                },
            )
            _write_agent_step_event(match_log_path, round_number, participant, state, step)
            if state.completed and state.stop_reason == "disconnected":
                handle_match_disconnect(gym, participant, state, scheduler, match_log_path, round_number)
        _write_round_completed(match_log_path, round_number, states)
    return round_number


def _run_parallel_barrier_match(
        gym: BbsGym,
        states: list[tuple[MatchParticipantRuntime, Any]],
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
) -> int:
    rng = random.Random(scheduler.seed)
    round_number = 0
    while round_number < scheduler.max_rounds and any(not state.completed for _, state in states):
        round_number += 1
        scheduled_states = match_round_order(states, scheduler.order, rng, round_number)
        _write_round_started(match_log_path, round_number, scheduled_states, scheduler)
        preparations = _prepare_parallel_steps(scheduled_states, scheduler, match_log_path, round_number)
        _write_match_event(
            match_log_path,
            {
                "type": "commit_order",
                "round": round_number,
                "order": [participant.spec.agent_id for participant, _ in scheduled_states],
                "match_order": scheduler.order,
                "match_seed": scheduler.seed,
                "commit_policy": "barrier_order",
                "timestamp": time.time(),
            },
        )
        for participant, state in scheduled_states:
            prepared = preparations.get(participant.spec.agent_id)
            _commit_parallel_step(gym, participant, state, prepared, scheduler, match_log_path, round_number)
        _write_round_completed(match_log_path, round_number, states)
    return round_number


def _run_parallel_race_match(
        gym: BbsGym,
        states: list[tuple[MatchParticipantRuntime, Any]],
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
) -> int:
    rng = random.Random(scheduler.seed)
    round_number = 0
    while round_number < scheduler.max_rounds and any(not state.completed for _, state in states):
        round_number += 1
        scheduled_states = match_round_order(states, scheduler.order, rng, round_number)
        _write_round_started(match_log_path, round_number, scheduled_states, scheduler)
        commit_order: list[str] = []
        futures: dict[Future[PreparedActivityStep | None], tuple[MatchParticipantRuntime, Any, float]] = {}
        with ThreadPoolExecutor(max_workers=_max_workers(scheduler, len(scheduled_states))) as executor:
            for participant, state in scheduled_states:
                _write_agent_step_started(match_log_path, round_number, participant)
                futures[executor.submit(participant.runner.prepare_step, state)] = (
                    participant,
                    state,
                    time.monotonic(),
                )
            for future in as_completed(futures):
                participant, state, started_at = futures[future]
                prepared = _future_preparation(future, participant, state, match_log_path, round_number)
                _write_agent_decision_completed(
                    match_log_path,
                    round_number,
                    participant,
                    elapsed_seconds=time.monotonic() - started_at,
                )
                commit_order.append(participant.spec.agent_id)
                _commit_parallel_step(gym, participant, state, prepared, scheduler, match_log_path, round_number)
        _write_match_event(
            match_log_path,
            {
                "type": "commit_order",
                "round": round_number,
                "order": commit_order,
                "match_order": scheduler.order,
                "match_seed": scheduler.seed,
                "commit_policy": "race_completion",
                "timestamp": time.time(),
            },
        )
        _write_round_completed(match_log_path, round_number, states)
    return round_number


def _prepare_parallel_steps(
        scheduled_states: list[tuple[MatchParticipantRuntime, Any]],
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
        round_number: int,
) -> dict[str, PreparedActivityStep | None]:
    preparations: dict[str, PreparedActivityStep | None] = {}
    futures: dict[Future[PreparedActivityStep | None], tuple[MatchParticipantRuntime, Any, float]] = {}
    with ThreadPoolExecutor(max_workers=_max_workers(scheduler, len(scheduled_states))) as executor:
        for participant, state in scheduled_states:
            _write_agent_step_started(match_log_path, round_number, participant)
            futures[executor.submit(participant.runner.prepare_step, state)] = (
                participant,
                state,
                time.monotonic(),
            )
        for future in as_completed(futures):
            participant, state, started_at = futures[future]
            preparations[participant.spec.agent_id] = _future_preparation(
                future,
                participant,
                state,
                match_log_path,
                round_number,
            )
            _write_agent_decision_completed(
                match_log_path,
                round_number,
                participant,
                elapsed_seconds=time.monotonic() - started_at,
            )
    return preparations


def _future_preparation(
        future: Future[PreparedActivityStep | None],
        participant: MatchParticipantRuntime,
        state: Any,
        match_log_path: Path,
        round_number: int,
) -> PreparedActivityStep | None:
    try:
        return future.result()
    except Exception as exc:
        state.stop_reason = "scheduler_error"
        state.completed = True
        _write_match_event(
            match_log_path,
            {
                "type": "agent_step_failed",
                "round": round_number,
                "agent_id": participant.spec.agent_id,
                "error": str(exc),
                "timestamp": time.time(),
            },
        )
        return None


def _commit_parallel_step(
        gym: BbsGym,
        participant: MatchParticipantRuntime,
        state: Any,
        prepared: PreparedActivityStep | None,
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
        round_number: int,
) -> None:
    started_at = time.monotonic()
    step = participant.runner.commit_prepared_step(state, prepared)
    _write_match_event(
        match_log_path,
        {
            "type": "agent_step_completed",
            "round": round_number,
            "agent_id": participant.spec.agent_id,
            "elapsed_seconds": time.monotonic() - started_at,
            "timestamp": time.time(),
        },
    )
    _write_agent_step_event(match_log_path, round_number, participant, state, step)
    if state.completed and state.stop_reason == "disconnected":
        handle_match_disconnect(gym, participant, state, scheduler, match_log_path, round_number)


def _max_workers(scheduler: MatchSchedulerConfig, active_count: int) -> int:
    if active_count <= 0:
        return 1
    if scheduler.max_workers is None:
        return active_count
    return max(1, min(scheduler.max_workers, active_count))


def _write_round_started(
        match_log_path: Path,
        round_number: int,
        scheduled_states: list[tuple[MatchParticipantRuntime, Any]],
        scheduler: MatchSchedulerConfig,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "round_started",
            "round": round_number,
            "agents": [participant.spec.agent_id for participant, _ in scheduled_states],
            "scheduler_mode": scheduler.mode,
            "match_order": scheduler.order,
            "match_seed": scheduler.seed,
            "timestamp": time.time(),
        },
    )


def _write_agent_step_started(match_log_path: Path, round_number: int, participant: MatchParticipantRuntime) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "agent_step_started",
            "round": round_number,
            "agent_id": participant.spec.agent_id,
            "timestamp": time.time(),
        },
    )


def _write_agent_decision_completed(
        match_log_path: Path,
        round_number: int,
        participant: MatchParticipantRuntime,
        elapsed_seconds: float,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "agent_decision_completed",
            "round": round_number,
            "agent_id": participant.spec.agent_id,
            "elapsed_seconds": elapsed_seconds,
            "timestamp": time.time(),
        },
    )


def _write_round_completed(
        match_log_path: Path,
        round_number: int,
        states: list[tuple[MatchParticipantRuntime, Any]],
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "round_completed",
            "round": round_number,
            "active_agents": [participant.spec.agent_id for participant, state in states if not state.completed],
            "timestamp": time.time(),
        },
    )


def _write_agent_step_event(
        match_log_path: Path,
        round_number: int,
        participant: MatchParticipantRuntime,
        state: Any,
        step: Any,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "agent_step",
            "round": round_number,
            "agent_id": participant.spec.agent_id,
            "step": step.step if step is not None else None,
            "completed": state.completed,
            "stop_reason": state.stop_reason if state.completed else "",
            "active_profile": (state.active_profile.name if state.active_profile is not None else ""),
            "action": step.action if step is not None else None,
            "agent_log_path": str(participant.log_path),
            "timestamp": step.timestamp if step is not None else None,
        },
    )


def _write_match_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _close_agent(agent: object) -> None:
    close = getattr(agent, "close", None)
    if callable(close):
        close()
