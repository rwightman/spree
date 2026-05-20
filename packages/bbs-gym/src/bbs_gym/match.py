"""Scheduled multi-agent match orchestration.

Log events use ``round`` as the scheduled round number for ``sequential``,
``parallel_barrier``, and ``parallel_race``. In ``continuous`` mode there are no
all-agent rounds, so continuous scheduler events use ``tick`` instead.
Continuous ``commit_order`` events also include ``queued_tick`` for the original
decision request that produced the committed action.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

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
    model: object
    model_metadata: dict[str, object]
    runner: ActivityRunner
    log_path: Path
    # Runtime accounting mutated by the scheduler while a match is active.
    reconnects: int = 0


@dataclass(frozen=True)
class MatchSchedulerConfig:
    """Scheduler settings for a shared multi-agent match.

    ``parallel_race`` and ``continuous`` intentionally turn model decision
    latency into initiative. Use ``parallel_barrier`` when all models should
    decide concurrently but commit in a fair scheduled order.
    """

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
    commit_count: int
    results: list[tuple[MatchParticipantRuntime, ActivityResult]]


@runtime_checkable
class ClosableAgent(Protocol):
    def close(self) -> None: ...


def run_scheduled_match(
        gym: BbsGym,
        participants: list[MatchParticipantRuntime],
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
) -> MatchRunResult:
    match_started_at = time.monotonic()
    _write_match_started(match_log_path, participants, scheduler)
    states: list[tuple[MatchParticipantRuntime, Any]] = []
    scheduler_count = 0
    try:
        for participant in participants:
            agent = gym.connect(participant.spec.agent_id, model_metadata=participant.model_metadata)
            state = participant.runner.start_state(
                agent,
                participant.model,
                ActivityBudget(
                    max_decision_ticks=scheduler.max_decision_ticks,
                    max_wall_seconds=scheduler.max_wall_seconds,
                    started_at=match_started_at,
                ),
            )
            states.append((participant, state))

        if scheduler.mode == "sequential":
            scheduler_count = _run_sequential_match(gym, states, scheduler, match_log_path, match_started_at)
        elif scheduler.mode == "parallel_barrier":
            scheduler_count = _run_parallel_barrier_match(gym, states, scheduler, match_log_path, match_started_at)
        elif scheduler.mode == "parallel_race":
            scheduler_count = _run_parallel_race_match(gym, states, scheduler, match_log_path, match_started_at)
        elif scheduler.mode == "continuous":
            scheduler_count = _run_continuous_match(gym, states, scheduler, match_log_path, match_started_at)
        else:
            raise ValueError(f"unknown match scheduler mode: {scheduler.mode}")

        if _match_time_exhausted(scheduler, match_started_at):
            _mark_active_states_completed(states, "match_wall_seconds")
        elif scheduler_count >= scheduler.max_rounds:
            for _, state in states:
                if not state.completed:
                    state.stop_reason = _match_limit_stop_reason(scheduler)
                    state.completed = True

        results = [(participant, participant.runner.finish_state(state)) for participant, state in states]
    except Exception as exc:
        _write_match_completed(
            match_log_path,
            scheduler_count,
            0,
            [],
            scheduler,
            match_started_at,
            clean_exit=False,
            error=str(exc),
        )
        raise

    commit_count = _commit_count(results)
    _write_match_completed(
        match_log_path,
        scheduler_count,
        commit_count,
        results,
        scheduler,
        match_started_at,
    )
    return MatchRunResult(
        commit_count=commit_count,
        results=results,
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
        round_number: int | None,
        tick: int | None = None,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "participant_disconnected",
            **_event_clock(round_number, tick),
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
                    **_event_clock(round_number, tick),
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
                **_event_clock(round_number, tick),
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
        match_started_at: float,
) -> int:
    rng = random.Random(scheduler.seed)
    round_number = 0
    while (
        round_number < scheduler.max_rounds
        and any(not state.completed for _, state in states)
        and not _match_time_exhausted(scheduler, match_started_at)
    ):
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
            if _match_time_exhausted(scheduler, match_started_at):
                break
            started_at = time.monotonic()
            _write_agent_step_started(match_log_path, round_number, participant, phase="started")
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
        match_started_at: float,
) -> int:
    rng = random.Random(scheduler.seed)
    round_number = 0
    while (
        round_number < scheduler.max_rounds
        and any(not state.completed for _, state in states)
        and not _match_time_exhausted(scheduler, match_started_at)
    ):
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
        match_started_at: float,
) -> int:
    rng = random.Random(scheduler.seed)
    round_number = 0
    while (
        round_number < scheduler.max_rounds
        and any(not state.completed for _, state in states)
        and not _match_time_exhausted(scheduler, match_started_at)
    ):
        round_number += 1
        scheduled_states = match_round_order(states, scheduler.order, rng, round_number)
        _write_round_started(match_log_path, round_number, scheduled_states, scheduler)
        commit_order: list[str] = []
        futures: dict[Future[PreparedActivityStep | None], tuple[MatchParticipantRuntime, Any, float]] = {}
        with ThreadPoolExecutor(max_workers=_max_workers(scheduler, len(scheduled_states))) as executor:
            for participant, state in scheduled_states:
                _write_agent_step_started(match_log_path, round_number, participant, phase="queued")
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


def _run_continuous_match(
        gym: BbsGym,
        states: list[tuple[MatchParticipantRuntime, Any]],
        scheduler: MatchSchedulerConfig,
        match_log_path: Path,
        match_started_at: float,
) -> int:
    rng = random.Random(scheduler.seed)
    initial_order = match_round_order(states, scheduler.order, rng, 1)
    committed_ticks = 0
    queued_ticks = 0
    futures: dict[Future[PreparedActivityStep | None], tuple[MatchParticipantRuntime, Any, float, int]] = {}

    def queue_step(
            executor: ThreadPoolExecutor,
            participant: MatchParticipantRuntime,
            state: Any,
    ) -> None:
        nonlocal queued_ticks
        queued_ticks += 1
        _write_agent_step_started(match_log_path, None, participant, phase="queued", tick=queued_ticks)
        futures[executor.submit(participant.runner.prepare_step, state)] = (
            participant,
            state,
            time.monotonic(),
            queued_ticks,
        )

    with ThreadPoolExecutor(max_workers=_max_workers(scheduler, len(initial_order))) as executor:
        for participant, state in initial_order:
            if queued_ticks < scheduler.max_rounds and not _match_time_exhausted(scheduler, match_started_at):
                queue_step(executor, participant, state)

        while futures and not _match_time_exhausted(scheduler, match_started_at):
            done, _ = wait(
                futures,
                timeout=_match_wall_seconds_remaining(scheduler, match_started_at),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                break
            for future in done:
                participant, state, started_at, queued_tick = futures.pop(future)
                prepared = _future_preparation(future, participant, state, match_log_path, None, tick=queued_tick)
                _write_agent_decision_completed(
                    match_log_path,
                    None,
                    participant,
                    elapsed_seconds=time.monotonic() - started_at,
                    tick=queued_tick,
                )
                committed_ticks += 1
                _write_match_event(
                    match_log_path,
                    {
                        "type": "commit_order",
                        "tick": committed_ticks,
                        "order": [participant.spec.agent_id],
                        "match_order": scheduler.order,
                        "match_seed": scheduler.seed,
                        "commit_policy": "continuous_completion",
                        "queued_tick": queued_tick,
                        "timestamp": time.time(),
                    },
                )
                _commit_parallel_step(
                    gym,
                    participant,
                    state,
                    prepared,
                    scheduler,
                    match_log_path,
                    None,
                    tick=committed_ticks,
                )
                if (
                    not state.completed
                    and state.budget.remaining()
                    and queued_ticks < scheduler.max_rounds
                    and not _match_time_exhausted(scheduler, match_started_at)
                ):
                    queue_step(executor, participant, state)
    return committed_ticks


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
            _write_agent_step_started(match_log_path, round_number, participant, phase="queued")
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
        round_number: int | None,
        tick: int | None = None,
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
                **_event_clock(round_number, tick),
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
        round_number: int | None,
        tick: int | None = None,
) -> None:
    started_at = time.monotonic()
    step = participant.runner.commit_prepared_step(state, prepared)
    _write_match_event(
        match_log_path,
        {
            "type": "agent_step_completed",
            **_event_clock(round_number, tick),
            "agent_id": participant.spec.agent_id,
            "elapsed_seconds": time.monotonic() - started_at,
            "timestamp": time.time(),
        },
    )
    _write_agent_step_event(match_log_path, round_number, participant, state, step, tick=tick)
    if state.completed and state.stop_reason == "disconnected":
        handle_match_disconnect(gym, participant, state, scheduler, match_log_path, round_number, tick=tick)


def _max_workers(scheduler: MatchSchedulerConfig, active_count: int) -> int:
    if active_count <= 0:
        return 1
    if scheduler.max_workers is None:
        return active_count
    return max(1, min(scheduler.max_workers, active_count))


def _match_wall_seconds_remaining(scheduler: MatchSchedulerConfig, match_started_at: float) -> float:
    return max(0.0, scheduler.max_wall_seconds - (time.monotonic() - match_started_at))


def _match_time_exhausted(scheduler: MatchSchedulerConfig, match_started_at: float) -> bool:
    return _match_wall_seconds_remaining(scheduler, match_started_at) <= 0


def _mark_active_states_completed(states: list[tuple[MatchParticipantRuntime, Any]], stop_reason: str) -> None:
    for _, state in states:
        if not state.completed:
            state.stop_reason = stop_reason
            state.completed = True


def _match_limit_stop_reason(scheduler: MatchSchedulerConfig) -> str:
    return "match_ticks" if scheduler.mode == "continuous" else "match_rounds"


def _commit_count(results: list[tuple[MatchParticipantRuntime, ActivityResult]]) -> int:
    return sum(len(result.steps) for _, result in results)


def _event_clock(round_number: int | None, tick: int | None = None) -> dict[str, int]:
    event: dict[str, int] = {}
    if round_number is not None:
        event["round"] = round_number
    if tick is not None:
        event["tick"] = tick
    return event


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


def _write_match_started(
        match_log_path: Path,
        participants: list[MatchParticipantRuntime],
        scheduler: MatchSchedulerConfig,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "match_started",
            "scheduler": _scheduler_event_dict(scheduler),
            "participants": [
                {
                    "agent_id": participant.spec.agent_id,
                    "provider": participant.spec.provider,
                    "model": participant.spec.model,
                    "model_metadata": participant.model_metadata,
                    "agent_log_path": str(participant.log_path),
                }
                for participant in participants
            ],
            "timestamp": time.time(),
        },
    )


def _write_match_completed(
        match_log_path: Path,
        scheduler_count: int,
        commit_count: int,
        results: list[tuple[MatchParticipantRuntime, ActivityResult]],
        scheduler: MatchSchedulerConfig,
        match_started_at: float,
        *,
        clean_exit: bool = True,
        error: str = "",
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "match_completed",
            "clean_exit": clean_exit,
            "commit_count": commit_count,
            "elapsed_seconds": time.monotonic() - match_started_at,
            "error": error,
            "scheduler": _scheduler_event_dict(scheduler),
            "scheduler_count": scheduler_count,
            "scheduler_count_unit": "ticks" if scheduler.mode == "continuous" else "rounds",
            "results": [
                {
                    "agent_id": result.agent_id,
                    "steps": len(result.steps),
                    "stop_reason": result.stop_reason,
                    "activity": result.activity,
                    "agent_log_path": str(participant.log_path),
                }
                for participant, result in results
            ],
            "timestamp": time.time(),
        },
    )


def _scheduler_event_dict(scheduler: MatchSchedulerConfig) -> dict[str, Any]:
    return {
        "mode": scheduler.mode,
        "order": scheduler.order,
        "seed": scheduler.seed,
        "disconnect_policy": scheduler.disconnect_policy,
        "max_reconnects": scheduler.max_reconnects,
        "reconnect_delay": scheduler.reconnect_delay,
        "max_rounds": scheduler.max_rounds,
        "max_decision_ticks": scheduler.max_decision_ticks,
        "max_wall_seconds": scheduler.max_wall_seconds,
        "max_workers": scheduler.max_workers,
    }


def _write_agent_step_started(
        match_log_path: Path,
        round_number: int | None,
        participant: MatchParticipantRuntime,
        *,
        phase: Literal["queued", "started"],
        tick: int | None = None,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "agent_step_started",
            "phase": phase,
            **_event_clock(round_number, tick),
            "agent_id": participant.spec.agent_id,
            "timestamp": time.time(),
        },
    )


def _write_agent_decision_completed(
        match_log_path: Path,
        round_number: int | None,
        participant: MatchParticipantRuntime,
        elapsed_seconds: float,
        tick: int | None = None,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "agent_decision_completed",
            **_event_clock(round_number, tick),
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
        round_number: int | None,
        participant: MatchParticipantRuntime,
        state: Any,
        step: Any,
        tick: int | None = None,
) -> None:
    _write_match_event(
        match_log_path,
        {
            "type": "agent_step",
            **_event_clock(round_number, tick),
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
    if isinstance(agent, ClosableAgent):
        agent.close()
