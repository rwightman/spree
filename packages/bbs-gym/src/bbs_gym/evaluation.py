"""Evaluator-owned metrics for Synchronet door games."""

from __future__ import annotations

import re
from typing import Any

from tty_agent.actions import Action
from tty_agent.evaluation import EvaluationProbe, EvaluationProfile
from tty_agent.terminal import Observation


_NUMBER = r"[-+]?\d[\d,]*"
_TW2_INFO_RE = re.compile(r"<Info>", re.IGNORECASE)
_TW2_SCOREBOARD_RE = re.compile(r"Player Rankings\s*\r?\n", re.IGNORECASE)
_TW2_SCOREBOARD_END_RE = re.compile(
    r"(?:\r?\n\s*Team Rankings\b|\r?\n\s*Computer command\b|\r?\n\s*Command\s+\(\?=Help\)\?)",
    re.IGNORECASE,
)
_TW2_MAIN_PROMPT_RE = re.compile(r"(?:^|\n)Command\s+\(\?=Help\)\?\s*$", re.IGNORECASE)
_TW2_COMPUTER_PROMPT_RE = re.compile(r"(?:^|\n)Computer command\s+\(\?=help\)\?\s*$", re.IGNORECASE)
_TW2_PILOT_RE = re.compile(r"Pilot's Name:\s*(?P<pilot>[^\r\n]+)", re.IGNORECASE)
_TW2_INFO_FIELDS = {
    "fighters": re.compile(rf"^\s*Fighters:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
    "sector": re.compile(rf"^\s*Sector Location:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
    "turns_left": re.compile(rf"^\s*Turns left:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
    "cargo_holds": re.compile(rf"^\s*Cargo Holds:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
    "cargo_ore": re.compile(rf"^\s*# with Ore:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
    "cargo_organics": re.compile(
        rf"^\s*# with Org:\s*(?P<value>{_NUMBER})\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "cargo_equipment": re.compile(
        rf"^\s*# with Equ:\s*(?P<value>{_NUMBER})\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "credits": re.compile(rf"^\s*Credits:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
    "door_points": re.compile(rf"^\s*Door points:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
}
_TW2_CREDITS_RE = re.compile(rf"\bYou (?:now )?have\s+(?P<value>{_NUMBER})\s+credits\b", re.IGNORECASE)
_TW2_TURNS_RE = re.compile(rf"\bYou have\s+(?P<value>{_NUMBER})\s+turns left\b", re.IGNORECASE)

_SRE_STATUS_EMPIRE_RE = re.compile(r"^\s*-\*(?P<empire>.+?)\*-\s*$", re.MULTILINE)
_SRE_STATUS_FIELDS = {
    "score": re.compile(rf"^\s*Score:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
    "turns_left": re.compile(rf"^\s*Turns Left:\s*(?P<value>{_NUMBER})\s*$", re.IGNORECASE | re.MULTILINE),
    "money": re.compile(
        rf"^\s*Money:\s*(?P<value>{_NUMBER})(?:\s+\(Bank Savings={_NUMBER}\))?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "bank_savings": re.compile(
        rf"^\s*Money:\s*{_NUMBER}\s+\(Bank Savings=(?P<value>{_NUMBER})\)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "net_worth": re.compile(rf"\(Net Worth=(?P<value>{_NUMBER})\)", re.IGNORECASE),
    "planets": re.compile(rf"\(Total=(?P<value>{_NUMBER})\)", re.IGNORECASE),
    "protection_turns_left": re.compile(
        rf"\bYou have\s+(?P<value>{_NUMBER})\s+turns of protection left\b",
        re.IGNORECASE,
    ),
}
_SRE_SCOREBOARD_RE = re.compile(r"List of Players/Scores:\s*\r?\n", re.IGNORECASE)
_SRE_SCOREBOARD_END_RE = re.compile(r"(?:\r?\n[^\r\n]*\[Galaxy Menu\]|\Z)", re.IGNORECASE)
_SRE_SCORE_ROW_RE = re.compile(
    rf"^\s*<(?P<player_id>[A-Y])>\s+(?P<empire>.+?)\s+"
    rf"(?P<planets>{_NUMBER})\s+(?P<score>{_NUMBER})\s+"
    rf"(?P<net_worth>{_NUMBER})(?:\s+(?P<points>{_NUMBER}))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SRE_GALAXY_PROMPT_RE = re.compile(
    r"(?:^|\n)Which one\?\s*(?:\[Enter=[^\]]+\])?\s*$",
    re.IGNORECASE,
)

_FIGHTER_VALUE = 100
_HOLD_VALUE = 500
_COMMODITY_VALUES = {
    "cargo_ore": 10,
    "cargo_organics": 20,
    "cargo_equipment": 35,
}


def extract_tw2_metrics(observation: Observation) -> dict[str, Any] | None:
    """Extract player state and the built-in TW2 leaderboard value.

    TW2's leaderboard ``Value`` is credits plus the configured base value of
    cargo holds, cargo, ship fighters, and deployed fighters. ``Door points``
    are a separate achievement counter and are retained as their own metric.
    """

    text = observation.new_text
    metrics: dict[str, Any] = {}
    pilot = _extract_tw2_info(text, metrics)
    if pilot is None:
        pilot = _metadata_alias(observation)

    _extract_partial_state(text, metrics)
    _add_onboard_value(metrics)
    _extract_leaderboard_score(text, pilot, metrics)
    _add_deployed_fighters(metrics)
    return metrics or None


def tw2_score_probe_ready(observation: Observation) -> bool:
    """Allow the turn-free score probe only at a safe TW2 menu prompt."""

    text = observation.model_text.rstrip()
    return bool(_TW2_MAIN_PROMPT_RE.search(text) or _TW2_COMPUTER_PROMPT_RE.search(text))


TW2_EVALUATION_PROFILE = EvaluationProfile(
    name="tw2-score",
    extractor=extract_tw2_metrics,
    final_probe=EvaluationProbe(
        name="info-and-ranking",
        # X exits the computer menu when necessary and is harmlessly rejected
        # at the main menu. I prints player state; C/R prints rankings; the
        # final X restores the main menu. None of these commands spends a turn.
        action=Action(action="type_text", text="XICRX"),
        ready=tw2_score_probe_ready,
        turn_cost="none",
        observe_timeout=10.0,
        stable_ms=300,
        byte_quiet_ms=50,
        prompt_fast_path=False,
    ),
)


def extract_sre_metrics(observation: Observation) -> dict[str, Any] | None:
    """Extract SRE's built-in score and related player state."""

    text = observation.new_text
    if observation.model_text and observation.model_text not in text:
        # SRE paints several screens with cursor-addressed fragments. The raw
        # delta preserves history such as scoreboards, while the rendered
        # screen reconstructs status fields that may not be contiguous there.
        text = f"{text}\n{observation.model_text}"
    metrics: dict[str, Any] = {}
    status_empire = _extract_sre_status(text, metrics)
    leaderboard = extract_sre_scoreboard(text)
    if leaderboard:
        metrics["leaderboard"] = leaderboard
        selected = _select_sre_score_row(leaderboard, status_empire, observation)
        if selected is not None:
            metrics.update(selected)
            metrics["rank"] = 1 + sum(row["score"] > selected["score"] for row in leaderboard)
    return metrics or None


def sre_score_probe_ready(observation: Observation) -> bool:
    """Allow the zero-turn status probe only at SRE's Galaxy menu."""

    text = observation.model_text.rstrip().rstrip("█").rstrip()
    menu_start = text.rfind("[Galaxy Menu]")
    if menu_start < 0:
        return False
    active_menu = text[menu_start:]
    return (
        "[5] See Your Status" in active_menu
        and "[8] Scores" in active_menu
        and bool(_SRE_GALAXY_PROMPT_RE.search(active_menu))
    )


SRE_EVALUATION_PROFILE = EvaluationProfile(
    name="sre-score",
    extractor=extract_sre_metrics,
    final_probe=EvaluationProbe(
        name="empire-status",
        # SRE's own status screen includes Score and does not spend a game
        # turn. Do not append Enter: it can select Play SRE on the next menu.
        action=Action(action="press_key", key="5"),
        ready=sre_score_probe_ready,
        turn_cost="none",
        observe_timeout=10.0,
        stable_ms=300,
        byte_quiet_ms=50,
        prompt_fast_path=False,
    ),
)


def _extract_tw2_info(text: str, metrics: dict[str, Any]) -> str | None:
    starts = list(_TW2_INFO_RE.finditer(text))
    if not starts:
        return None
    info_text = text[starts[-1].start() :]
    pilot_match = _TW2_PILOT_RE.search(info_text)
    pilot: str | None = None
    if pilot_match is not None:
        pilot = re.sub(r"\s+Team\s+\[\d+\]\s*$", "", pilot_match.group("pilot"), flags=re.IGNORECASE).strip()
        if pilot:
            metrics["pilot"] = pilot

    for name, pattern in _TW2_INFO_FIELDS.items():
        match = pattern.search(info_text)
        if match is not None:
            metrics[name] = _parse_int(match.group("value"))
    return pilot


def _extract_partial_state(text: str, metrics: dict[str, Any]) -> None:
    if "credits" not in metrics:
        credit_matches = list(_TW2_CREDITS_RE.finditer(text))
        if credit_matches:
            metrics["credits"] = _parse_int(credit_matches[-1].group("value"))
    if "turns_left" not in metrics:
        turn_matches = list(_TW2_TURNS_RE.finditer(text))
        if turn_matches:
            metrics["turns_left"] = _parse_int(turn_matches[-1].group("value"))


def _add_onboard_value(metrics: dict[str, Any]) -> None:
    required = {"fighters", "cargo_holds", "credits", *_COMMODITY_VALUES}
    if not required.issubset(metrics):
        return
    value = metrics["credits"] + metrics["fighters"] * _FIGHTER_VALUE + metrics["cargo_holds"] * _HOLD_VALUE
    value += sum(metrics[name] * unit_value for name, unit_value in _COMMODITY_VALUES.items())
    metrics["onboard_value"] = value


def _extract_leaderboard_score(text: str, pilot: str | None, metrics: dict[str, Any]) -> None:
    if not pilot:
        return
    scoreboards = list(_TW2_SCOREBOARD_RE.finditer(text))
    if not scoreboards:
        return
    scoreboard = text[scoreboards[-1].end() :]
    end_match = _TW2_SCOREBOARD_END_RE.search(scoreboard)
    if end_match is not None:
        scoreboard = scoreboard[: end_match.start()]

    row_pattern = re.compile(
        rf"^\s*(?P<rank>\d+)\s+(?P<score>{_NUMBER})\s+(?:(?P<team>\d+)\s+)?{re.escape(pilot)}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    row = row_pattern.search(scoreboard)
    if row is None:
        return
    metrics["score"] = _parse_int(row.group("score"))
    metrics["rank"] = int(row.group("rank"))
    if row.group("team") is not None:
        metrics["team"] = int(row.group("team"))


def _add_deployed_fighters(metrics: dict[str, Any]) -> None:
    score = metrics.get("score")
    onboard_value = metrics.get("onboard_value")
    if not isinstance(score, int) or not isinstance(onboard_value, int):
        return
    deployed_value = score - onboard_value
    if deployed_value >= 0 and deployed_value % _FIGHTER_VALUE == 0:
        metrics["deployed_fighters"] = deployed_value // _FIGHTER_VALUE


def _metadata_alias(observation: Observation) -> str | None:
    alias = observation.metadata.get("bbs_alias")
    return alias.strip() if isinstance(alias, str) and alias.strip() else None


def _parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def _extract_sre_status(text: str, metrics: dict[str, Any]) -> str | None:
    status_matches = list(_SRE_STATUS_EMPIRE_RE.finditer(text))
    if not status_matches:
        return None
    status_start = status_matches[-1]
    status_text = text[status_start.start() :]
    empire = status_start.group("empire").strip()
    if empire:
        metrics["empire"] = empire
    for name, pattern in _SRE_STATUS_FIELDS.items():
        match = pattern.search(status_text)
        if match is not None:
            metrics[name] = _parse_int(match.group("value"))
    return empire or None


def extract_sre_scoreboard(text: str) -> list[dict[str, Any]]:
    """Parse SRE's ANSI-stripped terminal or generated text scoreboard."""

    scoreboards = list(_SRE_SCOREBOARD_RE.finditer(text))
    if not scoreboards:
        return []
    scoreboard = text[scoreboards[-1].end() :]
    end_match = _SRE_SCOREBOARD_END_RE.search(scoreboard)
    if end_match is not None:
        scoreboard = scoreboard[: end_match.start()]
    rows: list[dict[str, Any]] = []
    for match in _SRE_SCORE_ROW_RE.finditer(scoreboard):
        row = {
            "player_id": match.group("player_id").upper(),
            "empire": match.group("empire").strip(),
            "planets": _parse_int(match.group("planets")),
            "score": _parse_int(match.group("score")),
            "net_worth": _parse_int(match.group("net_worth")),
        }
        if match.group("points") is not None:
            row["points"] = _parse_int(match.group("points"))
        rows.append(row)
    return rows


def _select_sre_score_row(
        leaderboard: list[dict[str, Any]],
        status_empire: str | None,
        observation: Observation,
) -> dict[str, Any] | None:
    candidate_names = [status_empire, _metadata_alias(observation)]
    for candidate in candidate_names:
        if not candidate:
            continue
        matching = [row for row in leaderboard if row["empire"].casefold() == candidate.casefold()]
        if len(matching) == 1:
            return matching[0]
    return leaderboard[0] if len(leaderboard) == 1 else None
