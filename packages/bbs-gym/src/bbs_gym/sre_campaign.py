"""Standalone DOSEMU adapter for accelerated SRE campaigns."""

from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tty_agent.agent import TerminalSessionAgent
from tty_agent.terminal import Observation, TerminalScreen, TurnObserver
from tty_agent.transports.base import SessionDisconnected
from tty_agent.transports.pty import PtySession

from .campaign import CampaignParticipantSpec
from .evaluation import extract_sre_metrics, extract_sre_scoreboard
from .profiles import BBS_PROFILE
from .sre_data import (
    SRE_EXECUTABLE_SIZE,
    SRE_GALAXY_MAINTENANCE_TIME_OFFSETS,
    SRE_GALAXY_RECORD_SIZE,
    SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET,
    SRE_SYSTEM_RECORD_SIZE,
    SRE_SYSTEM_TIME_OFFSETS,
    decode_sre_record,
    patch_sre_reset_time,
)


_DOS_DATE_RE = re.compile(r"Current date is \w+ (?P<month>\d{2})-(?P<day>\d{2})-(?P<year>\d{4})")
_DOS_TIME_RE = re.compile(r"Current time is (?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})")
_SRE_RETURN_TO_BBS_BYTES = b"\x1b[15~"
_SRE_GRACEFUL_EXIT_TIMEOUT = 30.0
_SRE_SCORE_SNAPSHOT = ".spree-scoreboard.json"
_SRE_GLOBAL_SCORE_IDENTITY = "SpreeScores"
_RESOURCE_OVERLAY = """; Spree standalone SRE campaign settings.
System.LocalOnly                        yes
System.Multitasker                      no
System.Fossil                           no
Display.Video                           bios
CheckTimeLimit                          no
TimeOut                                 3600
Scores.File.ANSI                        "SRESCORE.ANS"
Scores.File.Text                        "SRESCORE.TXT"
"""


def _sre_no_empire_menu(text: str) -> bool:
    folded = text.casefold()
    return "do not have an empire" in folded or ("[4] Join this Game" in text and "[5] See Your Status" not in text)


def _sre_status_screen_ready(text: str) -> bool:
    """Return whether SRE has painted enough of the status screen to score it."""

    return _sre_no_empire_menu(text) or all(
        marker in text
        for marker in (
            "Empire Status",
            "Score:",
            "Turns Left:",
            "Money:",
            "PAUSED",
        )
    )


@dataclass(frozen=True)
class SreCampaignAdapterConfig:
    """Filesystem and container settings for one isolated SRE world."""

    campaign_dir: Path
    source_world: Path = Path("doors/sre")
    dosemu_config: Path = Path("docker/synchronet/dosemu.conf")
    docker_image: str = "spree/synchronet-dosemu:local"
    docker_executable: str = "docker"
    control_timeout: float = 300.0
    verify_clock: bool = True
    native_reset: bool = True


class ManagedDockerPtySession:
    """PTY session that gracefully exits a door before reaping its container."""

    def __init__(
            self,
            session: PtySession,
            container_name: str,
            docker_executable: str,
            on_close: Callable[[str], None] | None = None,
            graceful_exit_bytes: bytes | None = None,
            graceful_exit_timeout: float = _SRE_GRACEFUL_EXIT_TIMEOUT,
    ) -> None:
        self._session = session
        self.container_name = container_name
        self.docker_executable = docker_executable
        self._on_close = on_close
        self._graceful_exit_bytes = graceful_exit_bytes
        self._graceful_exit_timeout = graceful_exit_timeout
        self._closed = False

    @property
    def transcript_path(self) -> Path | None:
        return self._session.transcript_path

    @property
    def encoding(self) -> str:
        return self._session.encoding

    def connect(self) -> None:
        self._session.connect()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._request_graceful_exit()
        finally:
            try:
                self._session.close()
            finally:
                try:
                    subprocess.run(
                        [self.docker_executable, "rm", "--force", self.container_name],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=15,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if self._on_close is not None:
                    self._on_close(self.container_name)

    def _request_graceful_exit(self) -> None:
        if self._graceful_exit_bytes is None:
            return
        try:
            self._session.send_bytes(self._graceful_exit_bytes)
        except SessionDisconnected:
            return

        deadline = time.monotonic() + self._graceful_exit_timeout
        while time.monotonic() < deadline:
            try:
                self._session.read(min(0.5, max(0.0, deadline - time.monotonic())))
            except SessionDisconnected:
                return
        raise TimeoutError(
            f"SRE did not return to the BBS within {self._graceful_exit_timeout:g} seconds; "
            "refusing to treat the session as safely persisted"
        )

    def send_text(self, text: str) -> None:
        self._session.send_text(text)

    def send_line(self, text: str = "") -> None:
        self._session.send_line(text)

    def send_key(self, key: str) -> None:
        self._session.send_key(key)

    def send_bytes(self, payload: bytes) -> None:
        self._session.send_bytes(payload)

    def drain_sent_bytes(self) -> tuple[bytes, ...]:
        return self._session.drain_sent_bytes()

    def transcript_position(self) -> int:
        return self._session.transcript_position()

    def read(self, seconds: float = 1.0) -> bytes:
        return self._session.read(seconds)


class SreCampaignAdapter:
    """Run SRE directly in a disposable, clock-controlled DOSEMU container."""

    name = "sre-dosemu"

    def __init__(self, config: SreCampaignAdapterConfig) -> None:
        self.config = config
        self.campaign_dir = config.campaign_dir.resolve()
        self.world_path = self.campaign_dir / "world"
        self.dos_path = self.campaign_dir / "dos"
        self.home_path = self.campaign_dir / "home"
        self.logs_path = self.campaign_dir / "control-logs"
        self.dosemu_config_path = self.campaign_dir / "dosemu.conf"
        self._active_containers: set[str] = set()

    def initialize_world(self, virtual_time: datetime) -> dict[str, Any]:
        """Create, Y2K-patch, and clock-check a fresh SRE world."""

        if self.world_path.exists():
            raise ValueError(f"SRE campaign world already exists: {self.world_path}")
        source_world = self.config.source_world.resolve()
        dosemu_config = self.config.dosemu_config.resolve()
        self._validate_source_world(source_world)
        if not dosemu_config.is_file():
            raise ValueError(f"DOSEMU configuration not found: {dosemu_config}")

        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.dos_path.mkdir(parents=True, exist_ok=True)
        self.home_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_world, self.world_path)
        shutil.copy2(dosemu_config, self.dosemu_config_path)
        self._remove_runtime_files()
        self._write_text_file(self.world_path / "RESOURCE.1", _RESOURCE_OVERLAY)

        reset: dict[str, Any]
        if self.config.native_reset:
            reset = self._reset_world()
        else:
            reset = {"status": "skipped", "reason": "source_declared_fresh"}
        self._remove_runtime_files()

        backup_dir = self.campaign_dir / "bootstrap-backup"
        patch = patch_sre_reset_time(
            self.world_path / "DATA" / "SYSTEM.II",
            self.world_path / "SRE.EXE",
            galaxy_path=self.world_path / "DATA" / "GALAXY.II",
            timestamp=int(self._utc(virtual_time).timestamp()),
            system_backup_path=backup_dir / "SYSTEM.II",
            galaxy_backup_path=backup_dir / "GALAXY.II",
            executable_backup_path=backup_dir / "SRE.EXE",
        )
        clock = self.verify_guest_clock(virtual_time) if self.config.verify_clock else {"status": "skipped"}
        return {
            "source_world": str(source_world),
            "dosemu_config": str(dosemu_config),
            "docker_image": self.config.docker_image,
            "patched_timestamp": patch.timestamp,
            "executable_changed": patch.executable_changed,
            "reset": reset,
            "clock": clock,
        }

    def validate_world(self, virtual_time: datetime) -> dict[str, Any]:
        """Validate protected records and report SRE's persisted clocks."""

        del virtual_time
        state = self._maintenance_state()
        if (self.world_path / "SRE.EXE").stat().st_size != SRE_EXECUTABLE_SIZE:
            raise ValueError("SRE campaign executable has an unexpected size")
        return {
            "system_times": [self._epoch_iso(value) for value in state["system_times"]],
            "maintenance_times": [self._epoch_iso(value) for value in state["maintenance_times"]],
        }

    def open_player_session(
            self,
            participant: CampaignParticipantSpec,
            virtual_time: datetime,
            transcript_path: Path,
            model_metadata: dict[str, object],
    ) -> TerminalSessionAgent:
        """Launch one player identity against the campaign's shared world."""

        self._require_idle()
        self._write_dropfile(participant.player_name)
        batch_name = "PLAY.BAT"
        self._write_batch(batch_name, virtual_time, ["SRE -n1"])
        session = self._open_dosemu(
            batch_name,
            transcript_path,
            "play",
            participant.agent_id,
            graceful_exit=True,
        )
        terminal = TerminalScreen(columns=80, lines=25, encoding="cp437")
        metadata: dict[str, Any] = {
            "transport": "pty",
            "runtime": "docker-dosemu",
            "game": "sre",
            "campaign_dir": str(self.campaign_dir),
            "virtual_time": self._utc(virtual_time).isoformat(),
            "bbs_alias": participant.player_name,
            "authenticated": True,
            "model": dict(model_metadata),
        }
        observer = TurnObserver(
            participant.agent_id,
            session,
            terminal=terminal,
            profile=BBS_PROFILE,
            metadata=metadata,
        )
        return TerminalSessionAgent(participant.agent_id, session, observer, metadata=metadata)

    def close_player_session(self, agent: object) -> None:
        close = getattr(agent, "close", None)
        if callable(close):
            close()

    def cleanup_after_session(self) -> None:
        """Clear only SRE's known ephemeral files after its process is gone."""

        if self._active_containers:
            active = ", ".join(sorted(self._active_containers))
            raise RuntimeError(f"refusing SRE cleanup while containers are active: {active}")
        self._unlink_casefold("inuse.sr")
        self._unlink_casefold("doorfile.sr")

    def score_player(
            self,
            participant: CampaignParticipantSpec,
            virtual_time: datetime,
            transcript_path: Path,
    ) -> dict[str, Any]:
        """Run a hidden, zero-turn status session after the player process closes."""

        self._require_idle()
        self._write_dropfile(participant.player_name)
        batch_name = "SCORE.BAT"
        self._write_batch(batch_name, virtual_time, ["SRE -n1"])
        session = self._open_dosemu(
            batch_name,
            transcript_path,
            "score",
            participant.agent_id,
            graceful_exit=True,
        )
        terminal = TerminalScreen(columns=80, lines=25, encoding="cp437")
        observer = TurnObserver(
            f"{participant.agent_id}-score",
            session,
            terminal=terminal,
            profile=BBS_PROFILE,
            metadata={
                "transport": "pty",
                "runtime": "docker-dosemu",
                "game": "sre",
                "bbs_alias": participant.player_name,
                "evaluator_owned": True,
                "visible_to_model": False,
                "virtual_time": self._utc(virtual_time).isoformat(),
            },
        )
        try:
            self._wait_for_screen(observer, lambda text: "PAUSED" in text and "Solar Realms Elite" in text)
            session.send_key("enter")
            self._wait_for_screen(observer, lambda text: "Visit the Galaxy" in text and "[System]" in text)
            session.send_key("2")
            galaxy = self._wait_for_screen(
                observer,
                lambda text: "[Galaxy Menu]" in text or "do not have an empire" in text.casefold(),
            )
            if _sre_no_empire_menu(galaxy.model_text):
                return self._score_probe_result({"status": "no_empire"})
            session.send_key("5")
            status = self._wait_for_screen(
                observer,
                _sre_status_screen_ready,
            )
            if _sre_no_empire_menu(status.model_text):
                return self._score_probe_result({"status": "no_empire"})
            session.send_key("enter")
            self._wait_for_screen(
                observer,
                lambda text: "[Galaxy Menu]" in text and "[8] Scores" in text and "Which one?" in text,
            )
            session.send_key("8")
            scoreboard = self._wait_for_screen(
                observer,
                lambda text: "List of Players/Scores:" in text and "[Galaxy Menu]" in text,
            )
            metrics = extract_sre_metrics(
                replace(
                    scoreboard,
                    new_text=f"{status.model_text}\n{scoreboard.new_text}",
                )
            )
            if not metrics or "score" not in metrics:
                raise RuntimeError("SRE evaluator reached status but could not extract a score")
            leaderboard = metrics.get("leaderboard")
            if (
                not isinstance(leaderboard, list)
                or not leaderboard
                or not all(
                    isinstance(row, dict)
                    for row in leaderboard
                )
            ):
                raise RuntimeError("SRE evaluator reached scores but could not extract the leaderboard")
            self._write_score_snapshot([dict(row) for row in leaderboard])
            return self._score_probe_result(metrics)
        finally:
            try:
                session.close()
            finally:
                self.cleanup_after_session()

    def probe_scores(
            self,
            virtual_time: datetime,
            transcript_path: Path,
    ) -> list[dict[str, Any]]:
        """Probe the post-maintenance leaderboard without mutating live game state."""

        self._require_idle()
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".score-probe-", dir=self.campaign_dir) as temporary:
            probe = SreCampaignAdapter(
                replace(
                    self.config,
                    campaign_dir=Path(temporary),
                    verify_clock=False,
                    native_reset=False,
                )
            )
            probe.dos_path.mkdir(parents=True)
            probe.home_path.mkdir(parents=True)
            probe.logs_path.mkdir(parents=True)
            shutil.copytree(self.world_path, probe.world_path, symlinks=True)
            shutil.copy2(self.dosemu_config_path, probe.dosemu_config_path)
            scores = probe._run_global_score_probe(virtual_time, transcript_path)

        self._write_score_snapshot(scores)
        return [dict(row) for row in scores]

    def run_maintenance(self, virtual_time: datetime) -> dict[str, Any]:
        """Advance SRE with its native sysop maintenance command."""

        self._require_idle()
        before = self._maintenance_state()
        self._write_dropfile("SpreeMaintenance")
        result_path = self.dos_path / "MAINT.OK"
        result_path.unlink(missing_ok=True)
        batch_name = "MAINT.BAT"
        self._write_batch(
            batch_name,
            virtual_time,
            [
                "SRE MAINTENANCE -n1",
                "IF ERRORLEVEL 1 GOTO FAILED",
                "ECHO OK>D:\\MAINT.OK",
                "GOTO FINISHED",
                ":FAILED",
                "ECHO FAILED>D:\\MAINT.OK",
                ":FINISHED",
            ],
        )
        transcript = self.logs_path / f"maintenance-{self._file_timestamp(virtual_time)}.raw"
        self._run_control(batch_name, transcript, "maintenance", timeout=self.config.control_timeout)
        self.cleanup_after_session()
        if not result_path.is_file() or result_path.read_text(encoding="ascii", errors="replace").strip() != "OK":
            raise RuntimeError("SRE maintenance did not write its success marker")

        after = self._maintenance_state()
        before_daily = before["maintenance_times"][0]
        after_daily = after["maintenance_times"][0]
        expected = int(self._utc(virtual_time).timestamp())
        if after_daily <= before_daily:
            raise RuntimeError(
                f"SRE daily maintenance cursor did not advance: {before_daily} -> {after_daily}"
            )
        expected_day = expected - expected % 86_400
        if after_daily < expected_day:
            raise RuntimeError(
                f"SRE daily maintenance stopped before injected day: cursor={after_daily} expected={expected_day}"
            )
        return {
            "status": "ok",
            "before": self._state_json(before),
            "after": self._state_json(after),
            "transcript": str(transcript),
        }

    def read_scores(self) -> list[dict[str, Any]]:
        snapshot_path = self.world_path / _SRE_SCORE_SNAPSHOT
        if snapshot_path.is_file():
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid SRE score snapshot: {snapshot_path}") from exc
            if not isinstance(snapshot, list) or not all(isinstance(row, dict) for row in snapshot):
                raise ValueError(f"invalid SRE score snapshot: {snapshot_path}")
            return [dict(row) for row in snapshot]

        score_path = self.world_path / "SRESCORE.TXT"
        if not score_path.is_file():
            return []
        text = score_path.read_bytes().decode("cp437", errors="replace")
        return extract_sre_scoreboard(text)

    def _write_score_snapshot(self, rows: list[dict[str, Any]]) -> None:
        snapshot_path = self.world_path / _SRE_SCORE_SNAPSHOT
        temporary = snapshot_path.with_name(f".{snapshot_path.name}.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_text(
                json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            temporary.replace(snapshot_path)
        finally:
            temporary.unlink(missing_ok=True)

    def verify_guest_clock(self, virtual_time: datetime) -> dict[str, Any]:
        """Prove DOS DATE/TIME injection in the configured container runtime."""

        clock_path = self.dos_path / "CLOCK.TXT"
        clock_path.unlink(missing_ok=True)
        batch_name = "CLOCK.BAT"
        self._write_batch(
            batch_name,
            virtual_time,
            [
                "DATE < NUL > D:\\CLOCK.TXT",
                "TIME < NUL >> D:\\CLOCK.TXT",
            ],
        )
        transcript = self.logs_path / "clock-check.raw"
        self._run_control(batch_name, transcript, "clock", timeout=min(self.config.control_timeout, 90.0))
        if not clock_path.is_file():
            raise RuntimeError("DOSEMU clock check did not produce CLOCK.TXT")
        output = clock_path.read_text(encoding="ascii", errors="replace")
        date_match = _DOS_DATE_RE.search(output)
        time_match = _DOS_TIME_RE.search(output)
        if date_match is None or time_match is None:
            raise RuntimeError(f"could not parse DOSEMU clock output: {output!r}")
        actual = datetime(
            int(date_match.group("year")),
            int(date_match.group("month")),
            int(date_match.group("day")),
            int(time_match.group("hour")),
            int(time_match.group("minute")),
            int(time_match.group("second")),
            tzinfo=timezone.utc,
        )
        expected = self._utc(virtual_time).replace(microsecond=0)
        if abs((actual - expected).total_seconds()) > 2:
            raise RuntimeError(f"DOSEMU clock mismatch: expected {expected.isoformat()}, got {actual.isoformat()}")
        return {
            "status": "ok",
            "expected": expected.isoformat(),
            "actual": actual.isoformat(),
            "transcript": str(transcript),
        }

    def _reset_world(self) -> dict[str, Any]:
        """Drive SRE's native reset UI under its supported pre-Y2K clock."""

        reset_time = datetime(1999, 12, 31, 12, tzinfo=timezone.utc)
        self._write_dropfile("SpreeSysop")
        batch_name = "RESET.BAT"
        self._write_batch(batch_name, reset_time, ["SRE RESET -n1"])
        transcript = self.logs_path / "reset.raw"
        session = self._open_dosemu(batch_name, transcript, "reset", "sysop")
        observer = TurnObserver(
            "sre-reset",
            session,
            terminal=TerminalScreen(columns=80, lines=25, encoding="cp437"),
            profile=BBS_PROFILE,
            metadata={
                "transport": "pty",
                "runtime": "docker-dosemu",
                "game": "sre",
                "evaluator_owned": True,
                "visible_to_model": False,
                "virtual_time": reset_time.isoformat(),
            },
        )
        try:
            self._wait_for_text(observer, "Reset the Game")
            session.send_key("2")
            self._wait_for_text(observer, "Enter the name of your galaxy")
            session.send_line("Spree Galaxy")
            self._wait_for_text(observer, "Name your galaxy")
            # Hotkey prompts consume one byte immediately. Do not append Enter:
            # it would remain buffered and accept the following prompt's default.
            session.send_key("y")
            self._wait_for_text(observer, "Are you sure you want to reset the game")
            session.send_key("y")

            for prompt, response, hotkey in (
                ("Do you want Inflation", "y", True),
                ("How many days before maintenance runs", "1", False),
                ("How many turns should be allowed per maintenance interval", "5", False),
                ("How many turns of protection", "20", False),
                ("What is the galactic tax rate", "15", False),
                ("How many planets are initially available", "100", False),
                ("How many planets per turn can be bought", "", False),
                ("How many seconds", "60", False),
                ("How many thousands of megatons of food", "70", False),
                ("Would you like new pirate names", "y", True),
            ):
                self._wait_for_text(observer, prompt)
                if hotkey:
                    session.send_key(response)
                else:
                    session.send_line(response)

            self._wait_for_screen(
                observer,
                lambda text: "[Sysop]" in text and "Reset the Game" in text,
                timeout=self.config.control_timeout,
            )
            session.send_key("q")
            try:
                self._wait_for_screen(
                    observer,
                    lambda text: "[System]" in text and "Visit the Galaxy" in text,
                    timeout=min(self.config.control_timeout, 10.0),
                )
            except SessionDisconnected:
                pass
            else:
                session.send_key("q")
                self._drain_until_exit(session, "reset", self.config.control_timeout)
        finally:
            try:
                session.close()
            finally:
                self.cleanup_after_session()
        return {
            "status": "ok",
            "virtual_time": reset_time.isoformat(),
            "transcript": str(transcript),
        }

    def _open_dosemu(
            self,
            batch_name: str,
            transcript_path: Path,
            purpose: str,
            identity: str,
            *,
            graceful_exit: bool = False,
    ) -> ManagedDockerPtySession:
        container_name = self._container_name(purpose, identity)
        mount = f"{self.campaign_dir}:/sbbs-data"
        argv = [
            self.config.docker_executable,
            "run",
            "--rm",
            "--interactive",
            "--tty",
            "--pull=never",
            "--network=none",
            "--name",
            container_name,
            "--volume",
            mount,
            "--env",
            "HOME=/sbbs-data/home",
            "--env",
            "DOSDRIVE_D=/sbbs-data/dos",
            "--env",
            "TERM=xterm",
            self.config.docker_image,
            "/usr/bin/dosemu.bin",
            "-t",
            "-f/sbbs-data/dosemu.conf",
            f"-ED:{batch_name}",
            f"-o/sbbs-data/control-logs/{container_name}.dosemu.log",
        ]
        inner = PtySession(
            argv=argv,
            encoding="cp437",
            transcript_path=transcript_path,
            timeout=10.0,
            columns=80,
            lines=25,
        )
        session = ManagedDockerPtySession(
            inner,
            container_name,
            self.config.docker_executable,
            on_close=self._active_containers.discard,
            graceful_exit_bytes=_SRE_RETURN_TO_BBS_BYTES if graceful_exit else None,
        )
        self._active_containers.add(container_name)
        try:
            session.connect()
        except BaseException:
            session.close()
            raise
        return session

    def _run_control(
            self,
            batch_name: str,
            transcript_path: Path,
            purpose: str,
            *,
            timeout: float,
    ) -> None:
        session = self._open_dosemu(batch_name, transcript_path, purpose, purpose)
        try:
            self._drain_until_exit(session, purpose, timeout)
        finally:
            session.close()

    def _run_global_score_probe(
            self,
            virtual_time: datetime,
            transcript_path: Path,
    ) -> list[dict[str, Any]]:
        """Read the live Galaxy Scores menu in an evaluator-only session."""

        self._require_idle()
        self._remove_runtime_files()
        self._write_dropfile(_SRE_GLOBAL_SCORE_IDENTITY)
        batch_name = "GLOBALSCORE.BAT"
        self._write_batch(batch_name, virtual_time, ["SRE -n1"])
        session = self._open_dosemu(
            batch_name,
            transcript_path,
            "global-score",
            _SRE_GLOBAL_SCORE_IDENTITY,
            graceful_exit=True,
        )
        observer = TurnObserver(
            "sre-global-score",
            session,
            terminal=TerminalScreen(columns=80, lines=25, encoding="cp437"),
            profile=BBS_PROFILE,
            metadata={
                "transport": "pty",
                "runtime": "docker-dosemu",
                "game": "sre",
                "bbs_alias": _SRE_GLOBAL_SCORE_IDENTITY,
                "evaluator_owned": True,
                "visible_to_model": False,
                "virtual_time": self._utc(virtual_time).isoformat(),
            },
        )
        try:
            self._wait_for_screen(observer, lambda text: "PAUSED" in text and "Solar Realms Elite" in text)
            session.send_key("enter")
            self._wait_for_screen(observer, lambda text: "Visit the Galaxy" in text and "[System]" in text)
            session.send_key("2")
            self._wait_for_screen(
                observer,
                lambda text: "[Galaxy Menu]" in text and "[8] Scores" in text and "Which one?" in text,
            )
            session.send_key("8")
            scoreboard = self._wait_for_screen(
                observer,
                lambda text: "List of Players/Scores:" in text and "[Galaxy Menu]" in text,
            )
            score_text = scoreboard.model_text
            scores = extract_sre_scoreboard(score_text)
            if not scores and re.search(r"<[A-Z]>", score_text, flags=re.IGNORECASE):
                raise RuntimeError("SRE global score probe reached the leaderboard but could not parse its rows")
            return [dict(row) for row in scores]
        finally:
            try:
                session.close()
            finally:
                self.cleanup_after_session()

    @staticmethod
    def _drain_until_exit(
            session: ManagedDockerPtySession,
            purpose: str,
            timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                session.read(min(0.5, max(0.0, deadline - time.monotonic())))
            except SessionDisconnected:
                return
        raise TimeoutError(f"SRE {purpose} exceeded {timeout:g} seconds")

    def _wait_for_text(
            self,
            observer: TurnObserver,
            expected: str,
    ) -> Observation:
        expected_casefold = expected.casefold()
        return self._wait_for_screen(
            observer,
            lambda text: expected_casefold in text.casefold(),
            timeout=self.config.control_timeout,
        )

    def _wait_for_screen(
            self,
            observer: TurnObserver,
            predicate: Callable[[str], bool],
            timeout: float = 45.0,
    ) -> Observation:
        deadline = time.monotonic() + timeout
        last: Observation | None = None
        new_text_parts: list[str] = []
        first_transcript_byte: int | None = None
        bytes_read = 0
        while time.monotonic() < deadline:
            last = observer.observe_turn(
                timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                stable_ms=250,
                byte_quiet_ms=50,
                poll_interval=0.05,
                prompt_fast_path=False,
            )
            if first_transcript_byte is None:
                first_transcript_byte = last.transcript_byte_start
            new_text_parts.append(last.new_text)
            bytes_read += last.bytes_read
            if predicate(last.model_text):
                return replace(
                    last,
                    new_text="".join(new_text_parts),
                    transcript_byte_start=first_transcript_byte,
                    bytes_read=bytes_read,
                )
            time.sleep(0.05)
        tail = "" if last is None else last.model_text[-500:]
        raise TimeoutError(f"SRE evaluator did not reach the expected screen; tail={tail!r}")

    def _write_batch(self, name: str, virtual_time: datetime, commands: list[str]) -> None:
        moment = self._utc(virtual_time)
        date_value = moment.strftime("%m-%d-%Y")
        time_value = moment.strftime("%H:%M:%S")
        lines = [
            "@ECHO OFF",
            "@lredir -f E: linux\\fs/sbbs-data >NUL",
            "@SET TZ=UTC0",
            "E:",
            "CD WORLD",
            f"DATE {date_value} > NUL",
            f"TIME {time_value} > NUL",
            *commands,
            "IF EXIST INUSE.SR DEL INUSE.SR",
            "EXITEMU",
        ]
        self._write_dos_file(self.dos_path / name, lines)

    def _write_dropfile(self, player_name: str) -> None:
        if not player_name or "\r" in player_name or "\n" in player_name:
            raise ValueError("SRE player_name must be a non-empty single line")
        try:
            player_name.encode("cp437")
        except UnicodeEncodeError as exc:
            raise ValueError("SRE player_name must be representable in CP437") from exc
        self._write_dos_file(
            self.world_path / "DOORFILE.SR",
            [player_name, "1", "1", "24", "9600", "0", "-1", player_name],
        )

    def _maintenance_state(self) -> dict[str, tuple[int, ...]]:
        system_path = self.world_path / "DATA" / "SYSTEM.II"
        galaxy_path = self.world_path / "DATA" / "GALAXY.II"
        system = system_path.read_bytes()
        galaxy_ciphertext = galaxy_path.read_bytes()
        if len(system) != SRE_SYSTEM_RECORD_SIZE:
            raise ValueError(f"unexpected SRE SYSTEM.II size: {len(system)}")
        if len(galaxy_ciphertext) != SRE_GALAXY_RECORD_SIZE:
            raise ValueError(f"unexpected SRE GALAXY.II size: {len(galaxy_ciphertext)}")
        descriptor = system[SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET:SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET + 24]
        galaxy = decode_sre_record(galaxy_ciphertext, descriptor)
        return {
            "system_times": tuple(struct.unpack_from("<I", system, offset)[0] for offset in SRE_SYSTEM_TIME_OFFSETS),
            "maintenance_times": tuple(
                struct.unpack_from("<I", galaxy, offset)[0]
                for offset in SRE_GALAXY_MAINTENANCE_TIME_OFFSETS
            ),
        }

    def _state_json(self, state: dict[str, tuple[int, ...]]) -> dict[str, list[str]]:
        return {
            "system_times": [self._epoch_iso(value) for value in state["system_times"]],
            "maintenance_times": [self._epoch_iso(value) for value in state["maintenance_times"]],
        }

    def _validate_source_world(self, source: Path) -> None:
        required = [
            source / "SRE.EXE",
            source / "DATA" / "SYSTEM.II",
            source / "DATA" / "GALAXY.II",
            source / "DATA" / "EMPIRE.II",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"SRE source world is incomplete; missing: {', '.join(missing)}")

    def _remove_runtime_files(self) -> None:
        self._unlink_casefold("inuse.sr")
        self._unlink_casefold("doorfile.sr")

    def _unlink_casefold(self, name: str) -> None:
        if not self.world_path.is_dir():
            return
        for path in self.world_path.iterdir():
            if path.name.casefold() == name.casefold() and path.is_file():
                path.unlink()

    def _require_idle(self) -> None:
        if self._active_containers:
            active = ", ".join(sorted(self._active_containers))
            raise RuntimeError(f"SRE adapter already has an active process: {active}")

    def _container_name(self, purpose: str, identity: str) -> str:
        base = f"spree-sre-{self.campaign_dir.name}-{purpose}-{identity}"
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", base).strip("-.").lower()
        return f"{safe[:48]}-{uuid.uuid4().hex[:10]}"

    @staticmethod
    def _score_probe_result(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            **metrics,
            "score_probe": {
                "name": "sre-empire-status",
                "turn_cost": "none",
                "visible_to_model": False,
            },
        }

    @staticmethod
    def _write_dos_file(path: Path, lines: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp437"))

    @staticmethod
    def _write_text_file(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="ascii", newline="\n")

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("SRE virtual time must be timezone-aware")
        return value.astimezone(timezone.utc).replace(microsecond=0)

    @staticmethod
    def _epoch_iso(value: int) -> str:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    @classmethod
    def _file_timestamp(cls, value: datetime) -> str:
        return cls._utc(value).strftime("%Y%m%dT%H%M%SZ")
