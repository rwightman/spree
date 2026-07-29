import struct
from datetime import datetime, timezone
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from bbs_gym.evaluation import extract_sre_metrics, extract_sre_scoreboard
from bbs_gym.sre_campaign import (
    ManagedDockerPtySession,
    SreCampaignAdapter,
    SreCampaignAdapterConfig,
    _sre_no_empire_menu,
    _sre_status_screen_ready,
)
from bbs_gym.sre_data import (
    SRE_GALAXY_MAINTENANCE_TIME_OFFSETS,
    SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET,
    SRE_SYSTEM_TIME_OFFSETS,
    decode_sre_record,
    encode_sre_record,
)
from tty_agent.terminal import Observation
from tty_agent.transports.base import SessionDisconnected


class FakePtySession:
    transcript_path = None
    encoding = "cp437"

    def __init__(self, *, disconnect_on_read: bool) -> None:
        self.disconnect_on_read = disconnect_on_read
        self.events: list[object] = []

    def send_bytes(self, payload: bytes) -> None:
        self.events.append(("send", payload))

    def read(self, seconds: float = 1.0) -> bytes:
        self.events.append("read")
        if self.disconnect_on_read:
            raise SessionDisconnected("process exited")
        return b"still running"

    def close(self) -> None:
        self.events.append("close")


class SequenceObserver:
    def __init__(self, observations: list[Observation]) -> None:
        self.observations = list(observations)

    def observe_turn(self, **_kwargs: Any) -> Observation:
        return self.observations.pop(0)


def _observation(new_text: str, model_text: str, byte_start: int) -> Observation:
    return Observation(
        agent_id="score",
        pretty_screen=model_text,
        model_text=model_text,
        new_text=new_text,
        cursor=(0, 0),
        stable_ms=300,
        byte_quiet_ms=50,
        matched_prompt=None,
        ready_reason="stable",
        profile="synchronet-bbs",
        transcript_path=None,
        transcript_byte_start=byte_start,
        transcript_byte_end=byte_start + len(new_text),
        bytes_read=len(new_text),
        timed_out=False,
        timestamp=0.0,
        metadata={},
    )


def _fresh_reset_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    copytree(Path("doors/sre"), source)
    system_path = source / "DATA" / "SYSTEM.II"
    galaxy_path = source / "DATA" / "GALAXY.II"
    system = bytearray(system_path.read_bytes())
    descriptor_offset = SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET
    descriptor = system[descriptor_offset:descriptor_offset + 24]
    galaxy = bytearray(decode_sre_record(galaxy_path.read_bytes(), descriptor))
    reset_timestamp = int(datetime(1999, 12, 31, 12, tzinfo=timezone.utc).timestamp())
    for offset in SRE_SYSTEM_TIME_OFFSETS:
        struct.pack_into("<I", system, offset, reset_timestamp)
    for offset in SRE_GALAXY_MAINTENANCE_TIME_OFFSETS:
        struct.pack_into("<I", galaxy, offset, reset_timestamp)
    encrypted_galaxy, updated_descriptor = encode_sre_record(galaxy, nonce=descriptor[8:16])
    system[descriptor_offset:descriptor_offset + 24] = updated_descriptor
    system_path.write_bytes(system)
    galaxy_path.write_bytes(encrypted_galaxy)
    return source


def test_sre_campaign_initializes_patched_isolated_world(tmp_path):
    start = datetime(2026, 7, 28, 12, tzinfo=timezone.utc)
    source = _fresh_reset_source(tmp_path)
    adapter = SreCampaignAdapter(
        SreCampaignAdapterConfig(
            campaign_dir=tmp_path / "campaign",
            source_world=source,
            dosemu_config=Path("docker/synchronet/dosemu.conf"),
            verify_clock=False,
            native_reset=False,
        )
    )

    initialized = adapter.initialize_world(start)
    validation = adapter.validate_world(start)

    assert initialized["executable_changed"] is True
    assert initialized["reset"] == {"status": "skipped", "reason": "source_declared_fresh"}
    assert initialized["clock"] == {"status": "skipped"}
    assert validation["system_times"] == ["2026-07-28T12:00:00+00:00"] * 2
    assert validation["maintenance_times"] == [
        "2026-07-28T12:00:00+00:00",
        "2026-07-28T00:00:00+00:00",
        "2026-07-28T12:00:00+00:00",
    ]
    assert (adapter.campaign_dir / "bootstrap-backup" / "SRE.EXE").is_file()
    assert "System.LocalOnly" in (adapter.world_path / "RESOURCE.1").read_text(encoding="ascii")
    assert not (adapter.world_path / "DOORFILE.SR").exists()


def test_sre_campaign_writes_dropfile_and_clocked_batch(tmp_path):
    adapter = SreCampaignAdapter(
        SreCampaignAdapterConfig(campaign_dir=tmp_path / "campaign", verify_clock=False)
    )
    adapter.world_path.mkdir(parents=True)
    adapter.dos_path.mkdir(parents=True)

    adapter._write_dropfile("ClaudeS5")
    adapter._write_batch("PLAY.BAT", datetime(2026, 7, 29, 12, tzinfo=timezone.utc), ["SRE -n1"])

    assert (adapter.world_path / "DOORFILE.SR").read_bytes() == (
        b"ClaudeS5\r\n1\r\n1\r\n24\r\n9600\r\n0\r\n-1\r\nClaudeS5\r\n"
    )
    batch = (adapter.dos_path / "PLAY.BAT").read_bytes().decode("cp437")
    assert "DATE 07-29-2026 > NUL\r\n" in batch
    assert "TIME 12:00:00 > NUL\r\n" in batch
    assert "SRE -n1\r\n" in batch
    assert batch.endswith("IF EXIST INUSE.SR DEL INUSE.SR\r\nEXITEMU\r\n")


def test_extract_sre_scoreboard_supports_original_format_without_points():
    text = """List of Players/Scores:
ID  Empire                              Planets      Score    Net Worth
----------------------------------------------------------------------------
<A> First Empire                             27         23         689
<B> Second Empire                            31          7         412
----------------------------------------------------------------------------
"""

    assert extract_sre_scoreboard(text) == [
        {"player_id": "A", "empire": "First Empire", "planets": 27, "score": 23, "net_worth": 689},
        {"player_id": "B", "empire": "Second Empire", "planets": 31, "score": 7, "net_worth": 412},
    ]


def test_sre_campaign_reads_checkpointed_live_score_snapshot_before_legacy_export(tmp_path):
    adapter = SreCampaignAdapter(
        SreCampaignAdapterConfig(campaign_dir=tmp_path / "campaign", verify_clock=False)
    )
    adapter.world_path.mkdir(parents=True)
    rows = [
        {"player_id": "A", "empire": "Alpha Realm", "planets": 27, "score": 3, "net_worth": 300},
        {"player_id": "B", "empire": "Beta Realm", "planets": 28, "score": 2, "net_worth": 290},
    ]
    (adapter.world_path / "SRESCORE.TXT").write_text("stale legacy export", encoding="ascii")

    adapter._write_score_snapshot(rows)

    assert adapter.read_scores() == rows


def test_sre_global_score_probe_uses_disposable_world_and_commits_snapshot(tmp_path, monkeypatch):
    adapter = SreCampaignAdapter(
        SreCampaignAdapterConfig(campaign_dir=tmp_path / "campaign", verify_clock=False)
    )
    adapter.campaign_dir.mkdir()
    adapter.world_path.mkdir()
    (adapter.world_path / "state.bin").write_bytes(b"canonical")
    adapter.dosemu_config_path.write_text('$_cpu = "80486"\n', encoding="ascii")
    previous = [{"player_id": "A", "empire": "Before", "score": 3}]
    refreshed = [{"player_id": "A", "empire": "After", "score": 7}]
    adapter._write_score_snapshot(previous)
    probe_dirs: list[Path] = []

    def fake_probe(probe, virtual_time, transcript_path):
        assert virtual_time == datetime(2026, 7, 29, 12, tzinfo=timezone.utc)
        assert transcript_path == adapter.campaign_dir / "transcripts" / "maintenance.score.raw"
        assert probe.world_path != adapter.world_path
        assert (probe.world_path / "state.bin").read_bytes() == b"canonical"
        (probe.world_path / "state.bin").write_bytes(b"probe-mutated")
        probe_dirs.append(probe.campaign_dir)
        return refreshed

    monkeypatch.setattr(SreCampaignAdapter, "_run_global_score_probe", fake_probe)

    scores = adapter.probe_scores(
        datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
        adapter.campaign_dir / "transcripts" / "maintenance.score.raw",
    )

    assert scores == refreshed
    assert adapter.read_scores() == refreshed
    assert (adapter.world_path / "state.bin").read_bytes() == b"canonical"
    assert probe_dirs and not probe_dirs[0].exists()


def test_sre_global_score_probe_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    adapter = SreCampaignAdapter(
        SreCampaignAdapterConfig(campaign_dir=tmp_path / "campaign", verify_clock=False)
    )
    adapter.campaign_dir.mkdir()
    adapter.world_path.mkdir()
    (adapter.world_path / "state.bin").write_bytes(b"canonical")
    adapter.dosemu_config_path.write_text('$_cpu = "80486"\n', encoding="ascii")
    previous = [{"player_id": "A", "empire": "Before", "score": 3}]
    adapter._write_score_snapshot(previous)

    def fail_probe(_probe, _virtual_time, _transcript_path):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(SreCampaignAdapter, "_run_global_score_probe", fail_probe)

    with pytest.raises(RuntimeError, match="probe failed"):
        adapter.probe_scores(
            datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
            adapter.campaign_dir / "transcripts" / "maintenance.score.raw",
        )

    assert adapter.read_scores() == previous
    assert (adapter.world_path / "state.bin").read_bytes() == b"canonical"
    assert not list(adapter.campaign_dir.glob(".score-probe-*"))


def test_sre_unjoined_galaxy_menu_is_a_no_empire_score_result():
    assert _sre_no_empire_menu(
        """──[Galaxy Menu]──
[4] Join this Game
[6] Read Messages
[7] Send Messages
[8] Scores
"""
    )
    assert not _sre_no_empire_menu("[4] Join this Game\n[5] See Your Status")


def test_managed_sre_session_returns_to_bbs_before_reaping_container(monkeypatch):
    inner = FakePtySession(disconnect_on_read=True)
    docker_calls: list[list[str]] = []
    closed_containers: list[str] = []

    def fake_run(argv: list[str], **_kwargs: Any) -> None:
        docker_calls.append(argv)

    monkeypatch.setattr("bbs_gym.sre_campaign.subprocess.run", fake_run)
    session = ManagedDockerPtySession(
        inner,  # type: ignore[arg-type]
        "sre-test",
        "docker",
        on_close=closed_containers.append,
        graceful_exit_bytes=b"\x1b[15~",
    )

    session.close()
    session.close()

    assert inner.events == [("send", b"\x1b[15~"), "read", "close"]
    assert docker_calls == [["docker", "rm", "--force", "sre-test"]]
    assert closed_containers == ["sre-test"]


def test_managed_sre_session_reaps_container_but_reports_graceful_exit_timeout(monkeypatch):
    inner = FakePtySession(disconnect_on_read=False)
    docker_calls: list[list[str]] = []
    closed_containers: list[str] = []

    def fake_run(argv: list[str], **_kwargs: Any) -> None:
        docker_calls.append(argv)

    monkeypatch.setattr("bbs_gym.sre_campaign.subprocess.run", fake_run)
    session = ManagedDockerPtySession(
        inner,  # type: ignore[arg-type]
        "sre-test",
        "docker",
        on_close=closed_containers.append,
        graceful_exit_bytes=b"\x1b[15~",
        graceful_exit_timeout=0.0,
    )

    with pytest.raises(TimeoutError, match="refusing to treat the session as safely persisted"):
        session.close()

    assert inner.events == [("send", b"\x1b[15~"), "close"]
    assert docker_calls == [["docker", "rm", "--force", "sre-test"]]
    assert closed_containers == ["sre-test"]


def test_sre_status_waits_for_fields_and_accumulates_animated_output(tmp_path):
    adapter = SreCampaignAdapter(
        SreCampaignAdapterConfig(campaign_dir=tmp_path / "campaign", verify_clock=False)
    )
    heading = "Empire Status\n-*Persistent Empire*-\n"
    fields = "Score: 17\nTurns Left: 3\nMoney: 12,345\nPAUSED\n"
    observer = SequenceObserver(
        [
            _observation(heading, heading, 100),
            _observation(fields, heading + fields, 100 + len(heading)),
        ]
    )

    status = adapter._wait_for_screen(observer, _sre_status_screen_ready, timeout=1.0)  # type: ignore[arg-type]

    assert status.new_text == heading + fields
    assert status.transcript_byte_start == 100
    assert status.transcript_byte_end == 100 + len(heading) + len(fields)
    assert status.bytes_read == len(heading) + len(fields)
    assert _sre_status_screen_ready("Empire Status") is False
    assert extract_sre_metrics(status) == {
        "empire": "Persistent Empire",
        "score": 17,
        "turns_left": 3,
        "money": 12345,
    }
