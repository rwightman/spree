import json
from pathlib import Path

import pytest

from tty_agent.memory_replay import (
    StructuredJournalBatch,
    load_activity_memory_events,
    load_structured_journal_batches,
    select_batch_events,
    write_journal_prefix,
)
from tty_agent.memory_subsystem import MemoryEvent
from tty_agent.models import ScriptedModelAdapter
from tty_agent.structured_memory import StructuredMemoryConfig, StructuredMemorySubsystem


class RecordingModel(ScriptedModelAdapter):
    def __init__(self, responses):
        super().__init__(responses)
        self.compaction_messages = []

    def compaction_chat(self, messages):
        self.compaction_messages.append(messages)
        return super().chat(messages)


def _write_activity_log(path: Path, count: int) -> None:
    records = []
    for step in range(1, count + 1):
        records.append(
            {
                "step": step,
                "observation": {"model_text": f"observation {step}"},
                "action": {"action": "submit_line", "arguments": {"text": f"action {step}"}},
                "validation": {"accepted": True, "notes": []},
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_load_activity_memory_events_matches_runner_conversion(tmp_path):
    path = tmp_path / "activity.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "memory_context"}),
                json.dumps(
                    {
                        "step": 1,
                        "observation": {"model_text": "screen"},
                        "action": {"arguments": {"text": "north"}, "action": "submit_line"},
                        "validation": {"accepted": False, "notes": ["bad action"]},
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    events = load_activity_memory_events(path)

    assert events == {
        1: MemoryEvent(
            kind="terminal_step",
            step=1,
            observation="screen",
            action='{"action": "submit_line", "arguments": {"text": "north"}}',
            warnings=("bad action",),
        )
    }


def test_point_replay_uses_original_prefix_and_overlap_without_prior_drift(tmp_path):
    config = StructuredMemoryConfig(reconcile_every_events=2, overlap_events=2)
    source_root = tmp_path / "source"
    source = StructuredMemorySubsystem(source_root, config)
    source_handle = source.open_context("agent", "zork")
    source_handle.observe(MemoryEvent(kind="terminal_step", step=1, observation="observation 1"))
    source_handle.observe(MemoryEvent(kind="terminal_step", step=2, observation="observation 2"))
    source_handle.maybe_reconcile(
        ScriptedModelAdapter(['[{"op":"add","section":"fact","text":"original fact"}]']),
        force=True,
    )
    source_handle.observe(MemoryEvent(kind="terminal_step", step=3, observation="observation 3"))
    source_handle.observe(MemoryEvent(kind="terminal_step", step=4, observation="observation 4"))
    source_handle.maybe_reconcile(
        ScriptedModelAdapter(['[{"op":"set","key":"location","value":"original place"}]']),
        force=True,
    )
    source_handle.close()

    activity_path = tmp_path / "activity.jsonl"
    _write_activity_log(activity_path, 4)
    source_journal = source_root / "agent" / "zork" / "ops.jsonl"
    batches = load_structured_journal_batches(source_journal)
    events = load_activity_memory_events(activity_path)
    selected, overlap = select_batch_events(events, batches, 1, config.overlap_events)

    replay_root = tmp_path / "replay"
    replay_journal = replay_root / "agent" / "zork" / "ops.jsonl"
    write_journal_prefix(source_journal, replay_journal, 1)
    replay = StructuredMemorySubsystem(
        replay_root,
        config,
        reconcile_prompt_appendix="REPLAY APPENDIX",
    )
    replay_handle = replay.open_context("agent", "zork")
    assert [item.text for item in replay_handle.memory.facts] == ["original fact"]
    assert replay_handle.memory.state == {}
    replay_handle._covered_tail = overlap
    for event in selected:
        replay_handle.observe(event)
    model = RecordingModel(['[{"op":"set","key":"location","value":"alternate place"}]'])

    outcome = replay_handle.maybe_reconcile(model, force=True)

    assert outcome.status == "applied"
    assert replay_handle.memory.state == {"location": "alternate place"}
    prompt = "\n".join(message.content for message in model.compaction_messages[0])
    assert "REPLAY APPENDIX" in prompt
    assert "Already-recorded recent activity" in prompt
    assert "observation 2" in prompt
    assert "New activity to fold in" in prompt
    assert "observation 3" in prompt
    replay_handle.close()
    # The original second batch is untouched and remains independently replayable.
    assert load_structured_journal_batches(source_journal)[1].records == batches[1].records


def test_overlap_accumulates_across_small_batches_and_failure_markers():
    events = {
        step: MemoryEvent(kind="terminal_step", step=step, observation=f"observation {step}") for step in range(1, 5)
    }
    batches = [
        StructuredJournalBatch(0, "reconcile", (1, 2), ({"op": "add"},)),
        StructuredJournalBatch(1, "reconcile", (3,), ({"op": "set"},)),
        StructuredJournalBatch(2, "reconcile", (), ({"op": "reconcile_failed"},)),
        StructuredJournalBatch(3, "reconcile", (4,), ({"op": "set"},)),
    ]

    selected, overlap = select_batch_events(events, batches, 3, overlap_events=3)

    assert [event.step for event in selected] == [4]
    assert [event.step for event in overlap] == [1, 2, 3]


def test_overlap_stops_at_prior_session_audit():
    events = {
        step: MemoryEvent(kind="terminal_step", step=step, observation=f"observation {step}") for step in range(1, 5)
    }
    batches = [
        StructuredJournalBatch(0, "reconcile", (1, 2, 3), ({"op": "add"},)),
        StructuredJournalBatch(1, "audit", (), ({"op": "no_memory_change"},)),
        StructuredJournalBatch(2, "reconcile", (4,), ({"op": "set"},)),
    ]

    _selected, overlap = select_batch_events(events, batches, 2, overlap_events=3)

    assert overlap == []


def test_non_replayable_batch_error_names_batch_and_operations():
    batch = StructuredJournalBatch(
        0,
        "reconcile",
        (),
        ({"op": "reconcile_failed"},),
    )

    with pytest.raises(ValueError, match=r"reconcile; operations=reconcile_failed"):
        select_batch_events({}, [batch], 0, overlap_events=3)


@pytest.mark.parametrize(
    "torn_tail",
    [
        b'{"records":[',
        b'{"records":[{"reason":"\xe2',
    ],
)
def test_journal_loader_ignores_only_torn_final_fragment(tmp_path, torn_tail):
    journal = tmp_path / "ops.jsonl"
    complete = (
        json.dumps(
            {
                "records": [
                    {
                        "op": "set",
                        "batch": "reconcile",
                        "source_steps": [1],
                    }
                ]
            }
        ).encode("utf-8")
        + b"\n"
    )
    journal.write_bytes(complete + torn_tail)

    batches = load_structured_journal_batches(journal)

    assert len(batches) == 1
    assert batches[0].source_steps == (1,)


def test_journal_loader_rejects_corruption_before_final_fragment(tmp_path):
    journal = tmp_path / "ops.jsonl"
    complete = json.dumps(
        {
            "records": [
                {
                    "op": "set",
                    "batch": "reconcile",
                    "source_steps": [1],
                }
            ]
        }
    )
    journal.write_text(f"{complete}\nnot-json\n{complete}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"ops\.jsonl:2"):
        load_structured_journal_batches(journal)


def test_point_replay_rejects_missing_evidence_and_existing_destination(tmp_path):
    journal = tmp_path / "ops.jsonl"
    journal.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "op": "set",
                        "batch": "reconcile",
                        "source_steps": [10],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batches = load_structured_journal_batches(journal)

    with pytest.raises(ValueError, match="missing step 10"):
        select_batch_events({}, batches, 0, 3)

    destination = tmp_path / "copy" / "ops.jsonl"
    write_journal_prefix(journal, destination, 0)
    with pytest.raises(FileExistsError, match="already exists"):
        write_journal_prefix(journal, destination, 0)

    full_copy = tmp_path / "full" / "ops.jsonl"
    write_journal_prefix(journal, full_copy, 1)
    assert full_copy.read_bytes() == journal.read_bytes()
