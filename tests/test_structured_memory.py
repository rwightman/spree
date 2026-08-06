import json
from pathlib import Path

import pytest

from tty_agent.memory import JsonMemoryStore
from tty_agent.memory_subsystem import MemoryEvent, read_journal_records
from tty_agent.models import ModelError, ScriptedModelAdapter
from tty_agent.runner import ActivityBudget, ActivityProfile, ActivityRunner
from tty_agent.structured_memory import (
    StructuredMemoryConfig,
    StructuredMemorySubsystem,
)


def subsystem(tmp_path: Path, **config) -> StructuredMemorySubsystem:
    return StructuredMemorySubsystem(tmp_path / "memory", StructuredMemoryConfig(**config))


def events(handle, count: int, *, start: int = 1, text: str = "You are in a maze.") -> None:
    for index in range(start, start + count):
        handle.observe(MemoryEvent(kind="terminal_step", step=index, observation=text, action='{"action":"wait"}'))


def ops_model(*responses: str) -> ScriptedModelAdapter:
    return ScriptedModelAdapter(list(responses))


def test_operations_apply_and_journal(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 2)
    model = ops_model(
        json.dumps(
            [
                {"op": "add", "section": "goal", "text": "Enter the house", "priority": 1},
                {"op": "add", "section": "fact", "text": "The front door is boarded"},
                {"op": "add", "section": "hypothesis", "text": "The window might open"},
                {"op": "set", "key": "Location", "value": "West of House"},
            ]
        )
    )

    outcome = handle.maybe_reconcile(model, force=True)

    assert outcome.status == "applied"
    assert outcome.accepted_ops == 4
    assert outcome.rejected_ops == 0
    assert handle.memory.state == {"location": "West of House"}
    assert [goal.text for goal in handle.memory.goals] == ["Enter the house"]
    assert [item.text for item in handle.memory.facts] == ["The front door is boarded"]
    assert [item.text for item in handle.memory.hypotheses] == ["The window might open"]
    assert handle.memory.facts[0].source_steps == (1, 2)
    rendered = handle.render_context()
    assert "Enter the house" in rendered
    assert "location: West of House" in rendered
    records = read_journal_records(tmp_path / "memory" / "agent-001" / "zork" / "ops.jsonl")
    assert all(record["accepted"] for record in records)
    assert {record["op"] for record in records} == {"add", "set"}
    handle.close()


def test_promote_contradict_and_transition(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "goal", "text": "Open the window"},
                    {"op": "add", "section": "hypothesis", "text": "The window opens"},
                    {"op": "add", "section": "fact", "text": "The mailbox holds a leaflet"},
                ]
            )
        ),
        force=True,
    )
    events(handle, 1, start=2)

    outcome = handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "promote", "id": "m2"},
                    {"op": "contradict", "id": "m3", "replacement": "The mailbox is empty now"},
                    {"op": "transition", "id": "m1", "status": "done"},
                ]
            )
        ),
        force=True,
    )

    assert outcome.accepted_ops == 3
    assert [item.text for item in handle.memory.facts] == ["The window opens", "The mailbox is empty now"]
    assert handle.memory.hypotheses == []
    assert handle.memory.goals == []
    exits = {entry["exit"] for entry in handle.memory.archive}
    assert exits == {"contradicted", "goal_closed"}
    handle.close()


def test_invalid_operations_reject_individually(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)

    outcome = handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "A valid fact"},
                    {"op": "add", "section": "fact", "text": "A valid fact"},
                    {"op": "explode"},
                    {"op": "revise", "id": "m99", "text": "missing"},
                    {"op": "transition", "id": "m1", "status": "done"},
                    {"op": "add", "section": "fact", "text": ""},
                ]
            )
        ),
        force=True,
    )

    assert outcome.accepted_ops == 1
    assert outcome.rejected_ops == 5
    assert [item.text for item in handle.memory.facts] == ["A valid fact"]
    journal_path = handle.key.directory() / "ops.jsonl"
    records = read_journal_records(journal_path)
    rejected = [record for record in records if not record["accepted"]]
    assert len(rejected) == 5
    assert all(record["reason"] for record in rejected)
    handle.close()


def test_capacity_is_code_owned_and_journaled(tmp_path):
    handle = subsystem(tmp_path, max_facts=2).open_context("agent-001", "zork")
    events(handle, 1)

    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "fact one"},
                    {"op": "add", "section": "fact", "text": "fact two"},
                    {"op": "add", "section": "fact", "text": "fact three"},
                ]
            )
        ),
        force=True,
    )

    assert [item.text for item in handle.memory.facts] == ["fact two", "fact three"]
    assert handle.memory.archive[0]["exit"] == "capacity"
    assert handle.memory.archive[0]["item"]["id"] == "m1"
    journal_path = handle.key.directory() / "ops.jsonl"
    records = read_journal_records(journal_path)
    demotes = [record for record in records if record["op"] == "demote"]
    assert len(demotes) == 1
    assert demotes[0]["origin"] == "system"
    assert demotes[0]["reason"] == "capacity"
    handle.close()


def test_replay_property_reopen_reproduces_store(tmp_path):
    system = subsystem(tmp_path, max_facts=3)
    handle = system.open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "goal", "text": "goal a", "priority": 2},
                    {"op": "add", "section": "fact", "text": "fact a"},
                    {"op": "add", "section": "hypothesis", "text": "maybe b"},
                    {"op": "set", "key": "location", "value": "cellar"},
                ]
            )
        ),
        force=True,
    )
    events(handle, 1, start=2)
    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "promote", "id": "m3"},
                    {"op": "add", "section": "fact", "text": "fact c"},
                    {"op": "add", "section": "fact", "text": "fact d"},
                    {"op": "transition", "id": "m1", "status": "blocked", "reason": "grue"},
                ]
            )
        ),
        force=True,
    )
    snapshot = handle.memory.to_dict()
    handle.close()

    reopened = system.open_context("agent-001", "zork")
    assert reopened.memory.to_dict() == snapshot
    store_cache = json.loads((reopened.key.directory() / "store.json").read_text(encoding="utf-8"))
    assert store_cache == snapshot
    reopened.close()


def test_torn_journal_line_is_truncated_on_open(tmp_path):
    system = subsystem(tmp_path)
    handle = system.open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(json.dumps([{"op": "add", "section": "fact", "text": "durable fact"}])),
        force=True,
    )
    snapshot = handle.memory.to_dict()
    handle.close()
    journal_path = tmp_path / "memory" / "agent-001" / "zork" / "ops.jsonl"
    with journal_path.open("a", encoding="utf-8") as broken:
        broken.write('{"op":"add","accepted":true,"torn')

    reopened = system.open_context("agent-001", "zork")

    assert reopened.memory.to_dict() == snapshot
    assert not journal_path.read_text(encoding="utf-8").rstrip().endswith("torn")
    reopened.close()


def test_zero_ops_retries_once_then_advances_with_marker(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 2)
    model = ops_model("[]", "[]")

    outcome = handle.maybe_reconcile(model, force=True)

    assert outcome.status == "no_memory_change"
    assert outcome.covered_steps == 2
    assert model.responses == []  # both scripted responses consumed: one retry happened
    journal_path = handle.key.directory() / "ops.jsonl"
    records = read_journal_records(journal_path)
    assert records[-1]["op"] == "no_memory_change"
    # The boundary advanced: pending drained into the overlap tail.
    assert handle.maybe_reconcile(ops_model("[]"), force=True) is None
    handle.close()


class TruncatingModel(ScriptedModelAdapter):
    def compaction_chat(self, messages):
        text = super().compaction_chat(messages)
        self.last_provider_metadata = {"finish_reason": "length"}
        return text


def test_truncated_response_fails_without_advancing(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 2)

    outcome = handle.maybe_reconcile(
        TruncatingModel([json.dumps([{"op": "add", "section": "fact", "text": "half a f"}])]),
        force=True,
    )

    assert outcome.status == "failed"
    assert "truncated" in outcome.error
    assert handle.memory.is_empty()
    # Events stay pending; a later good response still covers them.
    good = handle.maybe_reconcile(
        ops_model(json.dumps([{"op": "add", "section": "fact", "text": "whole fact"}])),
        force=True,
    )
    assert good.status == "applied"
    assert good.covered_steps == 2
    handle.close()


def test_reconcile_failure_sets_backoff(tmp_path):
    handle = subsystem(tmp_path, reconcile_every_events=2).open_context("agent-001", "zork")
    events(handle, 2)

    class FailingModel(ScriptedModelAdapter):
        def compaction_chat(self, messages):
            raise ModelError("provider down")

    outcome = handle.maybe_reconcile(FailingModel([]))
    assert outcome.status == "failed"
    # Cadence is due but backoff suppresses an immediate retry.
    assert handle.maybe_reconcile(ops_model("[]")) is None
    handle.close()


def test_commit_retries_then_reports_failure(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)

    class AlwaysFailing(ScriptedModelAdapter):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def compaction_chat(self, messages):
            self.calls += 1
            raise ModelError("provider down")

    model = AlwaysFailing()
    outcome = handle.commit(model)

    assert outcome.status == "failed"
    assert outcome.attempts == 2
    assert model.calls == 2
    journal_path = handle.key.directory() / "ops.jsonl"
    records = read_journal_records(journal_path)
    assert records[-1]["op"] == "commit_failed"
    handle.close()


def test_context_lock_rejects_concurrent_open(tmp_path):
    system = subsystem(tmp_path)
    handle = system.open_context("agent-001", "zork")

    with pytest.raises(RuntimeError, match="already in use"):
        system.open_context("agent-001", "zork")

    handle.close()
    reopened = system.open_context("agent-001", "zork")
    reopened.close()


def test_fingerprints_are_stable_and_config_sensitive(tmp_path):
    base = subsystem(tmp_path).fingerprints()
    again = subsystem(tmp_path).fingerprints()
    tuned = subsystem(tmp_path, max_facts=5).fingerprints()
    prompted = StructuredMemorySubsystem(
        tmp_path / "prompted",
        reconcile_prompt_appendix="Correct stale current values before adding facts.",
    ).fingerprints()

    assert set(base) == {"schema", "mutation", "prompt"}
    assert base == again
    assert base["mutation"] != tuned["mutation"]
    assert base["schema"] == tuned["schema"]
    assert base["schema"] == prompted["schema"]
    assert base["mutation"] == prompted["mutation"]
    assert base["prompt"] != prompted["prompt"]


def test_runner_integration_reconcile_prompt_and_commit(tmp_path):
    from test_runner import FakeAgent, PromptRecordingModel

    class MemoryModel(PromptRecordingModel):
        """Decisions come from the script; memory calls return ops."""

        def compaction_chat(self, messages):
            self.chats.append("\n".join(message.content for message in messages))
            return json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "The menu accepts single letters"},
                    {"op": "set", "key": "location", "value": "main menu"},
                ]
            )

        def memory_chat(self, messages):
            return self.compaction_chat(messages)

    model = MemoryModel(
        [
            '{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"action": "hangup", "arguments": {}}',
        ]
    )
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="test structured memory"),
        log_path=tmp_path / "activity.jsonl",
        memory_store=JsonMemoryStore(tmp_path / "legacy-memory"),
        memory_subsystem=subsystem(tmp_path),
        memory_context_id="bbs-menu",
    )

    result = runner.run(FakeAgent(), model, ActivityBudget(max_decision_ticks=3))

    assert result.stop_reason == "hangup"
    # Decision prompts carry the structured memory section, not legacy memory.
    second_prompt = result.steps[1].prompt["user"]
    assert "Memory:" in second_prompt
    assert "Campaign memory:" not in second_prompt
    memory_chats = [chat for chat in model.chats if "structured working memory" in chat]
    assert memory_chats
    assert all(
        "Keep reasoning concise and reserve enough output for the required JSON" in chat
        for chat in memory_chats
    )
    # The commit ran against the persistent context store.
    journal_path = tmp_path / "memory" / "agent-001" / "bbs-menu" / "ops.jsonl"
    records = read_journal_records(journal_path)
    assert any(record["batch"] == "commit" for record in records)
    store = json.loads((tmp_path / "memory" / "agent-001" / "bbs-menu" / "store.json").read_text(encoding="utf-8"))
    assert store["state"] == {"location": "main menu"}
    # Legacy campaign memory was not written.
    assert JsonMemoryStore(tmp_path / "legacy-memory").load("agent-001") == {}
    # The run log attributes the arm (subsystem + fingerprints) and records
    # the commit outcome.
    log_records = [json.loads(line) for line in (tmp_path / "activity.jsonl").read_text().splitlines()]
    context_record = next(record for record in log_records if record.get("type") == "memory_context")
    assert context_record["subsystem"] == "structured"
    assert set(context_record["fingerprints"]) == {"schema", "mutation", "prompt"}
    commit_record = next(record for record in log_records if record.get("type") == "memory_commit")
    assert commit_record["status"] == "applied"
    # The context directory carries a manifest for offline metric tooling.
    manifest = json.loads((tmp_path / "memory" / "agent-001" / "bbs-menu" / "manifest.json").read_text())
    assert manifest["subsystem"] == "structured"
    assert manifest["config"]["max_facts"] == 30
    # A second run reopens the same context and sees prior memory.
    model2 = MemoryModel(['{"action": "hangup", "arguments": {}}'])
    result2 = runner.run(FakeAgent(), model2, ActivityBudget(max_decision_ticks=2))
    assert "The menu accepts single letters" in result2.steps[0].prompt["user"]


def test_state_eviction_order_survives_replay(tmp_path):
    system = subsystem(tmp_path, max_state_entries=2)
    handle = system.open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "set", "key": "a", "value": "1"},
                    {"op": "set", "key": "b", "value": "1"},
                    {"op": "set", "key": "c", "value": "1"},
                    {"op": "set", "key": "a", "value": "2"},
                ]
            )
        ),
        force=True,
    )

    # Two evictions happened live (a, then b); the re-set "a" must survive.
    assert handle.memory.state == {"c": "1", "a": "2"}
    snapshot = handle.memory.to_dict()
    handle.close()

    reopened = system.open_context("agent-001", "zork")
    assert reopened.memory.to_dict() == snapshot
    reopened.close()


def test_archive_cap_survives_replay(tmp_path):
    system = subsystem(tmp_path, max_archive_entries=1)
    handle = system.open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "first"},
                    {"op": "add", "section": "fact", "text": "second"},
                    {"op": "contradict", "id": "m1"},
                    {"op": "contradict", "id": "m2"},
                ]
            )
        ),
        force=True,
    )

    assert len(handle.memory.archive) == 1
    assert handle.memory.archive[0]["item"]["text"] == "second"
    snapshot = handle.memory.to_dict()
    handle.close()

    reopened = system.open_context("agent-001", "zork")
    assert reopened.memory.to_dict() == snapshot
    reopened.close()


def test_contradict_replacement_respects_fact_cap(tmp_path):
    handle = subsystem(tmp_path, max_facts=1).open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "stable fact"},
                    {"op": "add", "section": "hypothesis", "text": "shaky idea"},
                ]
            )
        ),
        force=True,
    )
    events(handle, 1, start=2)

    handle.maybe_reconcile(
        ops_model(json.dumps([{"op": "contradict", "id": "m2", "replacement": "corrected idea"}])),
        force=True,
    )

    assert len(handle.memory.facts) == 1
    records = read_journal_records(handle.key.directory() / "ops.jsonl")
    assert any(record["op"] == "demote" and record["section"] == "fact" for record in records)
    handle.close()


def test_non_dict_operations_are_rejected_and_journaled(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)

    outcome = handle.maybe_reconcile(
        ops_model('[{"op": "add", "section": "fact", "text": "real"}, "garbage", 42]'),
        force=True,
    )

    assert outcome.accepted_ops == 1
    assert outcome.rejected_ops == 2
    records = read_journal_records(handle.key.directory() / "ops.jsonl")
    rejected = [record for record in records if not record["accepted"]]
    assert [record["op"] for record in rejected] == ["invalid", "invalid"]
    assert all(record["reason"] == "operation must be a JSON object" for record in rejected)
    handle.close()


class UsageModel(ScriptedModelAdapter):
    def compaction_chat(self, messages):
        text = super().compaction_chat(messages)
        self.last_usage = {"total_tokens": 10}
        return text

    def memory_chat(self, messages):
        text = super().memory_chat(messages)
        self.last_usage = {"total_tokens": 10}
        return text


def test_bounded_prefix_reconcile_and_commit_drain(tmp_path):
    handle = subsystem(tmp_path, max_events_per_reconcile=2).open_context("agent-001", "zork")
    events(handle, 5)

    first = handle.maybe_reconcile(
        UsageModel([json.dumps([{"op": "set", "key": "k1", "value": "v"}])]),
        force=True,
    )

    assert first.covered_steps == 2
    assert first.usage == {"total_tokens": 10}
    assert len(handle._pending) == 3

    commit_model = UsageModel(
        [
            json.dumps([{"op": "set", "key": "k2", "value": "v"}]),
            json.dumps([{"op": "set", "key": "k3", "value": "v"}]),
            "[]",
        ]
    )
    outcome = handle.commit(commit_model)

    assert outcome.status == "applied"
    assert outcome.attempts == 3
    assert handle._pending == []
    assert outcome.usage == {"total_tokens": 30}
    assert outcome.audit_status == "no_memory_change"
    handle.close()


def test_commit_separates_extraction_and_audit_seams(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)

    class SeamModel(ScriptedModelAdapter):
        def __init__(self):
            super().__init__([])
            self.compaction_calls = 0
            self.memory_calls = 0

        def compaction_chat(self, messages):
            self.compaction_calls += 1
            return json.dumps([{"op": "add", "section": "fact", "text": "from extraction seam"}])

        def memory_chat(self, messages):
            self.memory_calls += 1
            return "[]"

    model = SeamModel()
    outcome = handle.commit(model)

    assert outcome.status == "applied"
    assert outcome.audit_status == "no_memory_change"
    assert model.compaction_calls == 1
    assert model.memory_calls == 1
    assert [item.text for item in handle.memory.facts] == ["from extraction seam"]
    handle.close()


def test_runner_releases_memory_lock_on_unexpected_failure(tmp_path):
    from test_runner import FakeAgent

    class ExplodingAgent(FakeAgent):
        def observe_turn(self, **kwargs):
            raise RuntimeError("terminal wedged")

    system = subsystem(tmp_path)
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="lock release"),
        memory_store=JsonMemoryStore(tmp_path / "legacy"),
        memory_subsystem=system,
        memory_context_id="bbs-menu",
    )

    with pytest.raises(RuntimeError, match="terminal wedged"):
        runner.run(ExplodingAgent(), ops_model(), ActivityBudget(max_decision_ticks=2))

    handle = system.open_context("agent-001", "bbs-menu")
    handle.close()


def test_bootstrap_stage_gets_bootstrap_render(tmp_path):
    from test_runner import FakeAgent
    from tty_agent.memory_subsystem import CommitOutcome

    class StubHandle:
        def __init__(self):
            self.renders = []
            self.reconcile_forces = []
            self.forced_outcomes = []

        def observe(self, event):
            pass

        def render_context(self, budget_chars=None):
            self.renders.append("context")
            return "ctx"

        def render_bootstrap(self, budget_chars=None):
            self.renders.append("bootstrap")
            return "boot"

        def maybe_reconcile(self, model, *, force=False):
            self.reconcile_forces.append(force)
            if force and self.forced_outcomes:
                return self.forced_outcomes.pop(0)
            return None

        def commit(self, model, extra_evidence=""):
            return CommitOutcome(status="no_memory_change")

        def close(self):
            pass

    class StubSubsystem:
        name = "stub"

        def __init__(self):
            self.handle = StubHandle()

        def fingerprints(self):
            return {}

        def open_context(self, agent_id, context_id):
            return self.handle

    from tty_agent.memory_subsystem import ReconcileOutcome

    stub = StubSubsystem()
    stub.handle.forced_outcomes = [
        ReconcileOutcome(status="applied", accepted_ops=1),
        ReconcileOutcome(status="applied", accepted_ops=1),
    ]
    runner = ActivityRunner(
        ActivityProfile(name="bbs-menu", objective="stage renders", prompt_mode="stateful_delta"),
        memory_store=JsonMemoryStore(tmp_path / "memory"),
        memory_subsystem=stub,
        memory_context_id="bbs-menu",
    )
    model = ScriptedModelAdapter(
        [
            '{"action": "submit_line", "arguments": {"text": "?"}}',
            '{"action": "hangup", "arguments": {}}',
        ]
    )

    runner.run(FakeAgent(), model, ActivityBudget(max_decision_ticks=3))

    assert stub.handle.renders[0] == "bootstrap"
    assert set(stub.handle.renders[1:]) == {"context"}
    # The bootstrap drained the whole scripted backlog before rendering:
    # two applied catch-up passes plus the final empty-pending None.
    assert stub.handle.reconcile_forces.count(True) == 3


def test_randomized_operations_replay_property(tmp_path):
    import random

    for seed in (1, 2, 3):
        system = subsystem(
            tmp_path / f"seed{seed}",
            max_goals=2,
            max_facts=3,
            max_hypotheses=2,
            max_state_entries=2,
            max_archive_entries=4,
        )
        handle = system.open_context("agent-001", "zork")
        rng = random.Random(seed)
        keys = ["location", "score", "turns"]
        for step in range(1, 7):
            events(handle, 1, start=step)
            ops = []
            for _ in range(rng.randint(2, 7)):
                choice = rng.random()
                if choice < 0.35:
                    ops.append({"op": "set", "key": rng.choice(keys), "value": f"v{rng.randint(0, 30)}"})
                elif choice < 0.6:
                    section = rng.choice(["goal", "fact", "hypothesis"])
                    ops.append({"op": "add", "section": section, "text": f"{section} item {rng.randint(0, 12)}"})
                elif choice < 0.7:
                    ops.append({"op": "promote", "id": f"m{rng.randint(1, 12)}"})
                elif choice < 0.8:
                    ops.append(
                        {
                            "op": "contradict",
                            "id": f"m{rng.randint(1, 12)}",
                            "replacement": f"corrected {rng.randint(0, 12)}",
                        }
                    )
                elif choice < 0.9:
                    ops.append(
                        {
                            "op": "transition",
                            "id": f"m{rng.randint(1, 12)}",
                            "status": rng.choice(["done", "blocked"]),
                            "reason": "because",
                        }
                    )
                else:
                    ops.append(
                        {"op": "revise", "id": f"m{rng.randint(1, 12)}", "text": f"revised {rng.randint(0, 12)}"}
                    )
            handle.maybe_reconcile(ops_model(json.dumps(ops)), force=True)
        snapshot = handle.memory.to_dict()
        handle.close()

        reopened = system.open_context("agent-001", "zork")
        assert reopened.memory.to_dict() == snapshot, f"replay diverged for seed {seed}"
        reopened.close()


def test_non_string_op_name_is_rejected_not_crashing(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)

    outcome = handle.maybe_reconcile(
        ops_model('[{"op": []}, {"op": null}, {"op": {"nested": true}}]'),
        force=True,
    )

    assert outcome.status == "no_memory_change"
    assert outcome.rejected_ops == 3
    records = read_journal_records(handle.key.directory() / "ops.jsonl")
    rejected = [record for record in records if not record["accepted"]]
    assert all(record["reason"] == "op must be a string" for record in rejected)
    handle.close()


def test_corrupt_journal_does_not_strand_the_lock(tmp_path):
    system = subsystem(tmp_path)
    handle = system.open_context("agent-001", "zork")
    handle.close()
    journal_path = tmp_path / "memory" / "agent-001" / "zork" / "ops.jsonl"
    journal_path.write_text('not json at all\n{"records": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt memory journal"):
        system.open_context("agent-001", "zork")

    # The failed open released its lock: the same corruption error repeats
    # instead of a spurious "already in use".
    with pytest.raises(ValueError, match="corrupt memory journal"):
        system.open_context("agent-001", "zork")


def test_config_validation_rejects_broken_values():
    with pytest.raises(ValueError, match="max_events_per_reconcile"):
        StructuredMemoryConfig(max_events_per_reconcile=0)
    with pytest.raises(ValueError, match="max_state_entries"):
        StructuredMemoryConfig(max_state_entries=0)
    with pytest.raises(ValueError, match="overlap_events"):
        StructuredMemoryConfig(overlap_events=-1)
    with pytest.raises(ValueError, match="max_facts"):
        StructuredMemoryConfig(max_facts=-3)


def test_zero_overlap_keeps_no_covered_tail(tmp_path):
    handle = subsystem(tmp_path, overlap_events=0).open_context("agent-001", "zork")
    events(handle, 3)

    handle.maybe_reconcile(
        ops_model(json.dumps([{"op": "set", "key": "k", "value": "v"}])),
        force=True,
    )

    assert handle._covered_tail == []
    assert handle._pending == []
    handle.close()


def test_contradict_replacement_is_journaled_as_add(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(json.dumps([{"op": "add", "section": "fact", "text": "wrong fact"}])),
        force=True,
    )
    events(handle, 1, start=2)

    handle.maybe_reconcile(
        ops_model(json.dumps([{"op": "contradict", "id": "m1", "replacement": "right fact"}])),
        force=True,
    )

    records = read_journal_records(handle.key.directory() / "ops.jsonl")
    contradict = next(record for record in records if record["op"] == "contradict")
    assert "replacement" not in contradict["fields"]
    replacement_add = next(
        record
        for record in records
        if record["op"] == "add" and record["reason"] == "replacement"
    )
    assert replacement_add["origin"] == "model"
    assert replacement_add["fields"] == {"text": "right fact", "replaces": "m1"}
    assert [item.text for item in handle.memory.facts] == ["right fact"]
    handle.close()


def test_events_render_observed_before_action(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    text = handle._events_text(
        [
            MemoryEvent(
                kind="terminal_step",
                step=4,
                observation="A grue lurks here.",
                action='{"action":"submit_line"}',
                warnings=("action_error: send failed",),
            )
        ]
    )

    assert text.index("observed:") < text.index("action in response:")
    assert "warning: action_error: send failed" in text
    handle.close()


def test_memory_event_carries_failure_warnings(tmp_path):
    from types import SimpleNamespace

    from tty_agent.runner import StepRecord

    observed = []

    class Recorder:
        def observe(self, event):
            observed.append(event)

    runner = ActivityRunner(ActivityProfile(name="bbs-menu", objective="warnings"))
    state = SimpleNamespace(memory_handle=Recorder())
    runner._observe_memory_event(
        state,
        StepRecord(
            step=3,
            observation={"model_text": "screen"},
            prompt={},
            action={"action": "submit_line", "arguments": {"text": "x"}},
            validation={"accepted": False, "notes": ["action_error: send failed"]},
            execution={},
            budget={},
        ),
    )
    # Terminal observation records carry no accepted flag and stay clean.
    runner._observe_memory_event(
        state,
        StepRecord(
            step=4,
            observation={"model_text": "bye"},
            prompt={},
            action=None,
            validation={"terminal": True},
            execution={},
            budget={},
        ),
    )

    assert observed[0].warnings == ("action_error: send failed",)
    assert observed[1].warnings == ()


def test_commit_presents_extra_evidence_once(tmp_path):
    handle = subsystem(tmp_path, max_events_per_reconcile=2).open_context("agent-001", "zork")
    events(handle, 5)

    class RecordingOpsModel(ScriptedModelAdapter):
        def __init__(self, responses):
            super().__init__(responses)
            self.compaction_prompts = []
            self.memory_prompts = []

        def compaction_chat(self, messages):
            self.compaction_prompts.append("\n".join(message.content for message in messages))
            return super().compaction_chat(messages)

        def memory_chat(self, messages):
            self.memory_prompts.append("\n".join(message.content for message in messages))
            return super().memory_chat(messages)

    model = RecordingOpsModel(
        [
            json.dumps([{"op": "set", "key": "k1", "value": "v"}]),
            json.dumps([{"op": "set", "key": "k2", "value": "v"}]),
            json.dumps([{"op": "set", "key": "k3", "value": "v"}]),
            "[]",
            "[]",
        ]
    )

    outcome = handle.commit(model, extra_evidence="FORUM-NOTE from another player")

    assert outcome.status == "applied"
    assert outcome.attempts == 5
    assert len(model.compaction_prompts) == 4
    assert len(model.memory_prompts) == 1
    # Social evidence appears exactly once in extraction and never leaks into
    # the independent eventless audit.
    assert ["FORUM-NOTE" in prompt for prompt in model.compaction_prompts] == [False, False, False, True]
    assert "FORUM-NOTE" not in model.memory_prompts[0]
    assert "cleanup-only final audit" in model.memory_prompts[0]
    handle.close()


def test_final_audit_cleans_extracted_memory_without_new_event_input(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1, text="The brass lantern is now in the trophy case.")

    class RecordingModel(ScriptedModelAdapter):
        def __init__(self):
            super().__init__([])
            self.audit_prompt = ""

        def compaction_chat(self, messages):
            return json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "The brass lantern is in hand"},
                    {"op": "add", "section": "fact", "text": "The brass lantern is in the trophy case"},
                ]
            )

        def audit_chat(self, messages):
            self.audit_prompt = "\n".join(message.content for message in messages)
            return json.dumps([{"op": "contradict", "id": "m1"}])

    model = RecordingModel()
    outcome = handle.commit(model)

    assert outcome.status == "applied"
    assert outcome.audit_status == "applied"
    assert outcome.audit_accepted_ops == 1
    assert [item.text for item in handle.memory.facts] == ["The brass lantern is in the trophy case"]
    assert "The brass lantern is in hand" in model.audit_prompt
    assert "The brass lantern is in the trophy case" in model.audit_prompt
    assert "New activity to fold in" not in model.audit_prompt
    records = read_journal_records(handle.key.directory() / "ops.jsonl")
    assert [record["batch"] for record in records if record["accepted"] and record["op"] != "no_memory_change"] == [
        "commit",
        "commit",
        "audit",
    ]
    handle.close()


def test_final_audit_rejects_operations_that_can_grow_memory(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(json.dumps([{"op": "add", "section": "fact", "text": "The cellar is dark"}])),
        force=True,
    )
    before = handle.memory.to_dict()

    outcome = handle.commit(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "An invented fact"},
                    {"op": "set", "key": "location", "value": "attic"},
                    {"op": "contradict", "id": "m1", "replacement": "The cellar is lit"},
                ]
            )
        )
    )

    assert outcome.status == "no_memory_change"
    assert outcome.audit_status == "no_memory_change"
    assert outcome.audit_rejected_ops == 3
    assert handle.memory.to_dict() == before
    audit_records = [
        record for record in read_journal_records(handle.key.directory() / "ops.jsonl") if record["batch"] == "audit"
    ]
    assert [record["op"] for record in audit_records] == ["add", "set", "contradict", "no_memory_change"]
    assert all(not record["accepted"] for record in audit_records[:-1])
    handle.close()


def test_final_audit_failure_is_nonfatal_after_successful_extraction(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 1)

    class AuditFailureModel(ScriptedModelAdapter):
        def compaction_chat(self, messages):
            return json.dumps([{"op": "add", "section": "fact", "text": "The window is open"}])

        def audit_chat(self, messages):
            raise ModelError("audit endpoint unavailable")

    outcome = handle.commit(AuditFailureModel([]))

    assert outcome.status == "applied"
    assert outcome.audit_status == "failed"
    assert outcome.audit_error == "audit endpoint unavailable"
    assert [item.text for item in handle.memory.facts] == ["The window is open"]
    records = read_journal_records(handle.key.directory() / "ops.jsonl")
    assert records[-1]["op"] == "audit_failed"
    assert not any(record["op"] == "commit_failed" for record in records)
    handle.close()


def test_final_audit_sees_every_active_item_beyond_bootstrap_render_budget(tmp_path):
    handle = subsystem(tmp_path, bootstrap_budget_chars=40).open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "first durable fact"},
                    {"op": "add", "section": "fact", "text": "middle durable fact"},
                    {"op": "add", "section": "fact", "text": "last durable fact"},
                ]
            )
        ),
        force=True,
    )

    class AuditPromptModel(ScriptedModelAdapter):
        def __init__(self):
            super().__init__([])
            self.audit_prompt = ""

        def audit_chat(self, messages):
            self.audit_prompt = "\n".join(message.content for message in messages)
            return "[]"

    model = AuditPromptModel()
    outcome = handle.commit(model)

    assert outcome.audit_status == "no_memory_change"
    assert "first durable fact" in model.audit_prompt
    assert "middle durable fact" in model.audit_prompt
    assert "last durable fact" in model.audit_prompt
    handle.close()


def test_render_shares_budget_across_sections(tmp_path):
    from tty_agent.structured_memory import SimpleItem

    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    handle.memory.state["location"] = "cellar"
    handle.memory.facts = [SimpleItem(id=f"m{index}", text="F" * 250) for index in range(1, 21)]
    handle.memory.hypotheses = [SimpleItem(id="m21", text="the troll might accept the fish")]

    rendered = handle.render_context(1_000)

    # A prefix cut would spend the whole budget on facts; fair shares keep
    # every populated section visible.
    assert "State:" in rendered
    assert "Hypotheses (unverified):" in rendered
    assert "the troll might accept the fish" in rendered
    assert len(rendered) <= 1_000
    handle.close()


def test_render_bootstrap_includes_unreconciled_pending(tmp_path):
    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    events(handle, 2, text="The trapdoor slams shut.")

    bootstrap = handle.render_bootstrap()

    assert "not yet reconciled" in bootstrap
    assert "The trapdoor slams shut." in bootstrap
    assert "not yet reconciled" not in handle.render_context()
    handle.close()


def test_read_journal_records_distinguishes_torn_from_corrupt(tmp_path):
    path = tmp_path / "ops.jsonl"
    good_line = json.dumps({"records": [{"op": "x", "accepted": True}]})

    path.write_text(good_line + "\n" + '{"torn', encoding="utf-8")
    assert len(read_journal_records(path)) == 1

    path.write_text(good_line + "\n" + "corrupt complete line\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_journal_records(path)


def test_reopen_with_changed_config_is_refused(tmp_path):
    system = subsystem(tmp_path, max_archive_entries=1)
    handle = system.open_context("agent-001", "zork")
    events(handle, 1)
    handle.maybe_reconcile(
        ops_model(
            json.dumps(
                [
                    {"op": "add", "section": "fact", "text": "first"},
                    {"op": "add", "section": "fact", "text": "second"},
                    {"op": "contradict", "id": "m1"},
                    {"op": "contradict", "id": "m2"},
                ]
            )
        ),
        force=True,
    )
    snapshot = handle.memory.to_dict()
    handle.close()

    # Replaying under a larger archive cap would resurrect a trimmed entry;
    # the manifest check refuses instead of silently changing memory.
    with pytest.raises(RuntimeError, match="mutation.*fingerprint"):
        subsystem(tmp_path, max_archive_entries=2).open_context("agent-001", "zork")

    # The refusal released the lock, and the original config still replays
    # to exactly the state the journal's writer saw.
    reopened = subsystem(tmp_path, max_archive_entries=1).open_context("agent-001", "zork")
    assert reopened.memory.to_dict() == snapshot
    reopened.close()


def test_render_bootstrap_reserves_budget_for_pending(tmp_path):
    from tty_agent.structured_memory import SimpleItem

    handle = subsystem(tmp_path).open_context("agent-001", "zork")
    handle.memory.facts = [SimpleItem(id=f"m{index}", text="F" * 250) for index in range(1, 21)]
    events(handle, 1, text="The cyclops fled through the wall.")

    bootstrap = handle.render_bootstrap(500)

    # Committed memory alone exceeds the budget, but pending evidence keeps
    # its reserved share instead of being prefix-clipped away.
    assert len(bootstrap) <= 500
    assert "not yet reconciled" in bootstrap
    assert "The cyclops fled" in bootstrap
    handle.close()
