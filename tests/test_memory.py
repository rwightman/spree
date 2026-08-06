from tty_agent.memory import JsonMemoryStore, MemoryDocumentLimits, bound_memory_document
from tty_agent.models import MemoryPatch


def test_memory_store_dedupes_and_caps_lists(tmp_path):
    store = JsonMemoryStore(tmp_path / "memory", max_list_items=3)
    store.save("agent", {"durable_facts": ["old", "repeat"]})

    merged = store.save_patch("agent", MemoryPatch({"durable_facts": ["repeat", "new1", "new2", "new3"]}))

    assert merged == {"durable_facts": ["new1", "new2", "new3"]}


def test_bound_memory_document_enforces_string_list_and_total_limits():
    bounded, report = bound_memory_document(
        {
            "durable_facts": ["Alpha", "Alpha", "alpha", "discarded by count"],
            "notes": {"location": "x" * 80},
        },
        MemoryDocumentLimits(max_document_chars=100, max_string_chars=20, max_list_items=2),
    )

    # Dedupe is exact and deterministic: names that differ by case are not
    # conflated, while an exact repeat is removed.
    assert bounded["durable_facts"] == ["Alpha", "alpha"]
    assert len(bounded["notes"]["location"]) <= 20
    assert report["after_chars"] <= 100
    assert report["deduped_list_items"] == 1
    assert report["trimmed_list_items"] == 1
    assert report["changed"] is True


def test_bounded_memory_merge_prefers_ranked_patch_then_preserves_old_entries(tmp_path):
    store = JsonMemoryStore(tmp_path / "memory")
    store.save("agent", {"durable_facts": ["old-a", "old-b"]})

    merged, report = store.save_patch_with_report(
        "agent",
        MemoryPatch({"durable_facts": ["new-high", "new-low"]}),
        limits=MemoryDocumentLimits(max_document_chars=500, max_string_chars=100, max_list_items=3),
    )

    assert merged == {"durable_facts": ["new-high", "new-low", "old-a"]}
    assert report["document"]["changed"] is True
    assert report["document"]["trimmed_list_items"] == 1


def test_memory_store_merges_nested_dicts(tmp_path):
    store = JsonMemoryStore(tmp_path / "memory")
    store.save("agent", {"notes": {"tw2": ["help"]}})

    merged = store.save_patch("agent", MemoryPatch({"notes": {"tw2": ["help", "trade"]}}))

    assert merged == {"notes": {"tw2": ["help", "trade"]}}


def test_memory_store_quarantines_corrupt_campaign_file(tmp_path):
    store = JsonMemoryStore(tmp_path)
    store.save("agent-001", {"durable_facts": ["ok"]})
    campaign = tmp_path / "agent-001" / "campaign.json"
    campaign.write_text('{"durable_facts": ["truncated', encoding="utf-8")

    assert store.load("agent-001") == {}
    assert not campaign.exists()
    assert (tmp_path / "agent-001" / "campaign.json.corrupt").exists()


def test_memory_store_quarantines_valid_json_with_wrong_root_type(tmp_path):
    store = JsonMemoryStore(tmp_path)
    campaign = tmp_path / "agent-001" / "campaign.json"
    campaign.parent.mkdir(parents=True)
    campaign.write_text('["not", "an", "object"]', encoding="utf-8")

    assert store.load("agent-001") == {}
    assert not campaign.exists()
    assert (tmp_path / "agent-001" / "campaign.json.corrupt").exists()
    assert store.save_patch("agent-001", MemoryPatch({"durable_facts": ["recovered"]})) == {
        "durable_facts": ["recovered"]
    }


def test_memory_store_save_leaves_no_temp_file(tmp_path):
    store = JsonMemoryStore(tmp_path)
    store.save("agent-001", {"durable_facts": ["ok"]})

    assert store.load("agent-001") == {"durable_facts": ["ok"]}
    assert not (tmp_path / "agent-001" / ".campaign.tmp").exists()


def test_memory_store_rejects_path_traversal_agent_id(tmp_path):
    import pytest

    store = JsonMemoryStore(tmp_path / "memory")

    for bad_id in ("../escape", "a/b", "..", "a\\b", ""):
        with pytest.raises(ValueError):
            store.load(bad_id)
        with pytest.raises(ValueError):
            store.save(bad_id, {})
    assert not (tmp_path / "escape").exists()


def test_memory_store_save_patch_is_safe_under_concurrency(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = JsonMemoryStore(tmp_path / "memory")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(store.save_patch, "agent-001", MemoryPatch({"durable_facts": [f"fact-{i}"]}))
            for i in range(8)
        ]
        for future in futures:
            future.result()

    facts = store.load("agent-001")["durable_facts"]
    assert sorted(facts) == [f"fact-{i}" for i in range(8)]


def test_memory_store_stale_loader_cannot_quarantine_fresh_save(tmp_path, monkeypatch):
    import json
    import threading

    import tty_agent.memory as memory_module

    store = JsonMemoryStore(tmp_path)
    campaign = tmp_path / "agent-001" / "campaign.json"
    campaign.parent.mkdir(parents=True)
    campaign.write_text('{"durable_facts": ["truncated', encoding="utf-8")

    parse_started = threading.Event()
    resume_parse = threading.Event()
    save_waiting_for_lock = threading.Event()
    original_loads = memory_module.json.loads
    original_flock = memory_module.fcntl.flock

    def delayed_loads(text):
        if threading.current_thread().name == "memory-loader":
            parse_started.set()
            assert resume_parse.wait(timeout=2)
            raise json.JSONDecodeError("stale corrupt input", text, 0)
        return original_loads(text)

    def tracked_flock(handle, operation):
        if threading.current_thread().name == "memory-saver":
            save_waiting_for_lock.set()
        return original_flock(handle, operation)

    monkeypatch.setattr(memory_module.json, "loads", delayed_loads)
    monkeypatch.setattr(memory_module.fcntl, "flock", tracked_flock)

    loader = threading.Thread(target=store.load, args=("agent-001",), name="memory-loader")
    saver = threading.Thread(
        target=store.save,
        args=("agent-001", {"durable_facts": ["fresh"]}),
        name="memory-saver",
    )
    loader.start()
    assert parse_started.wait(timeout=2)
    saver.start()
    assert save_waiting_for_lock.wait(timeout=2)
    resume_parse.set()
    loader.join(timeout=2)
    saver.join(timeout=2)

    assert not loader.is_alive()
    assert not saver.is_alive()
    assert store.load("agent-001") == {"durable_facts": ["fresh"]}
