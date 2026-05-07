from tty_agent.memory import JsonMemoryStore
from tty_agent.models import MemoryPatch


def test_memory_store_dedupes_and_caps_lists(tmp_path):
    store = JsonMemoryStore(tmp_path / "memory", max_list_items=3)
    store.save("agent", {"durable_facts": ["old", "repeat"]})

    merged = store.save_patch("agent", MemoryPatch({"durable_facts": ["repeat", "new1", "new2", "new3"]}))

    assert merged == {"durable_facts": ["new1", "new2", "new3"]}


def test_memory_store_merges_nested_dicts(tmp_path):
    store = JsonMemoryStore(tmp_path / "memory")
    store.save("agent", {"notes": {"tw2": ["help"]}})

    merged = store.save_patch("agent", MemoryPatch({"notes": {"tw2": ["help", "trade"]}}))

    assert merged == {"notes": {"tw2": ["help", "trade"]}}
