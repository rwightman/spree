from terminal_agent.models import AnthropicAdapter, CompactionPrompt, ModelMessage, SessionSummary, TextChatAdapter


def test_session_summary_parses_structured_mapping():
    summary = SessionSummary.from_mapping(
        {
            "current_state": "at menu",
            "last_error": "invalid command",
            "open_subgoals": ["enter TW2"],
            "discovered_facts": ["X opens external programs"],
            "failed_actions": ["P at main menu"],
            "strategy_notes": ["ask for help when stuck"],
        }
    )

    assert summary.current_state == "at menu"
    assert summary.open_subgoals == ("enter TW2",)
    assert summary.to_dict()["failed_actions"] == ["P at main menu"]


def test_compact_returns_structured_session_summary():
    class CompactModel:
        def chat(self, _messages):
            return '{"current_state": "inside TW2", "open_subgoals": ["trade"], "discovered_facts": [], "failed_actions": [], "strategy_notes": [], "last_error": ""}'

    summary = TextChatAdapter.compact(CompactModel(), CompactionPrompt("s", "u"))

    assert summary.current_state == "inside TW2"
    assert summary.open_subgoals == ("trade",)


def test_anthropic_adapter_uses_cache_control_for_system_prompt(monkeypatch):
    captured = {}

    def fake_post_json(url, payload, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {"content": [{"type": "text", "text": '{"action":"wait"}'}]}

    monkeypatch.setattr("terminal_agent.models._post_json", fake_post_json)
    adapter = AnthropicAdapter(model="test-model", api_key="key")

    adapter.chat([ModelMessage("system", "stable schema"), ModelMessage("user", "screen")])

    assert captured["payload"]["system"] == [
        {
            "type": "text",
            "text": "stable schema",
            "cache_control": {"type": "ephemeral"},
        }
    ]
