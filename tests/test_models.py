from terminal_agent.models import (
    AnthropicAdapter,
    CompactionPrompt,
    DecisionPrompt,
    ModelMessage,
    OpenAICompatibleAdapter,
    SessionSummary,
    TextChatAdapter,
)


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
    class CompactModel(TextChatAdapter):
        def chat(self, _messages):
            return (
                '{"current_state": "inside TW2", "open_subgoals": ["trade"], '
                '"discovered_facts": [], "failed_actions": [], "strategy_notes": [], "last_error": ""}'
            )

    summary = CompactModel().compact(CompactionPrompt("s", "u"))

    assert summary.current_state == "inside TW2"
    assert summary.open_subgoals == ("trade",)


def test_decide_filters_reasoning_blocks_but_keeps_raw_response():
    class ThinkingModel(TextChatAdapter):
        def chat(self, _messages):
            return '<think>choose a safe action</think>\n{"action": "wait"}'

    model = ThinkingModel()
    action = model.decide(DecisionPrompt("s", "u"))

    assert action.action == "wait"
    assert model.last_response == '<think>choose a safe action</think>\n{"action": "wait"}'
    assert model.last_parsed_response == '{"action": "wait"}'


def test_compact_does_not_store_truncated_reasoning_as_summary():
    class TruncatedThinkingModel(TextChatAdapter):
        def chat(self, _messages):
            return "<think>still reasoning when output was truncated"

    summary = TruncatedThinkingModel().compact(CompactionPrompt("s", "u"))

    assert summary.is_empty()


def test_openai_compatible_adapter_merges_extra_body(monkeypatch):
    captured = {}

    def fake_post_json(url, payload, headers, timeout):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": '{"action":"wait"}'}}]}

    monkeypatch.setattr("terminal_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(
        model="test-model",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    adapter.chat([ModelMessage("user", "screen")])

    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


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
