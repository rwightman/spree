from terminal_agent.models import (
    AnthropicAdapter,
    CompactionPrompt,
    DecisionPrompt,
    ModelMessage,
    OpenAICompatibleAdapter,
    SessionSummary,
    TextChatAdapter,
    output_filters_for_model,
    strip_gemma4_channel_reasoning,
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


def test_gemma4_filter_strips_thought_channel_and_keeps_json():
    response = '<|channel>thought\nchoose a safe action<channel|>\n<|channel>final\n{"action": "wait"}'

    assert strip_gemma4_channel_reasoning(response).strip() == '{"action": "wait"}'


def test_gemma4_filter_strips_empty_thought_channel_and_keeps_json():
    response = '<|channel>thought\n<channel|>{"action": "wait"}'

    assert strip_gemma4_channel_reasoning(response).strip() == '{"action": "wait"}'


def test_gemma4_filter_strips_unclosed_thought_channel():
    assert strip_gemma4_channel_reasoning("<|channel>thought\nstill reasoning").strip() == ""


def test_output_filters_infer_gemma4_from_model_id():
    filters = output_filters_for_model("google/gemma-4-31B-it")
    response = '<|channel>thought\nchoose action<channel|>\n<|channel>final\n{"action": "wait"}'
    for output_filter in filters:
        response = output_filter(response)

    assert response.strip() == '{"action": "wait"}'


def test_output_filters_can_be_disabled():
    assert output_filters_for_model("google/gemma-4-31B-it", "none") == ()


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


def test_openai_compatible_adapter_infers_gemma4_response_filter(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "content": '<|channel>thought\nchoose action<channel|>\n<|channel>final\n{"action":"wait"}'
                    }
                }
            ]
        }

    monkeypatch.setattr("terminal_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(model="google/gemma-4-31B-it")

    action = adapter.decide(DecisionPrompt("s", "u"))

    assert action.action == "wait"
    assert adapter.last_parsed_response == '{"action":"wait"}'


def test_openai_compatible_adapter_captures_reasoning_field(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "reasoning": "choose the low-risk action",
                        "content": '{"action":"wait"}',
                    }
                }
            ]
        }

    monkeypatch.setattr("terminal_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(model="google/gemma-4-31B-it")

    action = adapter.decide(DecisionPrompt("s", "u"))

    assert action.action == "wait"
    assert adapter.last_reasoning == "choose the low-risk action"
    assert adapter.last_response == '{"action":"wait"}'


def test_openai_compatible_adapter_captures_legacy_reasoning_content(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "legacy reasoning field",
                        "content": '{"action":"wait"}',
                    }
                }
            ]
        }

    monkeypatch.setattr("terminal_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(model="test-model")

    adapter.decide(DecisionPrompt("s", "u"))

    assert adapter.last_reasoning == "legacy reasoning field"


def test_openai_compatible_adapter_can_disable_response_filters():
    adapter = OpenAICompatibleAdapter(model="google/gemma-4-31B-it", output_filters=())

    assert adapter.output_filters == ()


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
