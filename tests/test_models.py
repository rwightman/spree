from tty_agent.models import (
    AnthropicAdapter,
    CodexCliAdapter,
    CompactionPrompt,
    DecisionPrompt,
    MemoryCommitPrompt,
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


def test_compact_parses_json_inside_markdown_fence():
    class FencedCompactModel(TextChatAdapter):
        def chat(self, _messages):
            return """```json
{"current_state": "inside TW2", "open_subgoals": ["trade"], "discovered_facts": ["sector 1"], "failed_actions": [], "strategy_notes": [], "last_error": ""}
```"""

    summary = FencedCompactModel().compact(CompactionPrompt("s", "u"))

    assert summary.current_state == "inside TW2"
    assert summary.discovered_facts == ("sector 1",)


def test_commit_memory_parses_json_inside_markdown_fence():
    class FencedCommitModel(TextChatAdapter):
        def chat(self, _messages):
            return """I will return the memory patch:
```json
{"durable_facts": ["sector 57 sells ore"], "open_tasks": ["sell ore"]}
```
"""

    patch = FencedCommitModel().commit_memory(MemoryCommitPrompt("s", "u"))

    assert patch.data == {
        "durable_facts": ["sector 57 sells ore"],
        "open_tasks": ["sell ore"],
    }


def test_decide_filters_reasoning_blocks_but_keeps_raw_response():
    class ThinkingModel(TextChatAdapter):
        def chat(self, _messages):
            return '<think>choose a safe action</think>\n{"action": "wait", "arguments": {}}'

    model = ThinkingModel()
    action = model.decide(DecisionPrompt("s", "u"))

    assert action.action == "wait"
    assert model.last_response == '<think>choose a safe action</think>\n{"action": "wait", "arguments": {}}'
    assert model.last_parsed_response == '{"action": "wait", "arguments": {}}'


def test_compact_does_not_store_truncated_reasoning_as_summary():
    class TruncatedThinkingModel(TextChatAdapter):
        def chat(self, _messages):
            return "<think>still reasoning when output was truncated"

    summary = TruncatedThinkingModel().compact(CompactionPrompt("s", "u"))

    assert summary.is_empty()


def test_gemma4_filter_strips_thought_channel_and_keeps_json():
    response = '<|channel>thought\nchoose a safe action<channel|>\n<|channel>final\n{"action": "wait", "arguments": {}}'

    assert strip_gemma4_channel_reasoning(response).strip() == '{"action": "wait", "arguments": {}}'


def test_gemma4_filter_strips_empty_thought_channel_and_keeps_json():
    response = '<|channel>thought\n<channel|>{"action": "wait", "arguments": {}}'

    assert strip_gemma4_channel_reasoning(response).strip() == '{"action": "wait", "arguments": {}}'


def test_gemma4_filter_strips_unclosed_thought_channel():
    assert strip_gemma4_channel_reasoning("<|channel>thought\nstill reasoning").strip() == ""


def test_output_filters_infer_gemma4_from_model_id():
    filters = output_filters_for_model("google/gemma-4-31B-it")
    response = '<|channel>thought\nchoose action<channel|>\n<|channel>final\n{"action": "wait", "arguments": {}}'
    for output_filter in filters:
        response = output_filter(response)

    assert response.strip() == '{"action": "wait", "arguments": {}}'


def test_output_filters_can_be_disabled():
    assert output_filters_for_model("google/gemma-4-31B-it", "none") == ()


def test_openai_compatible_adapter_merges_extra_body(monkeypatch):
    captured = {}

    def fake_post_json(url, payload, headers, timeout):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": '{"action":"wait","arguments":{}}'}}]}

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
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
                        "content": '<|channel>thought\nchoose action<channel|>\n<|channel>final\n{"action":"wait","arguments":{}}'
                    }
                }
            ]
        }

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(model="google/gemma-4-31B-it")

    action = adapter.decide(DecisionPrompt("s", "u"))

    assert action.action == "wait"
    assert adapter.last_parsed_response == '{"action":"wait","arguments":{}}'


def test_openai_compatible_adapter_captures_reasoning_field(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "reasoning": "choose the low-risk action",
                        "content": '{"action":"wait","arguments":{}}',
                    }
                }
            ]
        }

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(model="google/gemma-4-31B-it")

    action = adapter.decide(DecisionPrompt("s", "u"))

    assert action.action == "wait"
    assert adapter.last_reasoning == "choose the low-risk action"
    assert adapter.last_response == '{"action":"wait","arguments":{}}'


def test_openai_compatible_adapter_captures_reasoning_content_field(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "reasoning field",
                        "content": '{"action":"wait","arguments":{}}',
                    }
                }
            ]
        }

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(model="test-model")

    adapter.decide(DecisionPrompt("s", "u"))

    assert adapter.last_reasoning == "reasoning field"


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
        return {"content": [{"type": "text", "text": '{"action":"wait","arguments":{}}'}]}

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = AnthropicAdapter(model="test-model", api_key="key")

    adapter.chat([ModelMessage("system", "stable schema"), ModelMessage("user", "screen")])

    assert captured["payload"]["system"] == [
        {
            "type": "text",
            "text": "stable schema",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_codex_cli_adapter_invokes_codex_exec(monkeypatch):
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, input, text, capture_output, timeout, cwd, check):
        captured["command"] = command
        captured["input"] = input
        captured["text"] = text
        captured["capture_output"] = capture_output
        captured["timeout"] = timeout
        captured["cwd"] = cwd
        captured["check"] = check
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write('{"action": "wait", "arguments": {}}')
        return Result()

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    adapter = CodexCliAdapter(
        model="gpt-5.5",
        profile="bbs",
        timeout=42.0,
        sandbox="read-only",
        extra_args=["--ignore-rules"],
    )

    action = adapter.decide(DecisionPrompt("system schema", "current screen"))

    assert action.action == "wait"
    assert captured["command"][:2] == ["codex", "exec"]
    assert "--ephemeral" in captured["command"]
    assert captured["command"][captured["command"].index("--model") + 1] == "gpt-5.5"
    assert captured["command"][captured["command"].index("--profile") + 1] == "bbs"
    assert captured["command"][captured["command"].index("--sandbox") + 1] == "read-only"
    assert captured["command"][-2:] == ["--ignore-rules", "-"]
    assert captured["timeout"] == 42.0
    assert "SYSTEM MESSAGE:\nsystem schema" in captured["input"]
    assert "USER MESSAGE:\ncurrent screen" in captured["input"]


def test_codex_cli_adapter_resumes_stateful_session(monkeypatch, tmp_path):
    commands = []
    session_id = "11111111-2222-3333-4444-555555555555"

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout=""):
            self.stdout = stdout

    def fake_run(command, input, text, capture_output, timeout, cwd, check):
        del input, text, capture_output, timeout, cwd, check
        commands.append(command)
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write('{"action": "wait", "arguments": {}}')
        if len(commands) == 1:
            return Result(f'{{"type":"session_configured","session_id":"{session_id}"}}\n')
        return Result()

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    session_file = tmp_path / "codex.session"
    adapter = CodexCliAdapter(model="gpt-5.5", stateful=True, session_file=session_file)

    first = adapter.decide(DecisionPrompt("system schema", "current screen", mode="stateful_delta", stage="bootstrap"))
    second = adapter.decide(DecisionPrompt("delta system", "delta screen", mode="stateful_delta", stage="delta"))

    assert first.action == "wait"
    assert second.action == "wait"
    assert adapter.session_id == session_id
    assert session_file.read_text(encoding="utf-8").strip() == session_id
    assert commands[0][:2] == ["codex", "exec"]
    assert "--ephemeral" not in commands[0]
    assert "--json" in commands[0]
    assert commands[0][-1] == "-"
    assert commands[1][:3] == ["codex", "exec", "resume"]
    assert session_id in commands[1]
    assert commands[1][-1] == "-"


def test_codex_cli_adapter_raises_on_command_failure(monkeypatch):
    class Result:
        returncode = 2
        stdout = "stdout detail"
        stderr = "stderr detail"

    def fake_run(*_args, **_kwargs):
        return Result()

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    adapter = CodexCliAdapter()

    try:
        adapter.chat([ModelMessage("user", "screen")])
    except RuntimeError as exc:
        assert "codex exec failed" in str(exc)
        assert "stderr detail" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
