import io
import json
import subprocess
import urllib.error

import pytest

from tty_agent.models import (
    GEMMA4_OUTPUT_FILTERS,
    AnthropicAdapter,
    ClaudeCliAdapter,
    CodexCliAdapter,
    CompactionPrompt,
    DecisionPrompt,
    MemoryCommitPrompt,
    ModelError,
    ModelMessage,
    ModelStateError,
    ModelTimeoutError,
    OpenAICompatibleAdapter,
    ResponsesCompatibleAdapter,
    SessionSummary,
    TextChatAdapter,
    _is_missing_response_state_error,
    _post_json,
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


def test_compact_rejects_truncated_reasoning_without_visible_summary():
    class TruncatedThinkingModel(TextChatAdapter):
        def chat(self, _messages):
            self.last_reasoning = "still reasoning when output was truncated"
            self.last_provider_metadata = {"finish_reason": "length"}
            return "<think>still reasoning when output was truncated"

    with pytest.raises(ModelError, match="truncated"):
        TruncatedThinkingModel().compact(CompactionPrompt("s", "u"))


def test_memory_commit_rejects_empty_content():
    class EmptyMemoryModel(TextChatAdapter):
        def chat(self, _messages):
            return ""

    with pytest.raises(ModelError, match="no usable content"):
        EmptyMemoryModel().commit_memory(MemoryCommitPrompt("s", "u"))


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


def test_openai_compatible_adapter_can_override_reasoning_for_utility_calls(monkeypatch):
    payloads = []
    responses = iter(
        [
            '{"action":"wait","arguments":{}}',
            (
                '{"current_state":"at prompt","last_error":"","open_subgoals":[],'
                '"discovered_facts":[],"failed_actions":[],"strategy_notes":[]}'
            ),
            '{"durable_facts":["at prompt"]}',
        ]
    )

    def fake_post_json(url, payload, headers, timeout):
        payloads.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": next(responses)},
                }
            ]
        }

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(
        model="test-model",
        max_tokens=2048,
        extra_body={
            "reasoning": {"enabled": True, "effort": "medium"},
            "response_format": {"type": "json_object"},
        },
        compaction_reasoning=False,
        compaction_extra_body={"utility_operation": "compact"},
        memory_reasoning=False,
        memory_extra_body={"utility_operation": "memory"},
    )

    adapter.decide(DecisionPrompt("s", "u"))
    adapter.compact(CompactionPrompt("s", "u"))
    adapter.commit_memory(MemoryCommitPrompt("s", "u"))

    assert [payload["max_tokens"] for payload in payloads] == [2048, 2048, 2048]
    assert payloads[0]["reasoning"] == {"enabled": True, "effort": "medium"}
    assert payloads[1]["reasoning"] == {"enabled": False, "effort": "medium"}
    assert payloads[2]["reasoning"] == {"enabled": False, "effort": "medium"}
    assert payloads[1]["utility_operation"] == "compact"
    assert payloads[2]["utility_operation"] == "memory"
    assert all(payload["response_format"] == {"type": "json_object"} for payload in payloads)


def test_openai_compatible_adapter_merges_extra_headers_without_allowing_credential_override(monkeypatch):
    captured = {}

    def fake_post_json(url, payload, headers, timeout):
        captured["headers"] = headers
        return {"choices": [{"message": {"content": '{"action":"wait","arguments":{}}'}}]}

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(
        model="test-model",
        api_key="real-key",
        extra_headers={
            "Authorization": "Bearer wrong-key",
            "x-session-affinity": "agent-session",
        },
    )

    adapter.chat([ModelMessage("user", "screen")])

    assert captured["headers"] == {
        "Authorization": "Bearer real-key",
        "x-session-affinity": "agent-session",
    }


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
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "reasoning": "choose the low-risk action",
                        "content": '{"action":"wait","arguments":{}}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "perf_metrics": {"cached-prompt-tokens": 8},
        }

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(model="google/gemma-4-31B-it")

    action = adapter.decide(DecisionPrompt("s", "u"))

    assert action.action == "wait"
    assert adapter.last_reasoning == "choose the low-risk action"
    assert adapter.last_response == '{"action":"wait","arguments":{}}'
    assert adapter.last_response_id == "chatcmpl-test"
    assert adapter.last_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert adapter.last_provider_metadata == {
        "finish_reason": "stop",
        "perf_metrics": {"cached-prompt-tokens": 8},
    }


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


def _responses_result(
        text: str,
        response_id: str | None,
        *,
        reasoning: str = "",
) -> dict[str, object]:
    output: list[dict[str, object]] = []
    if reasoning:
        output.append(
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    output.append(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
    )
    return {
        "id": response_id,
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    }


def test_responses_adapter_stateless_request_and_typed_output(monkeypatch):
    captured = {}

    def fake_post_json(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return _responses_result(
            '{"action":"wait","arguments":{}}',
            None,
            reasoning="wait for more terminal output",
        )

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = ResponsesCompatibleAdapter(
        model="test-model",
        base_url="https://example.test/v1/",
        api_key="real-key",
        max_tokens=128,
        extra_headers={"Authorization": "Bearer wrong-key", "x-route": "agent-1"},
    )

    action = adapter.decide(DecisionPrompt("stable instructions", "current screen"))

    assert action.action == "wait"
    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["payload"] == {
        "temperature": 0.2,
        "model": "test-model",
        "input": "current screen",
        "max_output_tokens": 128,
        "store": False,
        "instructions": "stable instructions",
    }
    assert captured["headers"] == {
        "Authorization": "Bearer real-key",
        "x-route": "agent-1",
    }
    assert adapter.last_reasoning == "wait for more terminal output"
    assert adapter.last_usage == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}


def test_responses_adapter_stateful_decisions_chain_but_social_does_not(monkeypatch, tmp_path):
    payloads = []
    results = iter(
        [
            _responses_result('{"action":"wait","arguments":{}}', "resp-bootstrap"),
            _responses_result('{"action":"wait","arguments":{}}', "resp-delta"),
            _responses_result('{"action":"wait","arguments":{}}', None),
        ]
    )

    def fake_post_json(url, payload, headers, timeout):
        payloads.append(payload)
        return next(results)

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    state_file = tmp_path / "responses-state.json"
    adapter = ResponsesCompatibleAdapter(
        model="test-model",
        stateful=True,
        state_file=state_file,
    )

    adapter.decide(DecisionPrompt("bootstrap system", "screen 1", mode="stateful_delta", stage="bootstrap"))
    adapter.decide(DecisionPrompt("delta system", "screen 2", mode="stateful_delta", stage="delta"))
    adapter.decide(DecisionPrompt("forum system", "forum", mode="stateless_full", stage="campaign_social"))

    assert payloads[0]["store"] is True
    assert "previous_response_id" not in payloads[0]
    assert payloads[1]["previous_response_id"] == "resp-bootstrap"
    assert payloads[1]["instructions"] == "bootstrap system\n\ndelta system"
    assert payloads[2]["store"] is False
    assert "previous_response_id" not in payloads[2]
    assert adapter.response_id == "resp-delta"
    assert json.loads(state_file.read_text(encoding="utf-8"))["response_id"] == "resp-delta"


def test_responses_adapter_new_bootstrap_starts_fresh_chain(monkeypatch):
    payloads = []
    results = iter(
        [
            _responses_result('{"action":"wait","arguments":{}}', "resp-1"),
            _responses_result('{"action":"wait","arguments":{}}', "resp-2"),
            _responses_result('{"action":"wait","arguments":{}}', "resp-3"),
        ]
    )

    def fake_post_json(url, payload, headers, timeout):
        payloads.append(dict(payload))
        return next(results)

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = ResponsesCompatibleAdapter(
        model="test-model",
        stateful=True,
        response_id="resp-restored",
    )

    adapter.decide(DecisionPrompt("bootstrap 1", "screen 1", mode="stateful_delta", stage="bootstrap"))
    adapter.decide(DecisionPrompt("delta", "screen 2", mode="stateful_delta", stage="delta"))
    adapter.decide(DecisionPrompt("bootstrap 2", "screen 3", mode="stateful_delta", stage="bootstrap"))

    assert payloads[0]["previous_response_id"] == "resp-restored"
    assert payloads[1]["previous_response_id"] == "resp-1"
    assert "previous_response_id" not in payloads[2]
    assert adapter.response_id == "resp-3"


def test_responses_adapter_resumes_state_file_only_when_requested(monkeypatch, tmp_path):
    state_file = tmp_path / "responses-state.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_url": "http://localhost:11434/v1",
                "model": "test-model",
                "response_id": "resp-restored",
            }
        ),
        encoding="utf-8",
    )
    payloads = []

    def fake_post_json(url, payload, headers, timeout):
        payloads.append(dict(payload))
        return _responses_result('{"action":"wait","arguments":{}}', f"resp-{len(payloads)}")

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    fresh = ResponsesCompatibleAdapter(model="test-model", stateful=True, state_file=state_file)
    fresh.decide(DecisionPrompt("fresh", "screen", mode="stateful_delta", stage="bootstrap"))

    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_url": "http://localhost:11434/v1",
                "model": "test-model",
                "response_id": "resp-restored",
            }
        ),
        encoding="utf-8",
    )
    resumed = ResponsesCompatibleAdapter(model="test-model", stateful=True, state_file=state_file, resume=True)
    resumed.decide(DecisionPrompt("resume", "screen", mode="stateful_delta", stage="bootstrap"))

    assert "previous_response_id" not in payloads[0]
    assert payloads[1]["previous_response_id"] == "resp-restored"


def test_responses_adapter_restarts_stale_explicit_resume_for_full_bootstrap(monkeypatch):
    payloads = []

    def fake_post_json(url, payload, headers, timeout):
        payloads.append(dict(payload))
        if len(payloads) == 1:
            raise ModelError(
                "HTTP 404: previous_response_id was not found",
                stderr="previous_response_id was not found",
                status_code=404,
            )
        return _responses_result('{"action":"wait","arguments":{}}', "resp-new")

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = ResponsesCompatibleAdapter(
        model="test-model",
        stateful=True,
        response_id="resp-stale",
    )

    action = adapter.decide(
        DecisionPrompt("full system", "full context", mode="stateful_delta", stage="bootstrap")
    )

    assert action.action == "wait"
    assert payloads[0]["previous_response_id"] == "resp-stale"
    assert "previous_response_id" not in payloads[1]
    assert adapter.response_id == "resp-new"


def test_responses_adapter_requires_new_bootstrap_when_delta_state_is_missing():
    adapter = ResponsesCompatibleAdapter(model="test-model", stateful=True)

    with pytest.raises(ModelStateError, match="bootstrap"):
        adapter.decide(DecisionPrompt("delta system", "screen", mode="stateful_delta", stage="delta"))


def test_responses_adapter_rejects_state_file_for_another_model(tmp_path):
    state_file = tmp_path / "responses-state.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_url": "http://localhost:11434/v1",
                "model": "another-model",
                "response_id": "resp-old",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        ResponsesCompatibleAdapter(model="test-model", stateful=True, state_file=state_file, resume=True)


def test_responses_adapter_does_not_treat_model_not_found_as_missing_state():
    error = ModelError(
        "HTTP 404 from https://example.test/v1/responses: model was not found",
        stderr='{"error":{"message":"The model gpt-9 was not found or is invalid"}}',
        status_code=404,
    )

    assert _is_missing_response_state_error(error) is False
    assert (
        _is_missing_response_state_error(
            ModelError(
                "HTTP 404 from https://example.test/v1/responses",
                stderr='{"error":{"message":"previous_response_id was not found"}}',
                status_code=404,
            )
        )
        is True
    )


def test_responses_adapter_uses_reasoning_effort_for_utility_calls(monkeypatch):
    payloads = []
    results = iter(
        [
            _responses_result(
                '{"current_state":"ready","open_subgoals":[],"discovered_facts":[],"failed_actions":[],"strategy_notes":[]}',
                None,
            ),
            _responses_result('{"durable_facts":["ready"]}', None),
        ]
    )

    def fake_post_json(url, payload, headers, timeout):
        payloads.append(dict(payload))
        return next(results)

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = ResponsesCompatibleAdapter(
        model="test-model",
        extra_body={"reasoning": {"effort": "high"}},
        compaction_reasoning=True,
        memory_reasoning=False,
    )

    adapter.compact(CompactionPrompt("s", "u"))
    adapter.commit_memory(MemoryCommitPrompt("s", "u"))

    assert payloads[0]["reasoning"] == {"effort": "high"}
    assert payloads[1]["reasoning"] == {"effort": "none"}


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


def test_anthropic_adapter_infers_gemma4_response_filter():
    adapter = AnthropicAdapter(model="google/gemma-4-31B-it", api_key="key")

    assert adapter.output_filters == GEMMA4_OUTPUT_FILTERS


def test_anthropic_adapter_can_disable_response_filters():
    adapter = AnthropicAdapter(model="google/gemma-4-31B-it", api_key="key", output_filters=())

    assert adapter.output_filters == ()


def test_post_json_wraps_http_error_as_model_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "overloaded", {}, io.BytesIO(b"server busy"))

    monkeypatch.setattr("tty_agent.models.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ModelError) as excinfo:
        _post_json("http://localhost:11434/v1/chat/completions", {})

    assert "HTTP 503" in str(excinfo.value)
    assert "server busy" in str(excinfo.value)
    assert excinfo.value.status_code == 503


def test_post_json_wraps_unreachable_host_as_model_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr("tty_agent.models.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ModelError):
        _post_json("http://localhost:11434/v1/chat/completions", {})


def test_post_json_wraps_read_timeout_as_model_timeout(monkeypatch):
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("tty_agent.models.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ModelTimeoutError):
        _post_json("http://localhost:11434/v1/chat/completions", {}, timeout=5.0)


def test_post_json_wraps_non_json_body_as_model_error(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return b"<html>502 Bad Gateway</html>"

    monkeypatch.setattr("tty_agent.models.urllib.request.urlopen", lambda request, timeout: FakeResponse())

    with pytest.raises(ModelError):
        _post_json("http://localhost:11434/v1/chat/completions", {})


def test_openai_compatible_adapter_wraps_malformed_response_as_model_error(monkeypatch):
    monkeypatch.setattr("tty_agent.models._post_json", lambda *a, **k: {"unexpected": True})
    adapter = OpenAICompatibleAdapter(model="test-model")

    with pytest.raises(ModelError):
        adapter.chat([ModelMessage("user", "screen")])


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


def test_claude_cli_adapter_invokes_claude_print(monkeypatch):
    captured = {}
    session_id = "11111111-2222-3333-4444-555555555555"

    class Result:
        returncode = 0
        stderr = ""
        stdout = f'{{"session_id":"{session_id}","result":"{{\\"action\\": \\"wait\\", \\"arguments\\": {{}}}}"}}'

    def fake_run(command, input, text, capture_output, timeout, cwd, check):
        captured["command"] = command
        captured["input"] = input
        captured["text"] = text
        captured["capture_output"] = capture_output
        captured["timeout"] = timeout
        captured["cwd"] = cwd
        captured["check"] = check
        return Result()

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    adapter = ClaudeCliAdapter(
        model="claude-sonnet-4-6",
        timeout=42.0,
        extra_args=["--debug"],
    )

    action = adapter.decide(DecisionPrompt("system schema", "current screen"))

    assert action.action == "wait"
    assert captured["command"][:5] == ["claude", "-p", "--output-format", "json", "--input-format"]
    assert "text" in captured["command"]
    assert "--no-session-persistence" in captured["command"]
    assert "--bare" not in captured["command"]
    assert captured["command"][captured["command"].index("--model") + 1] == "claude-sonnet-4-6"
    assert captured["command"][captured["command"].index("--permission-mode") + 1] == "dontAsk"
    # Tool isolation must be explicit: the default disables the CLI's own tools.
    assert captured["command"][captured["command"].index("--tools") + 1] == ""
    assert captured["command"][-1] == "--debug"
    assert captured["timeout"] == 42.0
    assert "SYSTEM MESSAGE:\nsystem schema" in captured["input"]
    assert "USER MESSAGE:\ncurrent screen" in captured["input"]


def test_claude_cli_adapter_includes_non_empty_tools(monkeypatch):
    captured = {}

    class Result:
        returncode = 0
        stderr = ""
        stdout = '{"result":"{\\"action\\": \\"wait\\", \\"arguments\\": {}}"}'

    def fake_run(command, *_args, **_kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    adapter = ClaudeCliAdapter(tools="Bash,Read")

    adapter.decide(DecisionPrompt("system schema", "current screen"))

    assert captured["command"][captured["command"].index("--tools") + 1] == "Bash,Read"


def test_claude_cli_adapter_resumes_stateful_session(monkeypatch, tmp_path):
    commands = []
    session_id = "11111111-2222-3333-4444-555555555555"

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(command, input, text, capture_output, timeout, cwd, check):
        del input, text, capture_output, timeout, cwd, check
        commands.append(command)
        return Result(f'{{"session_id":"{session_id}","result":"{{\\"action\\": \\"wait\\", \\"arguments\\": {{}}}}"}}')

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    session_file = tmp_path / "claude.session"
    adapter = ClaudeCliAdapter(model="claude-sonnet-4-6", stateful=True, session_file=session_file)

    first = adapter.decide(DecisionPrompt("system schema", "current screen", mode="stateful_delta", stage="bootstrap"))
    second = adapter.decide(DecisionPrompt("delta system", "delta screen", mode="stateful_delta", stage="delta"))

    assert first.action == "wait"
    assert second.action == "wait"
    assert adapter.session_id == session_id
    assert session_file.read_text(encoding="utf-8").strip() == session_id
    assert "--no-session-persistence" not in commands[0]
    assert "--resume" not in commands[0]
    assert commands[1][commands[1].index("--resume") + 1] == session_id


def test_claude_cli_adapter_raises_on_command_failure(monkeypatch):
    class Result:
        returncode = 2
        stdout = "stdout detail"
        stderr = "stderr detail"

    def fake_run(*_args, **_kwargs):
        return Result()

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    adapter = ClaudeCliAdapter()

    try:
        adapter.chat([ModelMessage("user", "screen")])
    except RuntimeError as exc:
        assert "claude -p failed" in str(exc)
        assert "stderr detail" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_claude_cli_adapter_timeout_includes_stdout_and_stderr(monkeypatch):
    def fake_run(command, input, text, capture_output, timeout, cwd, check):
        del input, text, capture_output, cwd, check
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    adapter = ClaudeCliAdapter(timeout=12.0)

    try:
        adapter.chat([ModelMessage("user", "screen")])
    except ModelTimeoutError as exc:
        assert "claude -p timed out after 12s" in str(exc)
        assert "partial stdout" in str(exc)
        assert "partial stderr" in str(exc)
        assert exc.stdout == "partial stdout"
        assert exc.stderr == "partial stderr"
    else:
        raise AssertionError("expected ModelTimeoutError")


def test_claude_cli_adapter_returns_empty_result_as_empty(monkeypatch):
    class Result:
        returncode = 0
        stderr = ""
        stdout = '{"type":"result","session_id":"11111111-2222-3333-4444-555555555555","result":"","usage":{}}'

    monkeypatch.setattr("tty_agent.models.subprocess.run", lambda *_args, **_kwargs: Result())
    adapter = ClaudeCliAdapter()

    # An empty completion must not fall back to the raw JSON envelope, which
    # would otherwise be merged into campaign memory by commit_memory.
    assert adapter.chat([ModelMessage("user", "screen")]) == ""


def test_codex_cli_adapter_stateful_missing_output_returns_empty(monkeypatch, tmp_path):
    session_id = "11111111-2222-3333-4444-555555555555"

    class Result:
        returncode = 0
        stderr = ""
        stdout = f'{{"type":"session_configured","session_id":"{session_id}"}}\n'

    def fake_run(command, input, text, capture_output, timeout, cwd, check):
        del command, input, text, capture_output, timeout, cwd, check
        return Result()

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    adapter = CodexCliAdapter(stateful=True, session_file=tmp_path / "codex.session")

    # --json stdout is an event stream, never a chat message.
    assert adapter.chat([ModelMessage("user", "screen")]) == ""


def test_claude_cli_adapter_follows_forked_session_ids(monkeypatch, tmp_path):
    session_ids = iter(
        [
            "11111111-2222-3333-4444-555555555555",
            "66666666-7777-8888-9999-000000000000",
        ]
    )

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(*_args, **_kwargs):
        return Result(f'{{"session_id":"{next(session_ids)}","result":"ok"}}')

    monkeypatch.setattr("tty_agent.models.subprocess.run", fake_run)
    session_file = tmp_path / "claude.session"
    adapter = ClaudeCliAdapter(stateful=True, session_file=session_file)

    adapter.chat([ModelMessage("user", "one")])
    adapter.chat([ModelMessage("user", "two")])

    # A CLI that forks a session on resume reports the new id; resume that one.
    assert adapter.session_id == "66666666-7777-8888-9999-000000000000"
    assert session_file.read_text(encoding="utf-8").strip() == "66666666-7777-8888-9999-000000000000"


def test_openai_compatible_adapter_rejects_non_string_content(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        del url, payload, headers, timeout
        return {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]}

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = OpenAICompatibleAdapter(model="test-model")

    with pytest.raises(ModelError, match="unexpected OpenAI-compatible response"):
        adapter.chat([ModelMessage("user", "screen")])


def test_anthropic_adapter_rejects_response_without_text_blocks(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        del url, payload, headers, timeout
        return {"content": [{"type": "thinking", "thinking": "hmm"}]}

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = AnthropicAdapter(model="claude-sonnet-5", api_key="test")

    with pytest.raises(ModelError, match="no text blocks"):
        adapter.chat([ModelMessage("user", "screen")])


def test_anthropic_adapter_rejects_malformed_text_block(monkeypatch):
    def fake_post_json(url, payload, headers, timeout):
        del url, payload, headers, timeout
        return {"content": [{"type": "text", "text": None}]}

    monkeypatch.setattr("tty_agent.models._post_json", fake_post_json)
    adapter = AnthropicAdapter(model="claude-sonnet-5", api_key="test")

    with pytest.raises(ModelError, match="unexpected Anthropic response"):
        adapter.chat([ModelMessage("user", "screen")])
