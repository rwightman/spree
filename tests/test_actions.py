import pytest

from terminal_agent.actions import ActionError, ActionPolicy, parse_action, render_action_schema


def test_parse_action_accepts_json_fence():
    action = parse_action('```json\n{"action": "submit_line", "arguments": {"text": "?"}}\n```')

    assert action.action == "submit_line"
    assert action.text == "?"


def test_action_policy_rejects_disallowed_action():
    policy = ActionPolicy(allowed_actions=frozenset({"wait"}))

    with pytest.raises(ActionError):
        parse_action('{"action": "submit_line", "arguments": {"text": "?"}}', policy)


def test_action_policy_owns_field_combination_validation():
    with pytest.raises(ActionError, match="submit_line must not include lines or key"):
        parse_action('{"action": "submit_line", "arguments": {"lines": ["unexpected"]}}')

    with pytest.raises(ActionError, match="press_key must not include text or lines"):
        parse_action('{"action": "press_key", "arguments": {"key": "enter", "text": "unexpected"}}')


def test_action_policy_allows_open_ended_multiline_with_limits():
    policy = ActionPolicy(max_lines=2, max_line_chars=20)

    action = parse_action('{"action": "submit_lines", "arguments": {"lines": ["hello", "there"]}}', policy)

    assert action.lines == ("hello", "there")


def test_action_policy_rejects_control_chars_by_default():
    with pytest.raises(ActionError):
        parse_action('{"action": "submit_line", "arguments": {"text": "bad\\n"}}')


def test_action_policy_rejects_text_that_cannot_encode_to_cp437():
    policy = ActionPolicy(require_encoding="cp437")

    with pytest.raises(ActionError):
        parse_action('{"action": "submit_line", "arguments": {"text": "hello 😀"}}', policy)


def test_send_raw_rejects_newline_field():
    with pytest.raises(ActionError):
        parse_action('{"action": "send_raw", "arguments": {"text": "x", "newline": true}}')


def test_key_action_validates_supported_keys():
    policy = ActionPolicy(allowed_actions=frozenset({"press_key"}), supported_keys=frozenset({"enter"}))

    assert parse_action('{"action": "press_key", "arguments": {"key": "enter"}}', policy).key == "enter"
    assert parse_action('{"action": "press_key", "arguments": {"key": "q"}}', policy).key == "q"
    with pytest.raises(ActionError):
        parse_action('{"action": "press_key", "arguments": {"key": "home"}}', policy)


def test_send_line_empty_text_is_valid_enter_equivalent():
    action = parse_action('{"action": "submit_line", "arguments": {"text": ""}}')

    assert action.action == "submit_line"
    assert action.text == ""


def test_render_action_schema_only_lists_allowed_actions():
    policy = ActionPolicy(
        allowed_actions=frozenset({"submit_line", "press_key", "wait"}),
        supported_keys=frozenset({"enter"}),
    )

    schema = render_action_schema(policy)

    assert '"submit_line"' in schema
    assert '"press_key"' in schema
    assert '"wait"' in schema
    assert '"submit_lines"' not in schema
    assert '"send_raw"' not in schema
    assert "Supported named keys: enter" in schema


def test_render_action_schema_describes_type_text_as_primary_when_submit_line_is_unavailable():
    policy = ActionPolicy(
        allowed_actions=frozenset({"type_text", "press_key", "wait"}),
        supported_keys=frozenset({"enter"}),
    )

    schema = render_action_schema(policy)

    assert '"submit_line"' not in schema
    assert "Some programs accept typed values immediately" in schema
    assert "Most prompts want submit_line" not in schema


def test_action_to_dict_emits_action_arguments_shape():
    action = parse_action('{"action": "submit_line", "arguments": {"text": ""}}')

    assert action.to_dict() == {"action": "submit_line", "arguments": {"text": ""}}


def test_parse_action_rejects_tool_shape():
    with pytest.raises(ActionError, match="tool is not supported"):
        parse_action('{"tool": "press_key", "arguments": {"key": "enter"}}')
