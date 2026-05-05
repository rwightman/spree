import pytest

from terminal_agent.actions import ActionError, ActionPolicy, parse_action, render_action_schema


def test_parse_action_accepts_json_fence():
    action = parse_action('```json\n{"action": "send_line", "text": "?"}\n```')

    assert action.action == "send_line"
    assert action.text == "?"


def test_action_policy_rejects_disallowed_action():
    policy = ActionPolicy(allowed_actions=frozenset({"wait"}))

    with pytest.raises(ActionError):
        parse_action('{"action": "send_line", "text": "?"}', policy)


def test_action_policy_owns_field_combination_validation():
    with pytest.raises(ActionError, match="send_line must not include lines or key"):
        parse_action('{"action": "send_line", "lines": ["unexpected"]}')

    with pytest.raises(ActionError, match="key action must not include text or lines"):
        parse_action('{"action": "key", "key": "enter", "text": "unexpected"}')


def test_action_policy_allows_open_ended_multiline_with_limits():
    policy = ActionPolicy(max_lines=2, max_line_chars=20)

    action = parse_action('{"action": "send_multiline", "lines": ["hello", "there"]}', policy)

    assert action.lines == ("hello", "there")


def test_action_policy_rejects_control_chars_by_default():
    with pytest.raises(ActionError):
        parse_action('{"action": "send_line", "text": "bad\\n"}')


def test_action_policy_rejects_text_that_cannot_encode_to_cp437():
    policy = ActionPolicy(require_encoding="cp437")

    with pytest.raises(ActionError):
        parse_action('{"action": "send_line", "text": "hello 😀"}', policy)


def test_send_raw_rejects_newline_field():
    with pytest.raises(ActionError):
        parse_action('{"action": "send_raw", "text": "x", "newline": true}')


def test_key_action_validates_supported_keys():
    policy = ActionPolicy(allowed_actions=frozenset({"key"}), supported_keys=frozenset({"enter"}))

    assert parse_action('{"action": "key", "key": "enter"}', policy).key == "enter"
    assert parse_action('{"action": "key", "key": "q"}', policy).key == "q"
    with pytest.raises(ActionError):
        parse_action('{"action": "key", "key": "home"}', policy)


def test_send_line_empty_text_is_valid_enter_equivalent():
    action = parse_action('{"action": "send_line", "text": ""}')

    assert action.action == "send_line"
    assert action.text == ""


def test_render_action_schema_only_lists_allowed_actions():
    policy = ActionPolicy(allowed_actions=frozenset({"send_line", "key", "wait"}), supported_keys=frozenset({"enter"}))

    schema = render_action_schema(policy)

    assert '"send_line"' in schema
    assert '"key"' in schema
    assert '"wait"' in schema
    assert '"send_multiline"' not in schema
    assert '"send_raw"' not in schema
    assert "Supported named keys: enter" in schema
