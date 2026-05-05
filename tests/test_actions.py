import pytest

from terminal_agent.actions import ActionError, ActionPolicy, parse_action


def test_parse_action_accepts_json_fence():
    action = parse_action('```json\n{"action": "send", "text": "?"}\n```')

    assert action.action == "send"
    assert action.text == "?"


def test_action_policy_rejects_disallowed_action():
    policy = ActionPolicy(allowed_actions=frozenset({"wait"}))

    with pytest.raises(ActionError):
        parse_action('{"action": "send", "text": "?"}', policy)


def test_action_policy_allows_open_ended_multiline_with_limits():
    policy = ActionPolicy(max_lines=2, max_line_chars=20)

    action = parse_action('{"action": "send_multiline", "lines": ["hello", "there"]}', policy)

    assert action.lines == ("hello", "there")


def test_action_policy_rejects_control_chars_by_default():
    with pytest.raises(ActionError):
        parse_action('{"action": "send", "text": "bad\\u0001"}')


def test_action_policy_rejects_text_that_cannot_encode_to_cp437():
    policy = ActionPolicy(require_encoding="cp437")

    with pytest.raises(ActionError):
        parse_action('{"action": "send", "text": "hello 😀"}', policy)


def test_send_raw_rejects_newline_field():
    with pytest.raises(ActionError):
        parse_action('{"action": "send_raw", "text": "x", "newline": true}')
