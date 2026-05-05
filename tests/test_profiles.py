from terminal_agent.profiles import EMPTY_PROFILE, SHELL_PROFILE


def test_empty_profile_never_matches():
    assert EMPTY_PROFILE.match("$") is None


def test_shell_profile_matches_common_prompt_endings():
    assert SHELL_PROFILE.match("user@host:/tmp$") == "shell-prompt"
    assert SHELL_PROFILE.match("root@host:/tmp#") == "shell-prompt"
    assert SHELL_PROFILE.match("sqlite>") == "shell-prompt"
