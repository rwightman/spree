from tty_agent.profiles import EMPTY_PROFILE, SHELL_PROFILE, TEXT_ADVENTURE_PROFILE


def test_empty_profile_never_matches():
    assert EMPTY_PROFILE.match("$") is None


def test_shell_profile_matches_common_prompt_endings():
    assert SHELL_PROFILE.match("user@host:/tmp$") == "shell-prompt"
    assert SHELL_PROFILE.match("root@host:/tmp#") == "shell-prompt"
    assert SHELL_PROFILE.match("sqlite>") == "shell-prompt"


def test_text_adventure_profile_matches_z_machine_prompts():
    assert TEXT_ADVENTURE_PROFILE.match("West of House\n>") == "command-prompt"
    assert TEXT_ADVENTURE_PROFILE.match("Do you wish to leave the game? (Y is affirmative):") == "yes-no-prompt"
    assert TEXT_ADVENTURE_PROFILE.match("[MORE]") == "more-prompt"
