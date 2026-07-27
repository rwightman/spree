"""Agent identifier validation shared by everything that builds storage paths."""

from __future__ import annotations

MAX_AGENT_ID_LENGTH = 128

_FORBIDDEN_SEGMENTS = {".", ".."}


def validate_agent_id(agent_id: str) -> str:
    """Return ``agent_id`` if it is safe to embed as one filesystem path segment.

    Transcript, memory, and log paths all interpolate the id, so an id that
    contains a separator (or is ``.``/``..``) could escape the configured
    storage root and read or write arbitrary files.
    """

    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    if len(agent_id) > MAX_AGENT_ID_LENGTH:
        raise ValueError(f"agent_id must be at most {MAX_AGENT_ID_LENGTH} characters: {agent_id[:32]!r}...")
    if agent_id in _FORBIDDEN_SEGMENTS:
        raise ValueError(f"agent_id must not be a relative path segment: {agent_id!r}")
    if any(char in "/\\" or ord(char) < 32 or char == "\x7f" for char in agent_id):
        raise ValueError(f"agent_id must not contain path separators or control characters: {agent_id!r}")
    return agent_id
