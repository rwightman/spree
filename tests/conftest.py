from pathlib import Path

import pytest


_SRE_REQUIRED_FILES = (
    "SRE.EXE",
    "DATA/SYSTEM.II",
    "DATA/GALAXY.II",
    "DATA/EMPIRE.II",
)


@pytest.fixture
def sre_source_world() -> Path:
    """Return the local SRE distribution, or skip tests when it is unavailable."""

    source = Path(__file__).resolve().parent.parent / "doors" / "sre"
    missing = [relative for relative in _SRE_REQUIRED_FILES if not (source / relative).is_file()]
    if missing:
        pytest.skip("requires local SRE 0.994b game files; run scripts/fetch_sre.sh")
    return source
