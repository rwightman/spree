# Repository Guidelines For Coding Agents

`CLAUDE.md` is a symlink to this file; edit this one.

## Build/Test Commands
- Install: `uv sync`
- Editable install without uv: `python -m pip install -e packages/tty-agent -e packages/bbs-gym`
- Run tests: `uv run pytest tests/`
- Run a package of tests: `uv run pytest tests/test_runner.py -v`
- Run a specific test: `uv run pytest tests/test_models.py::test_output_filters_infer_gemma4_from_model_id -v`
- Run tests in parallel: `uv run pytest -n 4 tests/`
- Filter tests: `uv run pytest -k "substring-to-match" tests/`
- Check package builds: `uv build packages/tty-agent --out-dir dist/tty-agent` and `uv build packages/bbs-gym --out-dir dist/bbs-gym`
- CLI smoke: `uv run bbs-gym --help`
- Format after code/docs edits: run `wruff format .` if `wruff` is installed; if unavailable, leave formatting consistent with nearby files

## Code Style Guidelines
- Line length: 120 chars (configured under `[tool.wruff]` in the root `pyproject.toml`)
- Indentation: 4-space hanging indents, arguments should have an extra level of indent, use 'sadface' (closing parenthesis and colon on a separate line)
- Typing: Use PEP484 type annotations in function signatures
- Docstrings: Google style (do not duplicate type annotations and defaults)
- Imports: Standard library first, then third-party, then local
- Function naming: snake_case
- Class naming: PascalCase
- Error handling: Use try/except with specific exceptions
- Conditional expressions: Use parentheses for complex expressions
- Keep reusable terminal-agent code in `packages/tty-agent/src/tty_agent`; keep BBS/Synchronet/TW2-specific code in `packages/bbs-gym/src/bbs_gym`
