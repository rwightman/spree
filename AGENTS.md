# AGENTS.md

## Build/Test Commands
- Install: `uv sync`
- Editable install without uv: `python -m pip install -e packages/tty-agent -e packages/bbs-gym`
- Run tests: `uv run pytest tests/`
- Run specific test: `uv run pytest tests/test_models.py::test_specific_function -v`
- Run tests in parallel: `uv run pytest -n 4 tests/`
- Filter tests: `uv run pytest -k "substring-to-match" tests/`
- Format changed code: `wruff format .` if `wruff` is installed

## Code Style Guidelines
- Line length: 120 chars
- Indentation: 4-space hanging indents, arguments should have an extra level of indent, use 'sadface' (closing parenthesis and colon on a separate line)
- Typing: Use PEP484 type annotations in function signatures
- Docstrings: Google style (do not duplicate type annotations and defaults)
- Imports: Standard library first, then third-party, then local
- Function naming: snake_case
- Class naming: PascalCase
- Error handling: Use try/except with specific exceptions
- Conditional expressions: Use parentheses for complex expressions
