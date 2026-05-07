import ast
from pathlib import Path


def test_tty_agent_core_does_not_import_bbs_gym():
    root = Path("packages/tty-agent/src/tty_agent")
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "bbs_gym" or alias.name.startswith("bbs_gym."):
                        offenders.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "bbs_gym" or node.module.startswith("bbs_gym."):
                    offenders.append((path, node.lineno, node.module))

    assert offenders == []
