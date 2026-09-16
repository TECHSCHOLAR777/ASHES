"""The spread engine must stay out of process (SRS 2.6.1 / 7.7 / NFR-16)."""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_src_never_imports_spread_service():
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "spread_service" or name.startswith("spread_service."):
                    offenders.append(f"{path}:{name}")
    assert offenders == []
