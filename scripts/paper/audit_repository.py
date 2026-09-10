"""Audit the explicit paper file set without importing private experiments."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def public_files():
    """Enumerate exactly the files intended for the clean repository."""
    result = [
        ROOT / p
        for p in ("README.md", "pyproject.toml", ".gitignore", ".gitattributes")
    ]
    for folder, suffixes in (
        ("src/active_wishart_tpp", {".py"}),
        ("tests/paper", {".py"}),
        ("scripts/paper", {".py"}),
        ("configs/paper", {".yaml"}),
        ("docs/paper", {".md", ".json"}),
    ):
        result += [
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in suffixes
        ]
    return sorted(result)


def audit():
    """Reject private dependencies, missing local imports, and oversized source files."""
    files = public_files()
    modules = set()
    for path in files:
        if path.suffix == ".py" and path.is_relative_to(ROOT / "src"):
            parts = list(path.relative_to(ROOT / "src").with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            modules.add(".".join(parts))
    maximum = 0
    for path in files:
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        maximum = max(maximum, len(source.splitlines()))
        if len(source.splitlines()) > 500:
            raise ValueError(f"File exceeds 500 lines: {path}")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module]
                    if isinstance(node, ast.ImportFrom) and node.module
                    else []
                )
            )
            for name in names:
                if (
                    name == "wishart_tpp"
                    or name.startswith("wishart_tpp.")
                    or name.startswith(("run_", "audit_"))
                ):
                    raise ValueError(f"Private import in {path}: {name}")
                if name.startswith("active_wishart_tpp") and name not in modules:
                    raise ValueError(f"Missing distributed module in {path}: {name}")
    return dict(
        files=len(files),
        python_files=sum(p.suffix == ".py" for p in files),
        maximum_python_lines=maximum,
        private_imports=0,
        file_sha256={
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in files
        },
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
