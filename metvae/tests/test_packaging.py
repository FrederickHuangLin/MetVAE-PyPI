"""Tests that keep the documentation and the source files consistent with the package."""

import re
from pathlib import Path

import metvae
from metvae.cli import build_parser

_FENCE = re.compile(r"```.*?\n(.*?)```", re.S)
_FLAG = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")


def _package_dir():
    """Return the directory of the installed metvae package."""
    return Path(metvae.__file__).resolve().parent


def test_readme_cli_flags_exist():
    """Every flag shown in a README command-line example is a real parser option."""
    readme = _package_dir().parent / "README.md"
    if not readme.exists():
        import pytest
        pytest.skip("README.md is not available next to the package")

    known = set()
    for action in build_parser()._actions:
        known.update(action.option_strings)

    documented = set()
    for block in _FENCE.findall(readme.read_text(encoding="utf-8")):
        if "metvae-cli" in block:
            documented.update(_FLAG.findall(block))

    unknown = sorted(documented - known)
    assert not unknown, f"README documents flags the parser does not define: {unknown}"


def test_source_is_ascii():
    """Every module of the package decodes as ASCII."""
    offenders = []
    for path in sorted(_package_dir().glob("*.py")):
        raw = path.read_bytes()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as exc:
            offenders.append(f"{path.name}: byte {exc.start} ({raw[exc.start:exc.end]!r})")
    assert not offenders, "non-ASCII bytes found: " + "; ".join(offenders)
