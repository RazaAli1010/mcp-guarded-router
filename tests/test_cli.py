"""The CLI surface F0 reserves for later features, and the offline parts of `doctor`.

`doctor` itself is not invoked here: with a configured `.env` it would make network calls,
and SPEC.md 12 requires the suite to run with no network. The individual checks that touch
the network are excluded; everything deterministic is tested directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mcpr import cli

runner = CliRunner()

RESERVED_NAMESPACES = ["snapshot", "data", "guard", "eval", "run", "report", "models"]


def test_help_lists_every_reserved_namespace() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for name in RESERVED_NAMESPACES:
        assert name in result.output, f"`mcpr --help` does not list the {name} namespace"


def test_version_command() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert "mcpr" in result.output


@pytest.mark.parametrize("namespace", RESERVED_NAMESPACES)
def test_reserved_namespaces_are_stubs_not_silent_successes(namespace: str) -> None:
    """A namespace that quietly exits 0 would hide unimplemented work."""
    result = runner.invoke(cli.app, [namespace])
    assert result.exit_code != 0
    assert "Usage" in result.output or "Commands" in result.output


def test_stub_commands_raise_not_implemented() -> None:
    """`snapshot` is F1's and no longer a stub, so this checks a namespace still unbuilt.

    Deliberately not `snapshot refresh`: that command opens MCP connections, and SPEC.md 12
    requires the suite to run with no network and no server listening.
    """
    result = runner.invoke(cli.app, ["data", "synth"])
    assert isinstance(result.exception, NotImplementedError)
    assert "F6" in str(result.exception)


def test_forbidden_check_reports_pass_on_a_clean_environment() -> None:
    """The row acceptance criterion 2 of F0 asks for."""
    assert cli._check_forbidden_importable().status == cli.PASS
    assert cli._check_forbidden_declared().status == cli.PASS


def test_forbidden_lists_cover_the_full_spec_set() -> None:
    assert len(cli.FORBIDDEN_IMPORTS) == 14
    assert len(cli.FORBIDDEN_DISTS) == 14
    assert set(cli.FORBIDDEN_IMPORTS) == {d.replace("-", "_") for d in cli.FORBIDDEN_DISTS}


def test_laptop_deps_are_importable() -> None:
    assert cli._check_laptop_deps().status == cli.PASS


def test_python_version_check_passes_on_the_running_interpreter() -> None:
    assert cli._check_python().status == cli.PASS


def test_sandbox_check_creates_a_missing_directory(tmp_path: Path) -> None:
    """sandbox/ is gitignored, so doctor must not FAIL merely because it is absent."""
    target = tmp_path / "does" / "not" / "exist"
    check = cli._check_sandbox(cli.Settings(sandbox_dir=str(target)))
    assert check.status == cli.PASS
    assert target.is_dir()
    assert not (target / ".mcpr_write_probe").exists()


def test_sandbox_check_fails_on_an_unwritable_path(tmp_path: Path) -> None:
    """A path whose parent is a regular file cannot be created - the portable failure case."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    check = cli._check_sandbox(cli.Settings(sandbox_dir=str(blocker / "sandbox")))
    assert check.status == cli.FAIL


def test_endpoint_check_is_skipped_when_unconfigured() -> None:
    """This is what keeps `mcpr doctor` exit 0 offline with an empty .env."""
    check = cli._check_endpoint("teacher", "", "", "")
    assert check.status == cli.WARN
    assert "skipped" in check.detail


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("torch", "torch"),
        ("torch>=2.0", "torch"),
        ("mcp>=1.29,<2", "mcp"),
        ("transformers[torch]==5.14.1", "transformers"),
        ("pytest ; python_version < '3.13'", "pytest"),
        ("Flash-Attn", "flash-attn"),
    ],
)
def test_requirement_name_strips_specifiers(spec: str, expected: str) -> None:
    assert cli._requirement_name(spec) == expected
