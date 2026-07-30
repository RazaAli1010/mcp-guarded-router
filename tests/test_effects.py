"""SPEC.md 8.1 effect derivation, and the launch-time substitutions of servers.toml.

Neither needs the snapshot: the derivation is a pure function over a name and an annotations
dict, and the config resolution reads only the committed TOML. Everything here runs with no
network and no MCP server (SPEC.md 12).
"""

from __future__ import annotations

import pytest

from mcpr import mcp_client
from mcpr.config import Settings
from mcpr.registry import effect_for, effect_from_parts
from mcpr.types import ToolSpec


def _spec(name: str, annotations: dict | None = None, server: str = "demo") -> ToolSpec:
    """Build a ToolSpec with a placeholder effect, for exercising the derivation."""
    return ToolSpec(
        server=server,
        name=name,
        qualified_name=f"{server}.{name}",
        description="",
        input_schema={"type": "object"},
        annotations=annotations or {},
        effect="read",
    )


# (name, annotations, overrides, expected) - the three tiers of SPEC.md 8.1, first match wins.
CASES = [
    # Tier 1: an explicit override beats everything, including a contradicting annotation.
    ("delete_file", {"destructiveHint": True}, {"demo.delete_file": "read"}, "read"),
    ("search_code", {}, {"demo.search_code": "destructive"}, "destructive"),
    # Tier 2: MCP annotations.
    ("anything", {"destructiveHint": True}, None, "destructive"),
    ("anything", {"readOnlyHint": True}, None, "read"),
    # A False hint is not a True hint: it must fall through to the name heuristic.
    ("search_code", {"destructiveHint": False}, None, "read"),
    # Tier 3: the no-annotations path, which exists because real servers omit them.
    ("search_code", {}, None, "read"),
    ("delete_file", {}, None, "destructive"),
    ("create_or_update_file", {}, None, "write"),
    ("list_directory", {}, None, "read"),
    ("merge_branch", {}, None, "destructive"),
    ("push_files", {}, None, "write"),
]


@pytest.mark.parametrize(("name", "annotations", "overrides", "expected"), CASES)
def test_effect_derivation(
    name: str, annotations: dict, overrides: dict | None, expected: str
) -> None:
    assert effect_for(_spec(name, annotations), overrides) == expected


def test_effect_for_and_effect_from_parts_agree() -> None:
    """snapshot.py calls the parts form before a ToolSpec exists; they must not diverge."""
    spec = _spec("delete_file", {"readOnlyHint": True})
    assert effect_for(spec) == effect_from_parts(
        spec.qualified_name, spec.name, spec.annotations, None
    )


def test_destructive_hint_wins_over_read_only_hint() -> None:
    """SPEC.md 8.1 lists destructiveHint first, so a server setting both gets the safe answer."""
    assert effect_for(_spec("x", {"destructiveHint": True, "readOnlyHint": True})) == "destructive"


def test_annotation_beats_name_heuristic() -> None:
    """A read-sounding name must not override an explicit destructive annotation."""
    assert effect_for(_spec("get_thing", {"destructiveHint": True})) == "destructive"


# --- config/servers.toml resolution -----------------------------------------------------------


def test_sandbox_arg_replaces_the_last_positional_arg() -> None:
    # An explicit Settings keeps every test here hermetic: load_server_configs() would
    # otherwise call load_env(), which re-reads the developer's .env and makes the result
    # depend on whose machine the suite is running on.
    configs = mcp_client.load_server_configs(settings=Settings(sandbox_dir="./sandbox"))

    for server_id in ("git", "filesystem"):
        cfg = configs[server_id]
        assert cfg.sandbox_arg is True
        assert cfg.args[-1].endswith("sandbox")
        assert mcp_client.SANDBOX_TOKEN not in cfg.args
    assert configs["fetch"].sandbox_arg is False


def test_unset_header_var_is_left_as_a_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The surviving ${VAR} is how a missing PAT becomes an actionable reason, not a 401."""
    monkeypatch.delenv("MCPR_GITHUB_PAT", raising=False)
    cfg = mcp_client.load_server_configs(settings=Settings())["github"]

    assert cfg.headers["Authorization"] == "Bearer ${MCPR_GITHUB_PAT}"
    with pytest.raises(mcp_client.McpUnavailable, match="MCPR_GITHUB_PAT"):
        mcp_client._require_reachable("github", cfg)


def test_set_header_var_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCPR_GITHUB_PAT", "ghp_example")
    cfg = mcp_client.load_server_configs(settings=Settings())["github"]

    assert cfg.headers["Authorization"] == "Bearer ghp_example"
    assert cfg.headers["X-MCP-Readonly"] == "true"


def test_minimal_env_carries_no_project_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """A third-party subprocess must never inherit MCPR_* (F1 implementation notes)."""
    monkeypatch.setenv("MCPR_TEACHER_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("MCPR_GITHUB_PAT", "ghp_should_not_leak")

    env = mcp_client._minimal_env()

    assert not [k for k in env if k.startswith("MCPR_")]
    assert "sk-should-not-leak" not in "".join(env.values())
    assert "PATH" in env


def test_package_name_skips_launcher_flags() -> None:
    configs = mcp_client.load_server_configs(settings=Settings())
    assert mcp_client.package_name(configs["git"]) == "mcp-server-git"
    assert mcp_client.package_name(configs["fetch"]) == "mcp-server-fetch"
    assert (
        mcp_client.package_name(configs["filesystem"]) == "@modelcontextprotocol/server-filesystem"
    )


def test_disabled_server_is_reported_not_launched() -> None:
    cfg = mcp_client.load_server_configs(settings=Settings())["fetch"].model_copy(
        update={"enabled": False}
    )
    with pytest.raises(mcp_client.McpUnavailable, match="disabled"):
        mcp_client._require_reachable("fetch", cfg)


def test_missing_executable_gives_an_actionable_reason() -> None:
    """F1 asks for an actionable message rather than a traceback when uv is absent."""
    cfg = mcp_client.load_server_configs(settings=Settings())["fetch"].model_copy(
        update={"command": "definitely-not-installed-xyz"}
    )
    with pytest.raises(mcp_client.McpUnavailable, match="not on PATH"):
        mcp_client._require_reachable("fetch", cfg)
