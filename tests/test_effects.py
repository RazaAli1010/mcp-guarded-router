"""SPEC.md 8.1 effect derivation, and the launch-time substitutions of servers.toml.

Neither needs the snapshot: the derivation is a pure function over a name and an annotations
dict, and the config resolution reads only the committed TOML. Everything here runs with no
network and no MCP server (SPEC.md 12).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcpr import mcp_client
from mcpr.config import Settings
from mcpr.registry import (
    DEFAULT_SANDBOXED_SERVERS,
    POLICY_SECTIONS,
    PolicyError,
    effect_for,
    effect_from_parts,
    effect_with_source,
    load_policy,
    load_registry,
)
from mcpr.types import Registry, ToolSpec

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURES / "registry_min.json"


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


# --- derivation provenance (SPEC.md 8.1, F3) ---------------------------------------------------

# (name, annotations, overrides, expected effect, expected source)
SOURCE_CASES = [
    ("delete_file", {"destructiveHint": True}, {"demo.delete_file": "read"}, "read", "override"),
    ("anything", {"destructiveHint": True}, None, "destructive", "annotation"),
    ("anything", {"readOnlyHint": True}, None, "read", "annotation"),
    # A False hint decides nothing, so the source is the tier that actually did.
    ("search_code", {"destructiveHint": False}, None, "read", "heuristic"),
    ("delete_file", {}, None, "destructive", "heuristic"),
    ("frobnicate", {}, None, "write", "heuristic"),
]


@pytest.mark.parametrize(
    ("name", "annotations", "overrides", "effect", "source"),
    SOURCE_CASES,
)
def test_effect_with_source_attributes_the_deciding_tier(
    name: str, annotations: dict, overrides: dict | None, effect: str, source: str
) -> None:
    assert effect_with_source(_spec(name, annotations), overrides) == (effect, source)


@pytest.mark.parametrize(("name", "annotations", "overrides", "expected"), CASES)
def test_effect_with_source_agrees_with_effect_for(
    name: str, annotations: dict, overrides: dict | None, expected: str
) -> None:
    """The two entry points must not drift: one is baked into the snapshot, one runs in the guard.

    `snapshot.py` bakes `ToolSpec.effect` via `effect_from_parts` at capture time while the guards
    call the runtime derivation, so a divergence here would mean the frozen registry and the live
    policy disagreed about which tools are destructive.
    """
    spec = _spec(name, annotations)
    assert effect_with_source(spec, overrides)[0] == effect_for(spec, overrides) == expected


# --- load_policy (SPEC.md 8, F3) ---------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry(FIXTURE_PATH)


def _policy_file(tmp_path: Path, body: str) -> Path:
    """Write a policy file at a path unique to the test, so the TOML lru_cache cannot leak."""
    path = tmp_path / "policy.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_policy_returns_every_section(tmp_path: Path, registry: Registry) -> None:
    """Callers index straight into a section; a missing table must not become a KeyError."""
    policy = load_policy(_policy_file(tmp_path, "[effects]\n"), Settings(), registry)
    assert set(POLICY_SECTIONS) <= set(policy)
    assert policy["rules"] == {} and policy["limits"] == {}


def test_load_policy_overlays_the_resolved_sandbox_dir(tmp_path: Path, registry: Registry) -> None:
    """The guard reads an absolute root and never an env var, which is what keeps it pure."""
    policy = load_policy(
        _policy_file(tmp_path, '[paths]\nsandbox_root = "./ignored"\n'),
        Settings(sandbox_dir="./sandbox"),
        registry,
    )
    root = Path(policy["paths"]["sandbox_root"])
    assert root.is_absolute()
    assert root.name == "sandbox"


def test_load_policy_defaults_the_sandboxed_servers(tmp_path: Path, registry: Registry) -> None:
    """SPEC.md 8.3 as amended: containment covers the servers that touch the local disk."""
    policy = load_policy(_policy_file(tmp_path, "[paths]\n"), Settings(), registry)
    assert policy["paths"]["sandboxed_servers"] == list(DEFAULT_SANDBOXED_SERVERS)


def test_load_policy_rejects_an_unknown_tool_key(tmp_path: Path, registry: Registry) -> None:
    """F3 scope 5: a typo in the override table must not silently do nothing."""
    body = '[effects]\n"github.delete_file" = "destructive"\n'
    with pytest.raises(PolicyError, match="not a tool in the registry"):
        load_policy(_policy_file(tmp_path, body), Settings(), registry)


def test_load_policy_rejects_an_invalid_effect_value(tmp_path: Path, registry: Registry) -> None:
    """Without this, `= "destroy"` falls through to the annotation tier and looks like it worked."""
    body = '[effects]\n"git.git_reset" = "destroy"\n'
    with pytest.raises(PolicyError, match="expected one of"):
        load_policy(_policy_file(tmp_path, body), Settings(), registry)


def test_load_policy_is_not_cached_across_sandbox_changes(
    tmp_path: Path, registry: Registry
) -> None:
    """The TOML read is cached; the env overlay must not be, or a moved sandbox goes unnoticed."""
    path = _policy_file(tmp_path, "[effects]\n")
    first = load_policy(path, Settings(sandbox_dir="./sandbox"), registry)
    second = load_policy(path, Settings(sandbox_dir="./other"), registry)
    assert first["paths"]["sandbox_root"] != second["paths"]["sandbox_root"]


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
