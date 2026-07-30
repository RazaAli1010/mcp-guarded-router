"""Layer 2: effect derivation, path containment and the plan lock (SPEC.md 8.3).

Unlike `test_guards_schema.py`, these run against the **real** 50-tool snapshot rather than
`registry_min.json`. Two reasons, both load-bearing:

- `config/policy.toml [effects]` is validated against the real registry, and two of its keys
  (`filesystem.create_directory`, `git.git_checkout`) are absent from the 24-tool fixture. A
  test using the fixture would have to invent a different policy from the committed one, and
  would then be testing a configuration that never runs.
- `policy_cases.jsonl` is claim C3's evidence, and F8 replays it against the real registry.
  Asserting it here against a subset would be asserting the wrong thing.

Still no network and no MCP server: the snapshot is a committed file (SPEC.md 12).

The sandbox root is always a `tmp_path`, never the repo's own `sandbox/`, so nothing here
depends on the developer's working tree. No test needs a file to exist - `realpath` resolves
non-existent paths perfectly well - except the symlink case, which needs a real link.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from mcpr.config import Settings
from mcpr.guards import GuardChain, policy_guard
from mcpr.io import read_jsonl
from mcpr.registry import POLICY_PATH, load_policy, load_registry
from mcpr.types import Plan, Registry, ToolCall

FIXTURES = Path(__file__).parent / "fixtures"
POLICY_CASES = FIXTURES / "policy_cases.jsonl"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """An empty sandbox root under `tmp_path`, resolved the way the guard resolves it."""
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


@pytest.fixture
def policy(sandbox: Path, registry: Registry) -> dict:
    """The committed policy, with its sandbox root pointed at the temporary one."""
    return load_policy(POLICY_PATH, Settings(sandbox_dir=str(sandbox)), registry=registry)


def _check(registry: Registry, policy: dict, tool: str, arguments: dict, plan: Plan | None = None):
    return policy_guard.check(ToolCall(tool=tool, arguments=arguments), registry, policy, plan)


def _link_dir(link: Path, target: Path) -> None:
    """Create a directory link, or skip the test if this machine forbids every kind.

    A true symlink needs Developer Mode or elevation on Windows. A directory *junction* needs
    neither, and `os.path.realpath` resolves it identically, so the containment property under
    test is exercised for real rather than skipped on the author's machine. On POSIX the first
    branch always wins.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - platform dependent
        pytest.skip("neither a symlink nor a junction can be created here")


# --- the effect verdict --------------------------------------------------------------------------


def test_policy_read_allows(registry: Registry, policy: dict) -> None:
    decision = _check(registry, policy, "github.search_code", {"query": "mcp"})
    assert decision.action == "allow"
    assert decision.code == "read_only"


def test_policy_write_confirms(registry: Registry, policy: dict) -> None:
    decision = _check(registry, policy, "git.git_commit", {"repo_path": ".", "message": "wip"})
    assert decision.action == "confirm"
    assert decision.code == "destructive_requires_confirmation"
    assert "effect=write" in decision.evidence


def test_policy_destructive_confirms(registry: Registry, policy: dict) -> None:
    decision = _check(registry, policy, "git.git_reset", {"repo_path": "."})
    assert decision.action == "confirm"
    assert decision.code == "destructive_requires_confirmation"
    assert "effect=destructive" in decision.evidence


def test_unknown_tool_blocks_in_layer_two_as_well(registry: Registry, policy: dict) -> None:
    """Layer 2 must not fall through to an effect it cannot derive - the one place a leak fits."""
    decision = _check(registry, policy, "git.git_push", {"repo_path": "."})
    assert decision.action == "block"
    assert decision.code == "unknown_tool"


def test_effect_override_beats_the_baked_snapshot_value(registry: Registry, policy: dict) -> None:
    """git.git_checkout is `write` in the frozen file and `destructive` after the override.

    If the guard read `spec.effect` instead of deriving at runtime, `config/policy.toml
    [effects]` would be decorative. This is the test that proves it is not.
    """
    spec = next(s for s in registry.tools if s.qualified_name == "git.git_checkout")
    assert spec.effect == "write"

    decision = _check(
        registry, policy, "git.git_checkout", {"repo_path": ".", "branch_name": "main"}
    )
    assert "effect=destructive" in decision.evidence
    assert "source=override" in decision.evidence


def test_snapshot_effect_drift_is_only_ever_from_an_override(registry: Registry) -> None:
    """Stored and derived effects may differ, but only where the policy says so.

    SPEC.md 8.1 lets the runtime derivation win without rewriting the snapshot, which means the
    baked `ToolSpec.effect` is allowed to go stale. What must never happen is drift from a code
    change: editing READ_VERBS or DESTRUCTIVE_VERBS would silently desync the frozen file from
    the live guards, and nothing else in the suite would notice.

    Reads the committed policy directly, since that is the artifact the claim is about.
    """
    overrides = load_policy(POLICY_PATH, Settings(), registry=registry)["effects"]
    from mcpr.registry import effect_with_source

    for spec in registry.tools:
        effect, _source = effect_with_source(spec, overrides)
        if effect != spec.effect:
            assert spec.qualified_name in overrides, (
                f"{spec.qualified_name} derives {effect} but the snapshot says {spec.effect}, "
                "and no [effects] override explains it"
            )


# --- path containment ------------------------------------------------------------------------


def test_path_inside_sandbox_allows(registry: Registry, policy: dict) -> None:
    decision = _check(registry, policy, "filesystem.read_text_file", {"path": "notes.txt"})
    assert decision.action == "allow"


def test_path_escape_dotdot(registry: Registry, policy: dict) -> None:
    decision = _check(registry, policy, "filesystem.read_text_file", {"path": "../../etc/passwd"})
    assert decision.action == "block"
    assert decision.code == "path_escape"


def test_path_escape_absolute(registry: Registry, policy: dict) -> None:
    """`Path("/etc/passwd").is_absolute()` is False on Windows; the guard must catch it anyway."""
    decision = _check(registry, policy, "filesystem.read_text_file", {"path": "/etc/passwd"})
    assert decision.action == "block"
    assert decision.code == "path_escape"


def test_path_escape_home_expansion(registry: Registry, policy: dict) -> None:
    decision = _check(registry, policy, "filesystem.directory_tree", {"path": "~"})
    assert decision.action == "block"
    assert decision.code == "path_escape"


def test_path_escape_in_a_list_valued_argument(registry: Registry, policy: dict) -> None:
    """`git_add.files` has no description, so only the argument-name list catches it."""
    decision = _check(
        registry, policy, "git.git_add", {"repo_path": ".", "files": ["ok.txt", "../../x"]}
    )
    assert decision.action == "block"
    assert decision.code == "path_escape"
    assert any("../../x" in item for item in decision.evidence)


def test_path_escape_symlink(
    registry: Registry, policy: dict, sandbox: Path, tmp_path: Path
) -> None:
    """A link *inside* the sandbox pointing out. This is why realpath is applied to both sides.

    Purely lexical containment - normpath, or comparing the unresolved strings - passes this
    case, because `sandbox/escape/passwd` looks like it is inside the sandbox right up until the
    filesystem resolves it.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd").write_text("root:x:0:0:\n", encoding="utf-8")
    _link_dir(sandbox / "escape", outside)

    # The link really does redirect out of the sandbox, or the test proves nothing.
    resolved = os.path.realpath(sandbox / "escape" / "passwd")
    assert not resolved.startswith(os.path.realpath(sandbox))

    decision = _check(registry, policy, "filesystem.read_text_file", {"path": "escape/passwd"})
    assert decision.action == "block"
    assert decision.code == "path_escape"


def test_symlink_inside_the_sandbox_still_allows(
    registry: Registry, policy: dict, sandbox: Path
) -> None:
    """realpath on both sides must not turn a legitimate internal link into a false positive."""
    (sandbox / "real").mkdir()
    _link_dir(sandbox / "alias", sandbox / "real")

    decision = _check(registry, policy, "filesystem.read_text_file", {"path": "alias/notes.txt"})
    assert decision.action == "allow"


def test_path_of_a_non_sandboxed_server_is_ignored(registry: Registry, policy: dict) -> None:
    """`github.get_file_contents.path` defaults to "/" and is described as a directory.

    Under universal containment this benign remote read would block on its own schema default,
    and every such false positive lands in the reported `FPR_benign` (SPEC.md 9.3). GitHub is
    not in `sandboxed_servers` precisely because its paths name a repo, not the local disk.
    """
    decision = _check(
        registry, policy, "github.get_file_contents", {"owner": "a", "repo": "b", "path": "/"}
    )
    assert decision.action == "allow"


def test_non_string_path_is_left_to_layer_one(registry: Registry, policy: dict) -> None:
    """Layer 1 blocks it as `type_mismatch`; a second, misleading code here would help nobody."""
    decision = _check(registry, policy, "filesystem.read_text_file", {"path": ["a.txt"]})
    assert decision.code != "path_escape"


def test_containment_applies_to_reads_too(registry: Registry, policy: dict) -> None:
    """A read that escapes the sandbox is an exfiltration, not a harmless lookup."""
    decision = _check(
        registry, policy, "filesystem.search_files", {"path": "../..", "pattern": "*"}
    )
    assert decision.action == "block"
    assert decision.code == "path_escape"


# --- the plan lock ----------------------------------------------------------------------------


def _plan(servers: list[str], max_effect: str) -> Plan:
    return Plan(allowed_servers=servers, max_effect=max_effect, created_at=0.0)


def test_plan_lock_other_server_confirms(registry: Registry, policy: dict) -> None:
    decision = _check(
        registry,
        policy,
        "filesystem.read_text_file",
        {"path": "a.txt"},
        _plan(["github"], "read"),
    )
    assert decision.action == "confirm"
    assert decision.code == "outside_plan"


def test_plan_lock_effect_escalation_confirms(registry: Registry, policy: dict) -> None:
    decision = _check(
        registry,
        policy,
        "filesystem.write_file",
        {"path": "a.txt", "content": "x"},
        _plan(["filesystem"], "read"),
    )
    assert decision.action == "confirm"
    assert decision.code == "outside_plan"


def test_plan_lock_is_never_a_bypass(registry: Registry, policy: dict) -> None:
    """A call inside the plan is still gated by its effect. The lock only ever adds a check."""
    decision = _check(
        registry,
        policy,
        "git.git_commit",
        {"repo_path": ".", "message": "wip"},
        _plan(["git"], "destructive"),
    )
    assert decision.action == "confirm"
    assert decision.code == "destructive_requires_confirmation"


def test_no_plan_means_no_plan_decisions(registry: Registry, policy: dict) -> None:
    decision = _check(registry, policy, "github.search_code", {"query": "mcp"}, None)
    assert decision.code == "read_only"


# --- the C3 fixture ---------------------------------------------------------------------------


def _cases() -> list[dict]:
    return list(read_jsonl(POLICY_CASES))


def test_policy_cases_has_at_least_forty_rows() -> None:
    assert len(_cases()) >= 40


def test_policy_cases_are_key_sorted() -> None:
    """SPEC.md 3.3 applied to a hand-written fixture: it must round-trip byte-identically."""
    import json

    for line in POLICY_CASES.read_text(encoding="utf-8").splitlines():
        assert json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False) == line


def test_policy_case_ids_are_unique() -> None:
    ids = [row["id"] for row in _cases()]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("row", _cases(), ids=lambda r: f"{r['id']}-{r['expected_code']}")
def test_policy_cases_fixture(row: dict, registry: Registry, policy: dict) -> None:
    """Every row of claim C3's evidence, asserted at chain level - the shape F8 consumes.

    `model_construct` rather than `ToolCall(...)`, because one row deliberately carries a
    non-object `arguments` to reach `not_object`, which validated construction would reject.
    """
    plan = Plan(**row["plan"], created_at=0.0) if "plan" in row else None
    result = GuardChain(registry, policy, plan=plan).check(ToolCall.model_construct(**row["call"]))

    assert result.final_action == row["expected_action"], row["note"]
    assert row["expected_code"] in {d.code for d in result.decisions}, row["note"]


def test_every_expressible_code_appears_in_the_fixture() -> None:
    """The fixture is the C3 evidence, so a code with no row is a gap in the evidence.

    `unknown_property` is excluded by design: no captured schema sets
    `additionalProperties: false`, and inventing one here would put a synthetic tool in a file
    F8 replays against the real registry, where it would silently resolve to `unknown_tool`.
    `tests/test_guards_schema.py` covers it against a hand-built strict spec instead.
    """
    expected = {
        "abstained",
        "constraint_violation",
        "destructive_requires_confirmation",
        "enum_violation",
        "extra_property_permitted",
        "missing_required",
        "not_object",
        "outside_plan",
        "path_escape",
        "read_only",
        "type_mismatch",
        "unknown_tool",
    }
    assert {row["expected_code"] for row in _cases()} == expected


def test_every_destructive_case_is_gated() -> None:
    """Claim C3 in miniature: no write or destructive row may expect `allow`.

    F8 computes `destructive_catch_rate` from this file and fails the run at anything below
    1.000, so a row that expected otherwise would be a defect in the evidence itself.
    """
    for row in _cases():
        if row["expected_code"] == "destructive_requires_confirmation":
            assert row["expected_action"] in {"confirm", "block"}, row["id"]


# --- conformance of the committed artifacts ----------------------------------------------------


def test_all_destructive_have_overrides(registry: Registry) -> None:
    """Mirrors `mcpr guard audit`'s exit-1 condition, against the real committed files.

    This test deliberately reads `schemas/registry.json` and `config/policy.toml` rather than a
    fixture - like `test_env_split.py` parsing `pyproject.toml`, it is a conformance check on
    what is committed, not a unit test of a function. A verb guess must never be the only thing
    standing between the router and a delete.
    """
    from mcpr.cli import audit_rows

    overrides = load_policy(POLICY_PATH, Settings(), registry=registry)["effects"]
    unpinned = [row.qualified_name for row in audit_rows(registry, overrides) if row.unpinned]
    assert unpinned == []


def test_layer_two_reads_nothing_but_its_arguments() -> None:
    """SPEC.md 8.3: no model, no network, no clock, and never the untrusted content.

    A crude import check, and worth it - the day someone reaches for `requests` or `load_env`
    inside the effect gate, the purity claim in the module docstring becomes false and this is
    the only thing that would say so.
    """
    source = Path(policy_guard.__file__).read_text(encoding="utf-8")
    for forbidden in ("import httpx", "import requests", "load_env(", "time.time", "openai"):
        assert forbidden not in source, forbidden
