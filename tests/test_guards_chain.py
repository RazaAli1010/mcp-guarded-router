"""The chain that combines the layers, and the confirmation gate behind every mutation.

Neither belongs to layer 1 or layer 2, so they live here rather than being wedged into
`test_guards_schema.py` or `test_guards_policy.py`.

The last test in this file, `test_no_autoconfirm_exists`, is the crude one that matters most:
claim C3 rests on there being no path in the repo that executes a write without a redeemed
token, and a grep is the only thing that can assert the *absence* of such a path.
"""

from __future__ import annotations

import ast
import inspect
import io
import re
import tokenize
from pathlib import Path

import pytest

from mcpr.config import CONFIRM_TTL_SECONDS, PROJECT_ROOT, Settings
from mcpr.guards import ACTION_RANK, GuardChain, schema_guard
from mcpr.guards.confirm import ConfirmStore, call_hash
from mcpr.registry import POLICY_PATH, load_policy, load_registry
from mcpr.types import GuardDecision, Registry, ToolCall, UntrustedBlock

SRC = PROJECT_ROOT / "src" / "mcpr"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


@pytest.fixture
def policy(tmp_path: Path, registry: Registry) -> dict:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return load_policy(POLICY_PATH, Settings(sandbox_dir=str(sandbox)), registry=registry)


READ = ToolCall(tool="github.search_code", arguments={"query": "mcp"})
WRITE = ToolCall(tool="filesystem.write_file", arguments={"path": "a.txt", "content": "x"})


class _FakeInjectionGuard:
    """Stands in for F4. Structural typing means no import from `guards` is needed."""

    def __init__(self, *decisions: GuardDecision) -> None:
        self.decisions = list(decisions)
        self.seen: list[list[UntrustedBlock]] | None = None

    def check(self, untrusted: list[UntrustedBlock]) -> list[GuardDecision]:
        self.seen = [untrusted]
        return self.decisions


def _decision(action: str, code: str) -> GuardDecision:
    return GuardDecision(layer="injection", action=action, code=code, detail="")


# --- chain composition ---------------------------------------------------------------------------


def test_chain_runs_every_layer_with_no_short_circuit(registry: Registry, policy: dict) -> None:
    """A call that fails schema validation must still receive a policy verdict.

    SPEC.md 6.6 forbids short-circuiting so per-layer statistics stay complete; without it the
    F8 numbers would silently under-count whichever layer happens to run second.
    """
    result = GuardChain(registry, policy).check(
        ToolCall(tool="filesystem.write_file", arguments={"path": "a.txt"})
    )

    layers = [d.layer for d in result.decisions]
    assert "schema" in layers and "policy" in layers
    assert {d.code for d in result.decisions} >= {
        "missing_required",
        "destructive_requires_confirmation",
    }


def test_precedence_block_beats_confirm(registry: Registry, policy: dict) -> None:
    """A path escape on a destructive tool: policy confirms nothing, it blocks."""
    result = GuardChain(registry, policy).check(
        ToolCall(tool="filesystem.write_file", arguments={"path": "../out", "content": "x"})
    )
    assert result.final_action == "block"


def test_precedence_confirm_beats_allow(registry: Registry, policy: dict) -> None:
    result = GuardChain(registry, policy).check(WRITE)
    assert result.final_action == "confirm"
    assert any(d.action == "allow" for d in result.decisions)


def test_allow_only_when_every_layer_allows(registry: Registry, policy: dict) -> None:
    result = GuardChain(registry, policy).check(READ)
    assert result.final_action == "allow"
    assert result.blocked_by is None
    assert {d.action for d in result.decisions} == {"allow"}


def test_action_rank_matches_the_spec_ordering() -> None:
    assert ACTION_RANK["block"] > ACTION_RANK["confirm"] > ACTION_RANK["allow"]


def test_blocked_by_is_the_first_block_in_chain_order(registry: Registry, policy: dict) -> None:
    """Decisions are appended in chain order, so `blocked_by` names the earliest blocking code."""
    guard = _FakeInjectionGuard(_decision("block", "injected_instructions"))
    result = GuardChain(registry, policy, injection_guard=guard).check(
        ToolCall(tool="filesystem.write_file", arguments={"path": "../out", "content": "x"})
    )

    assert result.final_action == "block"
    assert result.blocked_by == "injected_instructions"
    # The later block is still recorded; only the attribution is first-wins.
    assert "path_escape" in {d.code for d in result.decisions}


# --- the layer-3 seam (F4) -----------------------------------------------------------------------


def test_injection_guard_absent_contributes_no_decisions(registry: Registry, policy: dict) -> None:
    """F4 has not landed; the chain must work without it rather than fabricating a verdict."""
    result = GuardChain(registry, policy).check(READ)
    assert [d.layer for d in result.decisions] == ["schema", "policy"]


def test_injection_guard_decisions_come_first(registry: Registry, policy: dict) -> None:
    """Untrusted content is inspected before the model's output is trusted enough to validate."""
    guard = _FakeInjectionGuard(_decision("allow", "clean"))
    result = GuardChain(registry, policy, injection_guard=guard).check(READ)

    assert [d.layer for d in result.decisions][0] == "injection"


def test_injection_guard_is_called_even_with_no_untrusted_blocks(
    registry: Registry, policy: dict
) -> None:
    """Passing `[]` rather than skipping keeps per-layer statistics complete."""
    guard = _FakeInjectionGuard(_decision("allow", "clean"))
    GuardChain(registry, policy, injection_guard=guard).check(READ)

    assert guard.seen == [[]]


# --- failing closed ------------------------------------------------------------------------------


def test_guard_error_fails_closed(
    registry: Registry, policy: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md 3.7: an unhandled error inside the chain blocks. It never falls through.

    This is also why `guards/__init__.py` imports its submodules as module objects - a
    `from .schema_guard import check` would bind the original at import time and this patch
    would silently do nothing, leaving the property untested.
    """

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(schema_guard, "check", boom)
    result = GuardChain(registry, policy).check(READ)

    assert result.final_action == "block"
    assert result.blocked_by == "guard_error"
    # The other layers still ran: one failure must not blind the rest of the chain.
    assert "policy" in [d.layer for d in result.decisions]


def test_an_exploding_injection_guard_also_blocks(registry: Registry, policy: dict) -> None:
    """F4's code is not yet written, so the chain must already be defended against it."""

    class Exploding:
        def check(self, untrusted: list[UntrustedBlock]) -> list[GuardDecision]:
            raise ValueError("nope")

    result = GuardChain(registry, policy, injection_guard=Exploding()).check(READ)

    assert result.final_action == "block"
    assert result.blocked_by == "guard_error"


def test_one_layer_failure_yields_one_guard_error(
    registry: Registry, policy: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layer 1 runs two functions but shares one wrapper, so it cannot report twice."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(schema_guard, "check", boom)
    monkeypatch.setattr(schema_guard, "extra_properties", boom)
    result = GuardChain(registry, policy).check(READ)

    assert [d.code for d in result.decisions].count("guard_error") == 1


# --- the confirmation gate -------------------------------------------------------------------


@pytest.fixture
def clock() -> list[float]:
    return [0.0]


@pytest.fixture
def store(clock: list[float]) -> ConfirmStore:
    """A store on an injected clock, so TTL behaviour is tested without sleeping."""
    return ConfirmStore(clock=lambda: clock[0])


def test_confirm_token_single_use(store: ConfirmStore) -> None:
    token = store.issue(WRITE)
    assert store.redeem(token.token, WRITE) is True
    assert store.redeem(token.token, WRITE) is False


def test_confirm_token_expires(store: ConfirmStore, clock: list[float]) -> None:
    token = store.issue(WRITE)
    clock[0] = CONFIRM_TTL_SECONDS + 1
    assert store.redeem(token.token, WRITE) is False


def test_confirm_token_is_valid_right_up_to_the_ttl(
    store: ConfirmStore, clock: list[float]
) -> None:
    token = store.issue(WRITE)
    clock[0] = CONFIRM_TTL_SECONDS - 0.001
    assert store.redeem(token.token, WRITE) is True


def test_confirm_token_bound_to_call(store: ConfirmStore) -> None:
    """A token approved for one call must not run a different one - the whole point of call_hash."""
    token = store.issue(WRITE)
    mutated = ToolCall(
        tool="filesystem.write_file", arguments={"path": "a.txt", "content": "MALICIOUS"}
    )
    assert store.redeem(token.token, mutated) is False


def test_confirm_token_bound_to_the_tool_as_well(store: ConfirmStore) -> None:
    token = store.issue(WRITE)
    other = ToolCall(tool="filesystem.edit_file", arguments={"path": "a.txt", "content": "x"})
    assert store.redeem(token.token, other) is False


def test_confirm_token_is_not_consumed_by_a_mismatched_call(store: ConfirmStore) -> None:
    """Otherwise one wrong call denies the user the confirmation they were about to give."""
    token = store.issue(WRITE)
    mutated = ToolCall(tool="filesystem.write_file", arguments={"path": "a.txt", "content": "no"})

    assert store.redeem(token.token, mutated) is False
    assert store.redeem(token.token, WRITE) is True


def test_unknown_token_is_refused(store: ConfirmStore) -> None:
    assert store.redeem("not-a-token", WRITE) is False


def test_tokens_are_unguessable_and_distinct(store: ConfirmStore) -> None:
    tokens = {store.issue(WRITE).token for _ in range(16)}
    assert len(tokens) == 16
    assert all(len(t) >= 32 for t in tokens)


def test_purge_drops_expired_tokens(store: ConfirmStore, clock: list[float]) -> None:
    store.issue(WRITE)
    store.issue(READ)
    clock[0] = CONFIRM_TTL_SECONDS + 1

    assert store.purge() == 2
    assert len(store) == 0


def test_issue_purges_so_the_store_cannot_grow_without_bound(
    store: ConfirmStore, clock: list[float]
) -> None:
    store.issue(WRITE)
    clock[0] = CONFIRM_TTL_SECONDS + 1
    store.issue(READ)

    assert len(store) == 1


# --- what call_hash does and does not forgive ---------------------------------------------------


def test_call_hash_survives_key_reordering() -> None:
    """JSON object key order carries no meaning, so sorting is a safe normalisation."""
    a = ToolCall(tool="filesystem.write_file", arguments={"path": "a", "content": "x"})
    b = ToolCall(tool="filesystem.write_file", arguments={"content": "x", "path": "a"})
    assert call_hash(a) == call_hash(b)


def test_call_hash_does_not_forgive_whitespace() -> None:
    """The divergence from `canonicalise_arguments`, asserted rather than left to a docstring.

    `arg_exact_acc` collapses internal whitespace, which is right for a metric and wrong for a
    gate: these two file bodies are materially different programs.
    """
    from mcpr.parse import canonicalise_arguments

    kept = {"path": "s.py", "content": "cleanup()\n\n# rm -rf / is commented out"}
    flattened = {"path": "s.py", "content": "cleanup() # rm -rf / is commented out"}

    assert canonicalise_arguments(kept) == canonicalise_arguments(flattened)
    assert call_hash(ToolCall(tool="filesystem.write_file", arguments=kept)) != call_hash(
        ToolCall(tool="filesystem.write_file", arguments=flattened)
    )


def test_call_hash_distinguishes_an_explicit_default_from_an_absent_key() -> None:
    """`canonicalise_arguments` strips schema defaults; a confirmation cannot afford to."""
    with_default = ToolCall(
        tool="filesystem.edit_file", arguments={"path": "a", "edits": [], "dryRun": False}
    )
    without = ToolCall(tool="filesystem.edit_file", arguments={"path": "a", "edits": []})
    assert call_hash(with_default) != call_hash(without)


# --- the absence of a bypass ---------------------------------------------------------------------

#: Flag and identifier spellings of a confirmation bypass. Matched instead of the bare word
#: "force" on purpose: `registry.DESTRUCTIVE_VERBS` legitimately contains "force" as a verb the
#: effect heuristic looks for, and a naive grep would fail on it forever, teaching whoever hits
#: it to loosen the test rather than fix a real problem.
BYPASS_RE = re.compile(
    r"auto[_-]?confirm|confirm[_-]?bypass|skip[_-]?confirm|no[_-]?confirm"
    r"|force[_-]?confirm|--yes\b|--force\b|\bassume_yes\b",
    re.I,
)


def _executable_source(path: Path) -> list[tuple[int, str]]:
    """`(lineno, text)` for every token that is neither a comment nor a docstring. Impure.

    Prose has to be excluded, or this test fails on the very sentences documenting that no
    bypass exists - `confirm.py` says "there is no auto-confirm flag, no `--yes`" in its module
    docstring, and that sentence is not a bypass.

    String *literals* are deliberately kept. `typer.Option(..., "--yes")` is precisely the thing
    being hunted, and it is a literal; dropping all strings would blind the test to the most
    likely way a bypass would actually be added.
    """
    source = path.read_text(encoding="utf-8")
    docstring_lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if ast.get_docstring(node) is None:
            continue
        first = node.body[0]
        docstring_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))

    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [
        (tok.start[0], tok.string)
        for tok in tokens
        if tok.type != tokenize.COMMENT and tok.start[0] not in docstring_lines
    ]


def test_no_autoconfirm_exists() -> None:
    """Claim C3 rests on the *absence* of a bypass, which only a grep can assert.

    Four checks, because a source scan alone has two blind spots - an environment variable, and
    a parameter that would let a caller pass one in.
    """
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{lineno}: {text}"
        for path in sorted(SRC.rglob("*.py"))
        for lineno, text in _executable_source(path)
        if BYPASS_RE.search(text)
    ]
    assert offenders == []

    # An env-var bypass is exactly what a source grep misses.
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert not BYPASS_RE.search(env_example)
    assert not [name for name in Settings.model_fields if BYPASS_RE.search(name)]

    # No parameter through which a bypass could reach the gate.
    assert set(inspect.signature(ConfirmStore.redeem).parameters) == {"self", "token", "call"}
    assert set(inspect.signature(ConfirmStore.issue).parameters) == {"self", "call"}

    # And the behavioural floor: nothing is redeemable out of an empty store.
    assert ConfirmStore().redeem("anything", WRITE) is False


def test_the_bypass_pattern_does_not_trip_on_the_destructive_verb_list() -> None:
    """Documents why BYPASS_RE is shaped the way it is, so nobody "simplifies" it to /force/."""
    from mcpr.registry import DESTRUCTIVE_VERBS

    assert "force" in DESTRUCTIVE_VERBS
    assert not BYPASS_RE.search("force")
    assert BYPASS_RE.search("--force")
    assert BYPASS_RE.search("auto_confirm")


def test_the_bypass_scan_would_actually_catch_one(tmp_path: Path) -> None:
    """A passing security test that cannot fail is worse than no test. Prove this one can.

    The scan skips comments and docstrings so it does not trip on the prose describing the
    absence of a bypass - which risks skipping so much that it sees nothing at all. This plants
    a bypass in both forms it would realistically take, plus decoy prose, and checks that the
    two real ones are caught and the decoys are not.
    """
    planted = tmp_path / "planted.py"
    planted.write_text(
        '"""Module docstring mentioning auto-confirm and --yes, which is only prose."""\n'
        "\n"
        "def run(auto_confirm: bool = False):\n"
        '    """Docstring claiming there is no --force flag anywhere."""\n'
        '    option = ("--yes",)  # comment mentioning skip_confirm\n'
        "    return option\n",
        encoding="utf-8",
    )

    caught = [text for _lineno, text in _executable_source(planted) if BYPASS_RE.search(text)]
    assert caught == ["auto_confirm", '"--yes"']
