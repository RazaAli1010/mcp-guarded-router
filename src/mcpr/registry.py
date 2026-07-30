"""Read-only queries over the frozen tool-schema snapshot.

`schemas/registry.json` is the contract (SPEC.md 3.1): training, evaluation, the guardrails
and every test read it, and nothing here ever contacts a server. Only `snapshot.py` writes
the file; this module only ever reads it.

`effect_for` and `confusables` are pure - no I/O, no clock, no globals - because F2's prompt
builder and F8's metrics both depend on them producing byte-identical results across runs
and across machines (SPEC.md 3.3, 3.6).
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path

from mcpr.config import REGISTRY_PATH, Settings, load_env, resolve
from mcpr.io import sha256_file
from mcpr.types import Effect, EffectSource, Registry, ToolSpec

#: Tunable effect overrides. Local, not a SPEC.md 7.1 constant.
POLICY_PATH = "config/policy.toml"

#: Every section `config/policy.toml` may carry (SPEC.md 8). `load_policy` returns all of them
#: whether or not the file defines them, so no caller has to defend against a missing table.
POLICY_SECTIONS = ("effects", "rules", "limits", "thresholds", "paths")

#: Servers whose path arguments name something on the local disk, and are therefore subject to
#: containment (SPEC.md 8.3 as amended by F3). Overridable via `[paths] sandboxed_servers`.
DEFAULT_SANDBOXED_SERVERS = ("filesystem", "git")

_EFFECTS: frozenset[str] = frozenset({"read", "write", "destructive"})

#: SPEC.md 8.1 rule 3, verbatim. Matched against the first token of the tool name before `_`.
READ_VERBS = frozenset(
    {"get", "list", "search", "read", "show", "fetch", "find", "status", "log", "diff", "tree"}
)
DESTRUCTIVE_VERBS = frozenset(
    {"delete", "remove", "drop", "reset", "force", "merge", "revert", "unstar"}
)

#: Weights of the F1 confusability score. Same server + shared verb prefix must rank highest.
W_DESCRIPTION = 0.5
W_NAME = 0.3
W_SERVER = 0.2

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=8)
def load_registry(path: str | Path = REGISTRY_PATH) -> Registry:
    """Parse the frozen snapshot into a `Registry`.

    Impure on the first call for a given path; cached thereafter, which is why every caller
    must treat the result as read-only. The tool order on disk is preserved rather than
    re-sorted: `snapshot.py` guarantees the sort on write, so an out-of-order file is a bug
    to be caught by a test, not something to paper over on read.
    """
    return Registry.model_validate_json(resolve(path).read_text(encoding="utf-8"))


class PolicyError(ValueError):
    """A malformed `config/policy.toml`.

    Raised at load time rather than tolerated at check time: an override naming a tool that does
    not exist, or claiming an effect class that does not exist, would otherwise fall silently
    through to the annotation and heuristic tiers. A typo in the one table that pins destructive
    operations must be loud (F3 scope 5).
    """


@lru_cache(maxsize=8)
def _read_policy_toml(path: str) -> dict:
    """Parse `config/policy.toml`. Impure on the first call for a given path; cached thereafter.

    Cached on the path *string* only. `load_policy` deliberately is not cached, because its
    result folds in `MCPR_SANDBOX_DIR` and would go stale the moment a test moved the sandbox.
    """
    with open(resolve(path), "rb") as fh:
        return tomllib.load(fh)


@lru_cache(maxsize=8)
def load_effect_overrides(path: str | Path = POLICY_PATH) -> dict[str, str]:
    """Read `[effects]` from `config/policy.toml` - rule 1 of SPEC.md 8.1.

    Impure on the first call for a given path; cached thereafter. Values are **not** validated
    here - `snapshot.py` calls this while building the registry the validation would need, so
    the check lives in `load_policy`, which every guard goes through.
    """
    data = _read_policy_toml(str(path))
    return {str(k): str(v) for k, v in (data.get("effects") or {}).items()}


def load_policy(
    path: str | Path = POLICY_PATH,
    settings: Settings | None = None,
    registry: Registry | None = None,
) -> dict:
    """Load the whole guard policy, with the sandbox root resolved to an absolute path. Impure.

    This is the caller boundary that keeps `policy_guard.check` pure: the guard never reads an
    env var or touches the filesystem, it just reads `policy["paths"]["sandbox_root"]`, which
    arrives here already resolved from `MCPR_SANDBOX_DIR` (SPEC.md 7.2).

    Every section of SPEC.md 8's policy file is present in the result whether or not the file
    declares it, so callers index rather than `.get` twice.

    Raises `PolicyError` when an `[effects]` key is not a `qualified_name` in the registry, or
    its value is not one of read/write/destructive. Validating the keys is why this function
    needs a registry at all, and why it cannot live in `config.py` - that module is imported
    *by* this one.
    """
    data = _read_policy_toml(str(path))
    policy: dict = {name: dict(data.get(name) or {}) for name in POLICY_SECTIONS}

    effects = {str(k): str(v) for k, v in policy["effects"].items()}
    known = {spec.qualified_name for spec in (registry or load_registry()).tools}
    for qualified_name, effect in sorted(effects.items()):
        if effect not in _EFFECTS:
            raise PolicyError(
                f"[effects] {qualified_name!r} = {effect!r}: expected one of {sorted(_EFFECTS)}"
            )
        if qualified_name not in known:
            raise PolicyError(
                f"[effects] {qualified_name!r} is not a tool in the registry. An override for a "
                "tool that does not exist would silently do nothing."
            )
    policy["effects"] = effects

    sandbox_dir = (settings if settings is not None else load_env()).sandbox_dir
    policy["paths"]["sandbox_root"] = str(resolve(sandbox_dir))
    policy["paths"].setdefault("sandboxed_servers", list(DEFAULT_SANDBOXED_SERVERS))
    return policy


def get(qualified_name: str, registry: Registry | None = None) -> ToolSpec:
    """Look up one tool by its `qualified_name`.

    Impure only when `registry` is omitted, in which case the cached snapshot is loaded.
    Raises `KeyError` for an unknown name - including `"none"`, which is a reserved routing
    value and deliberately absent from the registry (SPEC.md 6.2).
    """
    for spec in (registry or load_registry()).tools:
        if spec.qualified_name == qualified_name:
            return spec
    raise KeyError(qualified_name)


def effect_for(spec: ToolSpec, overrides: dict[str, str] | None = None) -> Effect:
    """Derive a tool's effect class per SPEC.md 8.1, first match wins. Pure.

    Three tiers: an explicit `config/policy.toml [effects]` entry, then MCP annotations
    (`destructiveHint` / `readOnlyHint`), then a heuristic on the first token of the name.
    Tier 3 exists because many real servers ship no annotations at all - which is why
    `registry.meta.json` reports annotation coverage per server.
    """
    return effect_from_parts(spec.qualified_name, spec.name, spec.annotations, overrides)


def confusables(qualified_name: str, k: int, registry: Registry | None = None) -> list[str]:
    """Return the `k` tools most easily mistaken for this one, best first. Pure given `registry`.

    Score is the F1 formula: `0.5 * jaccard(description tokens) + 0.3 * jaccard(name tokens)
    + 0.2 * (same server)`, ties broken by `qualified_name` ascending so the result is stable
    across runs and machines. Tokens are lowercase `[a-z0-9]+` runs with no stopword removal
    - the spec does not define a stoplist, and inventing one would make the ranking depend on
    an unreviewable word list.

    The tool itself is never in its own result.
    """
    return [name for name, _score in confusable_scores(qualified_name, k, registry)]


def confusable_scores(
    qualified_name: str, k: int, registry: Registry | None = None
) -> list[tuple[str, float]]:
    """`confusables`, with each tool's score attached. Pure given `registry`.

    This is where the ranking actually happens; `confusables` drops the scores. Exposed so
    `mcpr snapshot confusables` can show the numbers - a ranking nobody can inspect is a
    ranking nobody can debug.
    """
    reg = registry or load_registry()
    target = get(qualified_name, reg)
    target_desc = _tokens(target.description)
    target_name = set(target.name.split("_"))

    # Negate the score so a single ascending sort ranks by score desc, then name asc.
    scored = sorted(
        (
            -(
                W_DESCRIPTION * _jaccard(target_desc, _tokens(other.description))
                + W_NAME * _jaccard(target_name, set(other.name.split("_")))
                + W_SERVER * float(other.server == target.server)
            ),
            other.qualified_name,
        )
        for other in reg.tools
        if other.qualified_name != qualified_name
    )
    return [(name, -score) for score, name in scored[:k]]


def effect_from_parts(
    qualified_name: str,
    name: str,
    annotations: dict,
    overrides: dict[str, str] | None = None,
) -> Effect:
    """The body of `effect_for`, callable before a `ToolSpec` exists. Pure.

    `snapshot.py` needs the effect in order to construct a `ToolSpec` at all - `effect` is a
    required field - so the derivation cannot itself require one.
    """
    return effect_with_source_from_parts(qualified_name, name, annotations, overrides)[0]


def effect_with_source(
    spec: ToolSpec, overrides: dict[str, str] | None = None
) -> tuple[Effect, EffectSource]:
    """`effect_for`, plus which of SPEC.md 8.1's three tiers decided it. Pure.

    `mcpr guard audit` exits 1 on any tool that is `destructive` by `heuristic` alone: a guess
    from a verb prefix must never be the only thing standing between the router and a delete.
    That check needs the provenance, not just the answer.
    """
    return effect_with_source_from_parts(
        spec.qualified_name, spec.name, spec.annotations, overrides
    )


def effect_with_source_from_parts(
    qualified_name: str,
    name: str,
    annotations: dict,
    overrides: dict[str, str] | None = None,
) -> tuple[Effect, EffectSource]:
    """The one implementation of SPEC.md 8.1, first match wins. Pure.

    Everything else in this module - `effect_for`, `effect_from_parts`, `effect_with_source` -
    delegates here, so the derivation cannot drift between the path `snapshot.py` bakes into the
    registry and the path the guards evaluate at runtime.
    """
    override = (overrides or {}).get(qualified_name)
    if override in _EFFECTS:
        return override, "override"  # type: ignore[return-value]

    if annotations.get("destructiveHint") is True:
        return "destructive", "annotation"
    if annotations.get("readOnlyHint") is True:
        return "read", "annotation"

    verb = name.split("_")[0].lower()
    if verb in READ_VERBS:
        return "read", "heuristic"
    if verb in DESTRUCTIVE_VERBS:
        return "destructive", "heuristic"
    return "write", "heuristic"


def registry_sha256(path: str | Path = REGISTRY_PATH) -> str:
    """Hex sha256 of the snapshot file as it sits on disk. Impure.

    SPEC.md 10.3 lets a metric be computed only from prediction files whose recorded registry
    hash matches the working tree, so this is the number F8 and F10 compare against.
    """
    return sha256_file(resolve(path))


# --- internals --------------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric token set of a string. Pure."""
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when both are empty. Pure."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0
