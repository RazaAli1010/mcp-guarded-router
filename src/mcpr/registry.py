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

from mcpr.config import REGISTRY_PATH, resolve
from mcpr.io import sha256_file
from mcpr.types import Effect, Registry, ToolSpec

#: Tunable effect overrides. Local, not a SPEC.md 7.1 constant.
POLICY_PATH = "config/policy.toml"

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


@lru_cache(maxsize=8)
def load_effect_overrides(path: str | Path = POLICY_PATH) -> dict[str, str]:
    """Read `[effects]` from `config/policy.toml` - rule 1 of SPEC.md 8.1.

    Impure on the first call for a given path; cached thereafter. The table is empty until
    F3 fills it in, so this returns `{}` today and the derivation falls through to rules 2
    and 3.
    """
    with open(resolve(path), "rb") as fh:
        data = tomllib.load(fh)
    return {str(k): str(v) for k, v in (data.get("effects") or {}).items()}


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
    override = (overrides or {}).get(qualified_name)
    if override in ("read", "write", "destructive"):
        return override  # type: ignore[return-value]

    if annotations.get("destructiveHint") is True:
        return "destructive"
    if annotations.get("readOnlyHint") is True:
        return "read"

    verb = name.split("_")[0].lower()
    if verb in READ_VERBS:
        return "read"
    if verb in DESTRUCTIVE_VERBS:
        return "destructive"
    return "write"


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
