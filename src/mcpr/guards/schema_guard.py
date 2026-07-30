"""Layer 1: validate a proposed call against the tool's real MCP JSON Schema (SPEC.md 8.2).

No model, no network, no clock - `check` is pure given its arguments. Every violation blocks;
the code says which kind, because the `schema_violation_rate` breakdown is reported evidence for
claim C2 and a single boolean would not be informative enough to act on.

The schemas here are captured verbatim from real servers, warts included. Two consequences worth
stating up front:

- Real MCP schemas almost never set `additionalProperties: false` - none of the 50 tools in the
  snapshot does - so an invented argument validates cleanly. We do **not** inject that keyword
  ourselves: fabricating violations would inflate the very metric C2 reports. `extra_properties`
  records the occurrence as a non-blocking `allow` decision instead, so the frequency stays
  measurable without being manufactured.
- Drafts differ per server, so the validator class is selected from each schema's own `$schema`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from jsonschema.exceptions import ValidationError
from jsonschema.protocols import Validator
from jsonschema.validators import Draft202012Validator, validator_for

from mcpr.registry import get as registry_get
from mcpr.types import GuardDecision, Registry, ToolCall

#: `error.validator` -> SPEC.md 8.2 code. `type` is resolved separately: at the document root
#: against `"object"` it means the whole `arguments` value was not an object, which is its own
#: code. Anything not listed is a `constraint_violation` - the catch-all for `minimum`,
#: `maxLength`, `pattern`, `format` and the rest of the vocabulary.
_VALIDATOR_CODES = {
    "required": "missing_required",
    "additionalProperties": "unknown_property",
    "type": "type_mismatch",
    "enum": "enum_violation",
    "const": "enum_violation",
}


def check(call: ToolCall, registry: Registry) -> GuardDecision:
    """Validate `call.arguments` against the tool's `input_schema`. Pure.

    Returns exactly one decision. `"none"` abstains cleanly; an unrecognised tool blocks before
    any schema work; otherwise the first validation error - under the deterministic ordering
    described in `_error_key` - decides the code.
    """
    if call.tool == "none":
        return GuardDecision(
            layer="schema",
            action="allow",
            code="abstained",
            detail="the router declined to call a tool",
        )

    try:
        spec = registry_get(call.tool, registry)
    except KeyError:
        return GuardDecision(
            layer="schema",
            action="block",
            code="unknown_tool",
            detail=f"{call.tool!r} is not in the registry",
            evidence=[call.tool],
        )

    # Unreachable through a validated `ToolCall`, but reachable through `model_construct`, which
    # is how F8 builds prediction rows in bulk. A guard that trusts its caller is not a guard.
    if not isinstance(call.arguments, dict):
        return GuardDecision(
            layer="schema",
            action="block",
            code="not_object",
            detail=f"arguments must be a JSON object, got {type(call.arguments).__name__}",
            evidence=[repr(call.arguments)[:200]],
        )

    errors = sorted(_validator(spec.input_schema).iter_errors(call.arguments), key=_error_key)
    if not errors:
        return GuardDecision(
            layer="schema",
            action="allow",
            code="valid",
            detail=f"arguments satisfy the {spec.qualified_name} schema",
        )

    error = errors[0]
    return GuardDecision(
        layer="schema",
        action="block",
        code=_code_for(error),
        detail=error.message,
        evidence=[
            error.json_path,
            f"{error.validator}={json.dumps(error.validator_value, sort_keys=True, default=str)}",
        ],
    )


def extra_properties(call: ToolCall, registry: Registry) -> GuardDecision | None:
    """Report arguments the schema does not declare, without blocking on them. Pure.

    Returns `None` when there is nothing to say. Real MCP schemas omit
    `additionalProperties: false`, so these arguments are *legal* - but they are also the shape a
    hallucinated argument takes, and SPEC.md's C2 claim is about format adherence. Recording them
    as a separate `allow` decision keeps the rate measurable without inventing a violation.

    Emits nothing when the schema does forbid extras: `check` has already blocked with
    `unknown_property`, and a second decision about the same keys would double-count.
    """
    if call.tool == "none" or not isinstance(call.arguments, dict):
        return None
    try:
        schema = registry_get(call.tool, registry).input_schema
    except KeyError:
        return None
    if schema.get("additionalProperties") is False:
        return None

    declared = set(schema.get("properties") or {})
    patterns = schema.get("patternProperties") or {}
    extras = sorted(k for k in call.arguments if k not in declared and not _matches(k, patterns))
    if not extras:
        return None

    return GuardDecision(
        layer="schema",
        action="allow",
        code="extra_property_permitted",
        detail=(
            f"{len(extras)} argument(s) are not declared by the {call.tool} schema, which does "
            "not set additionalProperties: false"
        ),
        evidence=extras,
    )


def validator_cache_info() -> object:
    """`CacheInfo` for the compiled-validator cache. Impure (reads process state)."""
    return _build_validator.cache_info()


def clear_validator_cache() -> None:
    """Drop every compiled validator. Impure; exists so tests can assert on cache behaviour."""
    _build_validator.cache_clear()


# --- internals --------------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _build_validator(schema_json: str) -> Validator:
    """Compile one validator. Impure only in that it memoises; the result is a pure validator.

    Keyed on the canonical JSON *string* rather than the sha256 the F3 spec suggests. A digest is
    one-way, so keying on it would need a second digest-to-schema map to recover the input, which
    buys nothing: the string is already hashable and is exactly the identity the digest would
    hash. The cost the spec warns about is compiling a validator per call, and that is what this
    avoids.
    """
    schema = json.loads(schema_json)
    return validator_for(schema, default=Draft202012Validator)(schema)


def _validator(schema: dict) -> Validator:
    """Fetch or compile the validator for a schema. Impure only through the cache."""
    return _build_validator(json.dumps(schema, sort_keys=True, ensure_ascii=False))


def _error_key(error: ValidationError) -> tuple:
    """Total order over validation errors, so "the first error" is reproducible. Pure.

    `best_match` is not deterministic enough for a number that gets reported, and the obvious
    `list(error.absolute_path)` key raises `TypeError` as soon as two errors sit under the same
    array with an index and a property name (`['edits', 0]` against `['edits', 'x']` compares an
    int to a str). Every path element is widened to `(kind, index, name)` so array positions
    still order numerically while staying comparable with property names.

    Root-level errors have an empty path and therefore sort first, which is the right priority:
    a missing required argument outranks a type failure nested inside one.
    """
    path = tuple((0, p, "") if isinstance(p, int) else (1, 0, str(p)) for p in error.absolute_path)
    return (
        path,
        str(error.validator or ""),
        json.dumps(error.validator_value, sort_keys=True, default=str),
        error.message,
    )


def _code_for(error: ValidationError) -> str:
    """Map one `ValidationError` onto a SPEC.md 8.2 code. Pure."""
    at_root = not list(error.absolute_path)
    wants_object = error.validator_value in ("object", ["object"])
    if error.validator == "type" and at_root and wants_object:
        return "not_object"
    return _VALIDATOR_CODES.get(str(error.validator), "constraint_violation")


def _matches(key: str, patterns: dict) -> bool:
    """Whether an argument name is covered by a `patternProperties` entry. Pure."""
    return any(re.search(pattern, key) for pattern in patterns)
