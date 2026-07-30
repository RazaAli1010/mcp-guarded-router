"""Layer 1: JSON Schema validation of a proposed call (SPEC.md 8.2).

Runs against the 24-tool `registry_min.json` fixture, so no network and no MCP server
(SPEC.md 12). The schemas in it are captured verbatim from real servers, which is the point:
these tests check behaviour against the drafts, enums and nested arrays that actually ship,
not against schemas written to make the guard look good.

One SPEC.md 8.2 code - `unknown_property` - cannot be produced by any real schema, because none
of them sets `additionalProperties: false`. F3's implementation notes forbid injecting that
keyword ourselves, since fabricating violations would inflate the very metric claim C2 reports.
It is covered here by a hand-built strict spec, and
`test_no_real_schema_forbids_extra_properties` pins the reason so the synthetic case can be
retired the day a vendor adds it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema.validators import Draft7Validator, Draft202012Validator

from mcpr.guards import schema_guard
from mcpr.registry import load_registry
from mcpr.types import Registry, ToolCall, ToolSpec

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURES / "registry_min.json"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry(FIXTURE_PATH)


def _strict_spec() -> ToolSpec:
    """A tool whose schema really does forbid extra properties.

    Synthetic on purpose: no captured schema sets `additionalProperties: false`, and the guard
    must not add it. This is the only way to reach `unknown_property` without fabricating a
    violation against a real tool.
    """
    return ToolSpec(
        server="demo",
        name="strict_tool",
        qualified_name="demo.strict_tool",
        description="A tool whose schema forbids undeclared arguments.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True},
        effect="read",
    )


def _call(tool: str, arguments: dict) -> ToolCall:
    return ToolCall(tool=tool, arguments=arguments)


# (tool, arguments, expected action, expected code) - one entry per reachable SPEC.md 8.2 code.
CASES = [
    ("none", {}, "allow", "abstained"),
    ("github.search_code", {"query": "mcp"}, "allow", "valid"),
    ("github.search_everything", {"query": "mcp"}, "block", "unknown_tool"),
    ("github.search_code", {}, "block", "missing_required"),
    # Required field two levels down, inside an array of objects.
    (
        "filesystem.edit_file",
        {"path": "a", "edits": [{"oldText": "x"}]},
        "block",
        "missing_required",
    ),
    ("github.search_code", {"query": 123}, "block", "type_mismatch"),
    ("git.git_add", {"repo_path": ".", "files": "a.txt"}, "block", "type_mismatch"),
    # The `enum` case the F3 test plan names explicitly.
    ("github.search_code", {"query": "mcp", "order": "sideways"}, "block", "enum_violation"),
    (
        "github.issue_read",
        {"method": "delete", "owner": "a", "repo": "b", "issue_number": 1},
        "block",
        "enum_violation",
    ),
    # The `minimum` case the F3 test plan names explicitly, plus its maximum and minLength kin.
    ("github.search_code", {"query": "mcp", "page": 0}, "block", "constraint_violation"),
    ("github.search_code", {"query": "mcp", "perPage": 500}, "block", "constraint_violation"),
    ("fetch.fetch", {"url": ""}, "block", "constraint_violation"),
]


@pytest.mark.parametrize(("tool", "arguments", "action", "code"), CASES)
def test_schema_guard_matrix(
    registry: Registry, tool: str, arguments: dict, action: str, code: str
) -> None:
    decision = schema_guard.check(_call(tool, arguments), registry)
    assert (decision.action, decision.code) == (action, code)
    assert decision.layer == "schema"


def test_abstain_allows(registry: Registry) -> None:
    """`none` is the reserved abstention id; declining to act is never a violation."""
    decision = schema_guard.check(_call("none", {}), registry)
    assert decision.action == "allow"
    assert decision.code == "abstained"


def test_unknown_tool_blocks(registry: Registry) -> None:
    """A hallucinated tool must never reach the policy layer's effect derivation."""
    decision = schema_guard.check(_call("github.search_everything", {"q": "x"}), registry)
    assert decision.action == "block"
    assert decision.code == "unknown_tool"
    assert "github.search_everything" in decision.evidence


def test_not_object_blocks(registry: Registry) -> None:
    """Reachable only via `model_construct`, which is how F8 builds prediction rows in bulk."""
    call = ToolCall.model_construct(tool="filesystem.write_file", arguments=["a.txt", "x"])
    decision = schema_guard.check(call, registry)
    assert decision.action == "block"
    assert decision.code == "not_object"


def test_unknown_property_needs_additional_properties_false() -> None:
    """The one code no captured schema can produce, exercised against a synthetic strict tool."""
    registry = Registry(tools=[_strict_spec()])
    decision = schema_guard.check(_call("demo.strict_tool", {"query": "x", "extra": 1}), registry)
    assert decision.action == "block"
    assert decision.code == "unknown_property"


def test_no_real_schema_forbids_extra_properties(registry: Registry) -> None:
    """Pins why `unknown_property` needs a synthetic spec, so the workaround can be retired.

    The day a vendor ships `additionalProperties: false`, this fails and the synthetic case in
    `test_unknown_property_needs_additional_properties_false` can be replaced with a real tool.
    """
    forbidding = [
        spec.qualified_name
        for spec in registry.tools
        if "additionalProperties" in str(spec.input_schema)
    ]
    assert forbidding == []


# --- the non-blocking undeclared-argument note --------------------------------------------------


def test_extra_property_permitted_is_allow_and_lists_the_keys(registry: Registry) -> None:
    """An invented argument is legal here, so it is recorded rather than blocked.

    Blocking it would need `additionalProperties: false` injected into a schema that does not
    have it, which would manufacture the violations claim C2 counts.
    """
    call = _call("github.search_code", {"query": "mcp", "reason": "why", "confidence": 0.9})
    decision = schema_guard.extra_properties(call, registry)

    assert decision is not None
    assert decision.action == "allow"
    assert decision.code == "extra_property_permitted"
    assert decision.evidence == ["confidence", "reason"]


def test_extra_properties_is_silent_when_there_is_nothing_to_say(registry: Registry) -> None:
    assert (
        schema_guard.extra_properties(_call("github.search_code", {"query": "x"}), registry) is None
    )
    assert schema_guard.extra_properties(_call("none", {}), registry) is None
    assert schema_guard.extra_properties(_call("no.such_tool", {"a": 1}), registry) is None


def test_extra_properties_stays_quiet_when_the_schema_already_forbids_them() -> None:
    """`check` has already blocked with `unknown_property`; a second decision would double-count."""
    registry = Registry(tools=[_strict_spec()])
    call = _call("demo.strict_tool", {"query": "x", "extra": 1})
    assert schema_guard.extra_properties(call, registry) is None


# --- validator construction ---------------------------------------------------------------------


def test_schema_guard_caches_validator(registry: Registry) -> None:
    """Compiling a validator per call is the performance trap F3 calls out; assert it is avoided."""
    schema_guard.clear_validator_cache()
    call = _call("github.search_code", {"query": "mcp"})

    schema_guard.check(call, registry)
    schema_guard.check(call, registry)

    info = schema_guard.validator_cache_info()
    assert (info.misses, info.hits) == (1, 1)


def test_draft_is_taken_from_the_schema(registry: Registry) -> None:
    """Servers ship different drafts; the validator class must follow each schema's own $schema."""
    by_name = {spec.qualified_name: spec for spec in registry.tools}
    filesystem = by_name["filesystem.write_file"].input_schema
    github = by_name["github.search_code"].input_schema

    assert "draft-07" in filesystem["$schema"]
    assert "$schema" not in github

    assert isinstance(schema_guard._validator(filesystem), Draft7Validator)
    assert isinstance(schema_guard._validator(github), Draft202012Validator)


# --- determinism ---------------------------------------------------------------------------------


def test_first_error_is_deterministic(registry: Registry) -> None:
    """The reported code is a published number, so "the first error" needs a total order.

    Three simultaneous violations at different depths, submitted twice with the keys inserted in
    opposite orders. `best_match` would not guarantee this, and a naive `absolute_path` sort key
    raises TypeError once an array index and a property name land at the same position.
    """
    forward = ToolCall(
        tool="github.search_code",
        arguments={"query": 1, "order": "sideways", "page": 0},
    )
    backward = ToolCall(
        tool="github.search_code",
        arguments={"page": 0, "order": "sideways", "query": 1},
    )

    first = schema_guard.check(forward, registry)
    again = schema_guard.check(forward, registry)
    reordered = schema_guard.check(backward, registry)

    assert first.code == again.code == reordered.code
    assert first.evidence == again.evidence == reordered.evidence


def test_mixed_index_and_name_paths_do_not_raise(registry: Registry) -> None:
    """The sort key must stay type-uniform: an int index and a str key at one position.

    `list(e.absolute_path)` would compare 0 against 'dryRun' here and raise TypeError, taking the
    whole guard down into `guard_error` instead of reporting an honest schema violation.
    """
    call = ToolCall(
        tool="filesystem.edit_file",
        arguments={"path": "a", "edits": [{"oldText": "x"}, "not an object"], "dryRun": "yes"},
    )
    decision = schema_guard.check(call, registry)
    assert decision.action == "block"


def test_missing_required_at_depth_reports_the_json_path(registry: Registry) -> None:
    """Evidence has to say *where*, or a nested failure is undebuggable from a metrics file."""
    call = _call("filesystem.edit_file", {"path": "a", "edits": [{"oldText": "x"}]})
    decision = schema_guard.check(call, registry)

    assert decision.code == "missing_required"
    assert decision.evidence[0] == "$.edits[0]"


def test_root_errors_outrank_nested_ones(registry: Registry) -> None:
    """A missing required argument is more actionable than a type failure inside another one."""
    call = _call("filesystem.edit_file", {"edits": [{"oldText": 1, "newText": 2}]})
    decision = schema_guard.check(call, registry)

    assert decision.code == "missing_required"
    assert decision.evidence[0] == "$"
