"""The parser must be diagnostic and must never raise (SPEC.md 6.2, 3.7).

The error_code distribution *is* claim C2, so a wrong code is not a cosmetic bug: it moves a
reported number. The matrix below is the contract - one case per code, plus the shapes real
small models actually emit.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from mcpr.config import SEED
from mcpr.parse import canonicalise_arguments, parse_router_output
from mcpr.registry import get, load_registry
from mcpr.types import ParseResult, Registry, ToolSpec

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURES / "registry_min.json"

HAPPY = '{"tool":"github.search_code","arguments":{"query":"edge runtime"}}'

#: (raw, expected ok, expected error_code). Every ParseErrorCode appears at least once.
CASES = [
    # --- the happy paths -----------------------------------------------------------------
    (HAPPY, True, None),
    ('{"tool":"none","arguments":{}}', True, None),
    (f"```json\n{HAPPY}\n```", True, None),
    (f"```\n{HAPPY}\n```", True, None),
    (f"```json\n{HAPPY}", True, None),  # truncated at MAX_GEN_TOKENS
    (f"```{HAPPY}```", True, None),
    ('{"tool":"github.search_code","arguments":{},"reasoning":"it fits"}', True, None),
    # --- extra_text ----------------------------------------------------------------------
    (f"{HAPPY}\nI chose this because it searches code.", False, "extra_text"),
    (f"Sure! {HAPPY}", False, "extra_text"),
    (f"I think {{maybe}} then {HAPPY}", False, "extra_text"),
    ("[1,2] and here is why", False, "extra_text"),
    # --- multiple_objects ----------------------------------------------------------------
    (f"{HAPPY}{HAPPY}", False, "multiple_objects"),
    (f"{HAPPY}\n{HAPPY}", False, "multiple_objects"),
    # --- not_object ----------------------------------------------------------------------
    ("[1,2]", False, "not_object"),
    ('"github.search_code"', False, "not_object"),
    ("42", False, "not_object"),
    ("null", False, "not_object"),
    # --- missing_keys --------------------------------------------------------------------
    ('{"tool":"github.search_code"}', False, "missing_keys"),
    ('{"arguments":{}}', False, "missing_keys"),
    ("{}", False, "missing_keys"),
    # --- bad_types -----------------------------------------------------------------------
    ('{"tool":1,"arguments":{}}', False, "bad_types"),
    ('{"tool":true,"arguments":{}}', False, "bad_types"),
    ('{"tool":null,"arguments":{}}', False, "bad_types"),
    ('{"tool":"x","arguments":[]}', False, "bad_types"),
    ('{"tool":"x","arguments":"query=y"}', False, "bad_types"),
    # --- invalid_json --------------------------------------------------------------------
    ('{"tool":"x","arguments":', False, "invalid_json"),
    ("", False, "invalid_json"),
    ("   \n\t ", False, "invalid_json"),
    ("I would use the search tool for this.", False, "invalid_json"),
    ("[" * 5000, False, "invalid_json"),  # RecursionError out of the C scanner
]


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry(FIXTURE_PATH)


@pytest.mark.parametrize(("raw", "ok", "code"), CASES, ids=lambda v: None)
def test_parse_matrix(raw: str, ok: bool, code: str | None) -> None:
    result = parse_router_output(raw)
    assert result.ok is ok, f"{raw!r} -> {result.error_code}"
    assert result.error_code == code, f"{raw!r}"
    assert (result.call is not None) is ok


def test_parse_preserves_raw_verbatim() -> None:
    """F8 reports fence frequency separately, which needs the original string, fence and all."""
    fenced = f"```json\n{HAPPY}\n```"
    result = parse_router_output(fenced)
    assert result.ok
    assert result.raw == fenced
    assert result.raw.lstrip().startswith("```")


def test_extra_keys_are_accepted() -> None:
    """A deliberate decision, not an oversight.

    ToolCall forbids extras so ToolCall(**value) would raise, violating "never raises";
    ParseErrorCode is a closed Literal with no code for this; and folding it into bad_types
    would poison the C2 distribution with something that is not a type error. A correct call
    carrying a stray key is a correct call, and F3's schema guard checks `arguments` anyway.
    """
    result = parse_router_output('{"tool":"github.search_code","arguments":{},"confidence":0.9}')
    assert result.ok
    assert result.call is not None
    assert result.call.tool == "github.search_code"


def test_abstention_parses() -> None:
    result = parse_router_output('{"tool":"none","arguments":{}}')
    assert result.ok
    assert result.call is not None
    assert result.call.tool == "none"


def test_parse_never_raises() -> None:
    """Fuzz floor. Seeded, because an unreproducible fuzz failure is not actionable."""
    rng = random.Random(SEED)
    for _ in range(500):
        raw = os.urandom(rng.randint(0, 400)).decode("latin-1")
        result = parse_router_output(raw)
        assert isinstance(result, ParseResult)
        assert result.raw == raw


def test_parse_is_not_quadratic_on_pathological_input() -> None:
    """The prefix probe is capped because json.decoder.errmsg rescans per raised error.

    Uncapped this takes tens of seconds and breaks the 60-second suite budget of SPEC.md 12.

    The inputs are built here rather than parametrised: pytest writes the parameter id into
    PYTEST_CURRENT_TEST, and Windows rejects an environment variable over 32767 characters.
    """
    for raw in ("{" * 200_000, "[" * 200_000, '{"a":' * 50_000):
        assert parse_router_output(raw).error_code == "invalid_json"


# --- canonicalisation ---------------------------------------------------------------------


PER_PAGE_SPEC = ToolSpec(
    server="demo",
    name="listing",
    qualified_name="demo.listing",
    description="A tool whose per_page argument defaults to 30.",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "string"},
            "per_page": {"type": "integer", "default": 30},
        },
    },
    effect="read",
)


def test_canonicalise() -> None:
    """F2 acceptance criterion 4, verbatim."""
    left = canonicalise_arguments({"a": 1, "b": "x  y", "per_page": 30}, PER_PAGE_SPEC)
    right = canonicalise_arguments({"b": "x y", "a": 1.0}, PER_PAGE_SPEC)
    assert left == right
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    assert left == {"a": 1, "b": "x y"}


def test_canonicalise_drops_real_schema_defaults(registry: Registry) -> None:
    """fetch.fetch carries three defaults; supplying them must be identical to omitting them."""
    spec = get("fetch.fetch", registry)
    supplied = {"url": "https://example.com", "max_length": 5000, "raw": False, "start_index": 0}
    assert canonicalise_arguments(supplied, spec) == {"url": "https://example.com"}
    assert canonicalise_arguments({"url": "https://example.com", "raw": True}, spec) == {
        "raw": True,
        "url": "https://example.com",
    }


def test_canonicalise_normalises_recursively() -> None:
    out = canonicalise_arguments({"z": {"b": 2, "a": " p   q "}, "y": [{"d": 1.0, "c": " e "}]})
    assert out == {"y": [{"c": "e", "d": 1}], "z": {"a": "p q", "b": 2}}
    assert list(out) == ["y", "z"]
    assert list(out["z"]) == ["a", "b"]


def test_canonicalise_keeps_list_order() -> None:
    """Array order is semantic - filesystem.edit_file applies its edits in sequence."""
    edits = {"edits": [{"oldText": "b"}, {"oldText": "a"}]}
    assert canonicalise_arguments(edits) == {"edits": [{"oldText": "b"}, {"oldText": "a"}]}


def test_canonicalise_does_not_conflate_booleans_with_numbers() -> None:
    """isinstance(True, int) is True in Python, so the bool branch must precede the numeric one.

    Asserted on the serialised form, not with `!=`: Python's `==` already considers
    `{"raw": True}` and `{"raw": 1}` equal, so a dict comparison cannot see the difference.
    What the rule actually protects is the JSON F8 compares - without the guard, `float(True)`
    would rewrite every boolean argument to `1.0`.
    """
    assert json.dumps(canonicalise_arguments({"raw": True})) == '{"raw": true}'
    assert json.dumps(canonicalise_arguments({"raw": 1})) == '{"raw": 1}'
    assert json.dumps(canonicalise_arguments({"n": 1.0})) == '{"n": 1}'


def test_canonicalise_is_pure() -> None:
    original = {"b": " x  y ", "a": 1.0, "n": [{"k": " v "}]}
    snapshot = json.dumps(original, sort_keys=True)
    canonicalise_arguments(original, PER_PAGE_SPEC)
    assert json.dumps(original, sort_keys=True) == snapshot


def test_canonicalise_tolerates_odd_schemas() -> None:
    """Vendor schemas are captured verbatim, so nothing about their shape is guaranteed."""
    bare = ToolSpec(
        server="s",
        name="n",
        qualified_name="s.n",
        description="d",
        input_schema={"type": "object"},
        effect="read",
    )
    assert canonicalise_arguments({"a": " x  y "}, bare) == {"a": "x y"}
    assert canonicalise_arguments({"a": 1}, None) == {"a": 1}
