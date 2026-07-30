"""The SPEC.md 6 models are the on-disk contract; they must round-trip and forbid extras."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from mcpr import types as t
from mcpr.config import PROMPT_VERSION

TOOL_SPEC = t.ToolSpec(
    server="github",
    name="search_repositories",
    qualified_name="github.search_repositories",
    title=None,
    description="Search for GitHub repositories.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    output_schema=None,
    annotations={"readOnlyHint": True},
    effect="read",
)

TOOL_CALL = t.ToolCall(
    tool="github.search_repositories",
    arguments={"query": "mcp language:python stars:>100"},
)

SAMPLES: list[BaseModel] = [
    TOOL_SPEC,
    TOOL_CALL,
    t.ToolCall(tool="none", arguments={}),
    t.ParseResult(ok=True, call=TOOL_CALL, error_code=None, raw='{"tool":"x","arguments":{}}'),
    t.ParseResult(ok=False, call=None, error_code="invalid_json", raw="not json"),
    t.UntrustedBlock(source="github.issue_read", content="hello", truncated=False, signals=[]),
    t.RouterPrompt(
        system="You are a tool router.",
        user="# Tools\n{}\n\n# Request\nx\n",
        prompt_hash="0123456789abcdef",
        tool_names=["github.search_code"],
    ),
    t.GuardDecision(
        layer="policy",
        action="confirm",
        code="destructive_requires_confirmation",
        detail="write effect needs a confirmation token",
        evidence=["effect=write"],
        score=None,
    ),
    t.GuardChainResult(
        decisions=[
            t.GuardDecision(layer="injection", action="allow", code="clean", detail=""),
            t.GuardDecision(layer="schema", action="allow", code="valid", detail=""),
            t.GuardDecision(layer="policy", action="block", code="path_escape", detail="../etc"),
        ],
        final_action="block",
        blocked_by="policy",
    ),
    t.DatasetRow(
        id="q_000123",
        query="which repos under vercel mention 'edge runtime' in their code?",
        tool_pool=["github.search_code", "github.search_repositories"],
        gold=t.ToolCall(tool="github.search_code", arguments={"q": "edge runtime org:vercel"}),
        source_tool="github.search_code",
        teacher_model="some-teacher",
        teacher_agreed=True,
        human_verified=False,
        difficulty="confusable",
        untrusted=[],
        prompt_hash="0123456789abcdef",
        split="test",
    ),
    t.PredictionParse(ok=True, error_code=None),
    t.PredictionRow(
        id="q_000123",
        model_tag="tuned",
        raw='{"tool":"github.search_code","arguments":{}}',
        parse=t.PredictionParse(ok=True, error_code=None),
        pred=t.ToolCall(tool="github.search_code", arguments={}),
        prompt_hash="0123456789abcdef",
        latency_ms=41.2,
        gen_tokens=37,
    ),
]


@pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
def test_round_trips_through_json(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


@pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
def test_extra_is_forbidden(model: BaseModel) -> None:
    assert model.model_config.get("extra") == "forbid"


def test_tool_call_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        t.ToolCall(tool="github.search_code", arguments={}, confidence=0.9)


def test_tool_call_rejects_unknown_key_from_json() -> None:
    raw = '{"tool":"github.search_code","arguments":{},"reasoning":"because"}'
    with pytest.raises(ValidationError):
        t.ToolCall.model_validate_json(raw)


def test_abstention_is_representable() -> None:
    """`none` is the reserved abstention id and must never be a registry qualified_name."""
    call = t.ToolCall(tool="none", arguments={})
    assert call.tool == "none"
    assert TOOL_SPEC.qualified_name != "none"


def test_parse_result_records_why_it_failed() -> None:
    """The error_code distribution is a reported result (claim C2), so it cannot be optional."""
    codes = {
        "invalid_json",
        "not_object",
        "missing_keys",
        "bad_types",
        "extra_text",
        "multiple_objects",
    }
    for code in codes:
        result = t.ParseResult(ok=False, error_code=code, raw="")
        assert result.error_code == code
    with pytest.raises(ValidationError):
        t.ParseResult(ok=False, error_code="something_else", raw="")


def test_prediction_row_allows_a_failed_parse() -> None:
    row = t.PredictionRow(
        id="q_1",
        model_tag="base",
        raw="I think you should use the search tool!",
        parse=t.PredictionParse(ok=False, error_code="extra_text"),
        pred=None,
        prompt_hash="deadbeefdeadbeef",
        latency_ms=12.0,
        gen_tokens=9,
    )
    assert row.pred is None


def test_prompt_version_type_tracks_the_constant() -> None:
    """`PromptVersion` is the closed set; `config.PROMPT_VERSION` is the live member of it.

    They live in different modules on purpose - `config.py` imports nothing from `mcpr` - so
    nothing but this assertion stops them drifting apart. Bumping the format means adding the
    new member here *and* moving the constant, and SPEC.md 6.3 says both invalidate every
    labelled dataset.
    """
    members = get_args(t.PromptVersion)
    assert PROMPT_VERSION in members
    assert members[-1] == PROMPT_VERSION


def test_dataset_row_rejects_an_unknown_split() -> None:
    with pytest.raises(ValidationError):
        t.DatasetRow(
            id="q_1",
            query="x",
            tool_pool=[],
            gold=TOOL_CALL,
            source_tool="github.search_repositories",
            teacher_model="m",
            teacher_agreed=True,
            human_verified=True,
            difficulty="easy",
            prompt_hash="h",
            split="holdout",
        )
