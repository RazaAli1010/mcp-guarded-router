"""The shared vocabulary: every Pydantic model defined in SPEC.md 6.

This module imports nothing outside pydantic and the standard library. That is a hard
constraint, not a style preference - `src/mcpr/` must stay installable on a laptop with no
GPU-class dependency (SPEC.md 2.4), and this file is the one every other module imports.

No field, model, or JSON key here may be renamed: `qualified_name` is the primary key across
datasets, predictions, policy and metrics, and the JSON shapes are the on-disk contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ParseErrorCode = Literal[
    "invalid_json",
    "not_object",
    "missing_keys",
    "bad_types",
    "extra_text",
    "multiple_objects",
]

GuardAction = Literal["allow", "confirm", "block"]
GuardLayer = Literal["schema", "policy", "injection"]
Effect = Literal["read", "write", "destructive"]


class ToolSpec(BaseModel):
    """One entry of `schemas/registry.json` (SPEC.md 6.1).

    `input_schema`, `output_schema` and `annotations` are held verbatim as captured from MCP
    `tools/list`; only `qualified_name` and `effect` are derived.
    """

    model_config = ConfigDict(extra="forbid")

    server: str
    name: str
    qualified_name: str
    title: str | None = None
    description: str
    input_schema: dict
    output_schema: dict | None = None
    annotations: dict = {}
    effect: Effect


class ToolCall(BaseModel):
    """The single JSON object the router emits (SPEC.md 6.2).

    `tool` is a `qualified_name` or the reserved value `"none"` for abstention.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict


class ParseResult(BaseModel):
    """Outcome of `parse_router_output` (SPEC.md 6.2).

    `error_code` is reported as a distribution in the results (claim C2), so a failure must
    always record *why* it failed rather than collapsing to a single flag.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    call: ToolCall | None = None
    error_code: ParseErrorCode | None = None
    raw: str


class UntrustedBlock(BaseModel):
    """Tool-result content injected into a prompt's `# Context` section (SPEC.md 6.7).

    `content` is always post-`sanitize.normalise()`; raw tool output never enters a prompt.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    content: str
    truncated: bool
    signals: list[str] = []


class GuardDecision(BaseModel):
    """One layer's verdict (SPEC.md 6.6). `code` is a stable machine string, never prose."""

    model_config = ConfigDict(extra="forbid")

    layer: GuardLayer
    action: GuardAction
    code: str
    detail: str
    evidence: list[str] = []
    score: float | None = None


class GuardChainResult(BaseModel):
    """The full chain's verdict (SPEC.md 6.6).

    All three layers always run - no short-circuit - so per-layer statistics stay complete;
    `final_action` is the most restrictive result, precedence block > confirm > allow.
    """

    model_config = ConfigDict(extra="forbid")

    decisions: list[GuardDecision]
    final_action: GuardAction
    blocked_by: str | None = None


class DatasetRow(BaseModel):
    """One line of `data/labeled/{train,val,test}.jsonl` (SPEC.md 6.4).

    `teacher_agreed` is `source_tool == gold.tool`; rows where it is false are excluded from
    training unless `human_verified` resolves them.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    query: str
    tool_pool: list[str]
    gold: ToolCall
    source_tool: str
    teacher_model: str
    teacher_agreed: bool
    human_verified: bool
    difficulty: Literal["easy", "confusable", "abstain", "injected"]
    untrusted: list[UntrustedBlock] = []
    prompt_hash: str
    split: Literal["train", "val", "test"]


class PredictionParse(BaseModel):
    """The `parse` sub-object of a prediction row (SPEC.md 6.5).

    A strict subset of `ParseResult` - the prediction row carries `raw` and `pred` at the top
    level, so re-using `ParseResult` here would duplicate them.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    error_code: ParseErrorCode | None = None


class PredictionRow(BaseModel):
    """One line of `data/predictions/<run_id>/<model_tag>.jsonl` (SPEC.md 6.5).

    `pred` is None when parsing failed. `prompt_hash` lets F8 assert that a reconstructed
    eval prompt matches the one the row was generated from; a mismatch means train/eval
    drift and fails the run.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    model_tag: Literal["tuned", "base", "baseline_api", "lexical"]
    raw: str
    parse: PredictionParse
    pred: ToolCall | None = None
    prompt_hash: str
    latency_ms: float
    gen_tokens: int
