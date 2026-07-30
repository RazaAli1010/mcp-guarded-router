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

#: Which of SPEC.md 8.1's three tiers decided a tool's effect. `mcpr guard audit` exits 1 on any
#: tool that is `destructive` by `heuristic` alone: a verb guess must never be the only thing
#: standing between the router and a delete.
EffectSource = Literal["override", "annotation", "heuristic"]

#: The closed set of prompt-format versions. The live value is `config.PROMPT_VERSION`, which
#: stays a plain `str` so that `config.py` need not import this module. Declaring the version as
#: a type rather than a bare string is what turns SPEC.md 6.3's freeze rule into something a type
#: checker enforces: adding `"v2"` here fans out to every file that must be regenerated.
PromptVersion = Literal["v1"]


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


class Registry(BaseModel):
    """The whole of `schemas/registry.json` (SPEC.md 6.1).

    `tools` is sorted by `qualified_name`; `snapshot.py` guarantees the ordering on write and
    `registry.load_registry` does not re-sort on read, so a hand-edited file that is out of
    order is a test failure rather than something silently repaired.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    tools: list[ToolSpec]


class ServerConfig(BaseModel):
    """One resolved `[servers.<id>]` table of `config/servers.toml` (SPEC.md 5).

    "Resolved" means every launch-time substitution has already happened: `${MCPR_*}` header
    interpolation and the `sandbox_arg` / `<MCPR_SANDBOX_DIR>` replacement. Nothing downstream
    of `mcp_client.load_server_configs` needs to know those placeholders existed.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    transport: Literal["stdio", "http"]
    command: str = ""
    args: list[str] = []
    url: str = ""
    headers: dict[str, str] = {}
    enabled: bool = True
    sandbox_arg: bool = False


class ToolResult(BaseModel):
    """The outcome of one MCP `tools/call` (F1 scope 1).

    `content_text` is the concatenation of the text blocks in the MCP `content` list; non-text
    blocks (image, audio, resource) are summarised rather than dropped silently, because a
    tool result that arrives as an image still has to be visible to the F4 injection guard.
    """

    model_config = ConfigDict(extra="forbid")

    content_text: str
    is_error: bool
    structured: dict | None = None


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


class RouterPrompt(BaseModel):
    """One rendered router prompt (SPEC.md 6.3).

    `prompt_hash` is `sha256(system + "\\n" + user)[:16]`, so it moves when the tool order moves.
    That is deliberate: F8 recomputes it from the stored row and fails the run on a mismatch,
    which is what catches drift between the training render and the evaluation render.

    `tool_names` is the *rendered* order - after `build_router_prompt`'s seed shuffle - not the
    sorted order `sample_tool_pool` returns. F6 must store it verbatim as `DatasetRow.tool_pool`;
    re-sorting it would make that reconstruction, and therefore the drift check, impossible.
    """

    model_config = ConfigDict(extra="forbid")

    system: str
    user: str
    prompt_hash: str
    tool_names: list[str]


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


class ConfirmToken(BaseModel):
    """One outstanding confirmation for a write or destructive call (SPEC.md 6.6, F3 delta).

    `call_hash` binds the token to one exact call, so a token issued for a harmless edit cannot
    be replayed against a different one. It is the sha256 of the *raw* `{tool, arguments}` under
    `json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":"))` - deliberately not
    `parse.canonicalise_arguments`, which is a lossy metric function. See `guards/confirm.py`.

    **Both timestamps are `time.monotonic()` readings, not wall-clock.** They are meaningless as
    dates and must never be serialised to disk or shown to a user; `time.time()` is for display
    only (F3 implementation notes). A monotonic clock cannot be moved backwards by an NTP step or
    a daylight-saving change, which is the whole point for a TTL that gates a destructive action.
    """

    model_config = ConfigDict(extra="forbid")

    token: str
    call_hash: str
    issued_at: float
    expires_at: float


class Plan(BaseModel):
    """The out-of-band lock taken from the first routing decision on the clean user query
    (SPEC.md 8.3, F3 delta).

    Any later call targeting a server outside `allowed_servers`, or with an effect ranking above
    `max_effect`, is forced to `confirm`. This is the layer that does not depend on *detecting*
    anything, so it is the one that survives a successful prompt injection (SPEC.md 13 D5).

    Unlike `ConfirmToken`, `created_at` is `time.time()`: it is provenance recorded alongside a
    run, never a deadline, and nothing compares it against a clock.
    """

    model_config = ConfigDict(extra="forbid")

    allowed_servers: list[str]
    max_effect: Effect
    created_at: float


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
