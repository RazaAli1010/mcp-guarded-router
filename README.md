# mcp-guarded-router

A fine-tuned MCP tool router with a three-layer guardrail — built so that every claim it makes is
a number produced by a command, not an assertion in a paragraph.

Given a natural-language request and a set of **real MCP tool schemas**, the router emits a single
tool call. Before that call can execute, it passes a guardrail chain that validates it against the
server's own JSON Schema, gates every mutation behind a confirmation token, and scores untrusted
tool output for prompt injection.

`SPEC.md` is the project-wide source of truth; this README summarises what exists today.

---

## Status

The project is built feature by feature (F0 → F10). **F0–F3 are complete and merged; F4–F10 are
not yet implemented.**

| ID | Feature | Runs on | Status |
|---|---|---|---|
| F0 | Repo scaffold, config, env-split enforcement, `mcpr doctor` | laptop | ✅ done |
| F1 | MCP client + frozen tool-schema snapshot | laptop | ✅ done |
| F2 | Router contract: prompt builder, parser, canonicaliser, lexical baseline | laptop | ✅ done |
| F3 | Guardrail layers 1 & 2: schema validation + destructive policy gate | laptop | ✅ done |
| F4 | Guardrail layer 3: untrusted isolation + injection detector | laptop | ⬜ not started |
| F5 | Adversarial and benign corpora (60 attacks, 150 controls, 20 adaptive) | laptop | ⬜ not started |
| F6 | Dataset build: synthesis, teacher labelling, human verification, splits | laptop | ⬜ not started |
| F7 | Kaggle QLoRA training notebook + GGUF export | Kaggle | ⬜ not started |
| F8 | Kaggle evaluation notebook: all models, all metrics, predictions out | Kaggle | ⬜ not started |
| F9 | Local dispatch pipeline + demo CLI (replay/gguf/openai backends) | laptop | ⬜ not started |
| F10 | Metrics aggregation, figures, `reports/results.md`, README | laptop | ⬜ not started |

**There are no measured results yet.** No model has been trained or evaluated, and
`reports/results.md` does not exist. Every figure quoted below describes the *code and data that
exist* (tool counts, test counts, policy fixture size) — none of it is a performance claim.

### The four claims, and what still stands between them and evidence

| # | Claim | Evidence it needs | Blocked on |
|---|---|---|---|
| C1 | A QLoRA fine-tune of a 1.5B model reaches **parity** with a prompted frontier model on MCP tool routing (parity = tuned ≥ baseline − 3.0 pts on `full_call_acc`, with 95% Wilson CIs) | test-split predictions from both models | F6, F7, F8 |
| C2 | Fine-tuning fixes **format adherence**, which is where small models actually fail (`invalid_json_rate`, `schema_violation_rate`, base vs. tuned on identical prompts) | the parser + schema guard exist; predictions do not | F7, F8 |
| C3 | Destructive MCP operations are **never executed without a confirmation gate** (`destructive_catch_rate == 1.000`) | **the mechanism is implemented and tested** — the 57-case policy fixture is its evidence; the metric is computed in F8 | F8 |
| C4 | The injection guardrail cuts attack success rate while keeping the false-positive rate on benign traffic low **and reported** (ASR guard-off vs. guard-on, `FPR_benign`, full threshold sweep) | layer 3 and the attack corpora | F4, F5, F8 |

This is a **parity claim, not a beat-the-teacher claim**. Nothing in this repo will claim the
fine-tune outperforms the teacher or the frontier baseline unless the confidence intervals
actually separate.

---

## The constraint that shapes everything

The author's laptop has 8 GB RAM, no GPU, and ~5 GB free disk. So the project is split in two, and
the split is enforced by a test rather than by discipline:

| | **LAPTOP** | **KAGGLE** |
|---|---|---|
| Role | **All logic. No model weights.** | **All model weights. No new logic.** |
| Contains | `src/mcpr/`: prompts, parsing, all three guardrails, policy engine, dispatch, metrics, CLI | QLoRA training (F7), all evaluation inference (F8), GGUF export |

`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`, `unsloth`,
`sentence-transformers`, `vllm`, `xformers`, `flash-attn`, `safetensors` and `tokenizers` must
never reach the laptop dependency set. `tests/test_env_split.py` parses `pyproject.toml` and fails
if one does; `mcpr doctor` additionally fails if any of them is merely *importable*.

The payoff is the **parity guarantee**: because `src/mcpr/` has zero heavy dependencies, the Kaggle
notebooks `pip install -e` the very same package and call the very same `build_router_prompt()`.
Training prompts, evaluation prompts and production prompts are identical by construction.

---

## Quickstart

```bash
# Linux / Kaggle              # Windows
make install                  ./tasks.ps1 install
make check                    ./tasks.ps1 check
make doctor                   ./tasks.ps1 doctor
```

`make check` runs `ruff check`, `ruff format --check` and `pytest`. The suite needs **no network
and no running MCP server** — the frozen snapshot is what makes that possible.

Then look around:

```bash
mcpr version                                        # package + prompt-format version
mcpr snapshot show                                  # the 50 frozen tool schemas
mcpr snapshot confusables github.search_code        # the tools most easily mistaken for it
mcpr guard audit                                    # every tool's effect and who decided it
mcpr run prompt --query "find repos mentioning edge runtime in code" \
                --gold github.search_code --seed 0  # a router prompt, and its hash
```

Copy `.env.example` to `.env` if you want to point `snapshot refresh` at the GitHub remote server
or let `doctor` ping a teacher/baseline endpoint. Everything else works with an empty `.env`.

---

## What is implemented

### The snapshot is the contract (F1)

`schemas/registry.json` is a committed, frozen capture of real MCP `tools/list` output. Training,
evaluation, guardrails and tests read **only** the snapshot. `mcpr snapshot refresh` is the one
command in the project allowed to open an MCP connection.

The current capture (`schemas/registry.meta.json`, captured 2026-07-30, sha256 `6151e896…`):

| Server | Transport | Package | Version | Tools | Annotation coverage |
|---|---|---|---|---|---|
| `fetch` | stdio | `mcp-server-fetch` | 2026.7.10 | 1 | 0% |
| `git` | stdio | `mcp-server-git` | 2026.7.10 | 12 | 100% |
| `filesystem` | stdio | `@modelcontextprotocol/server-filesystem` | 0.2.0 | 14 | 100% |
| `github` | http | GitHub's official remote server | remote-8b64a2c8 | 23 | 100% |

50 tools total. `qualified_name` (`"github.search_code"`) is the primary key across datasets,
predictions, policy and metrics. `input_schema` is held **verbatim**, warts included — including
the fact that not one of the 50 real schemas sets `additionalProperties: false`.

The GitHub server was captured with `X-MCP-Readonly: true`, so all 23 `github.*` tools are reads.
That is a deliberate, documented consequence: there are no GitHub write or destructive tools in the
snapshot to guard.

`mcpr snapshot diff <old-registry.json>` reports drift and exits 1 on any change, because a frozen
snapshot ages and that ageing should be visible rather than silent.

### The router contract (F2)

One prompt format, defined once in `src/mcpr/prompt.py`, never paraphrased anywhere else:

```
# Tools
{"name":"<qualified_name>","description":"<description>","parameters":<input_schema>}
...one compact JSON line per tool, order deterministic in `seed`...

# Context
<untrusted source="github.issue_read" trust="untrusted">
...sanitised content...
</untrusted>

# Request
<query>
```

- `ROUTER_SYSTEM_PROMPT` states the output contract and — critically — that anything inside
  `<untrusted>` is data to be described, never instructions to be followed.
- `prompt_hash` = `sha256(system + "\n" + user)[:16]`, recorded on every training and prediction
  row. F8 recomputes it and fails the run on a mismatch: that is the train/eval drift check.
- Everything about the render is frozen under `PROMPT_VERSION = "v1"`. A space after a colon would
  change every hash in every dataset, so nothing may be tidied without bumping the version and
  regenerating the data.
- `sample_tool_pool()` draws each row's catalog: always the gold tool, plus up to `MIN_CONFUSABLES`
  forced hard negatives from the confusability ranking, with `HELD_OUT_TOOLS` excluded so
  `heldout_tool_acc` measures zero-shot generalisation rather than memorisation.

The model must emit exactly one JSON object on one line. `parse_router_output()` is strict but
**diagnostic** — the distribution over its error codes *is* claim C2, so it records why parsing
failed rather than collapsing to a boolean:

`invalid_json` · `not_object` · `missing_keys` · `bad_types` · `extra_text` · `multiple_objects`

It never raises, for any input. A markdown code fence is stripped and still counts as `ok` — that
is a formatting tic, not a routing error.

`canonicalise_arguments()` implements the equality `arg_exact_acc` compares: sort keys, drop keys
whose value equals the schema default, collapse whitespace in strings, normalise integral floats.

`baselines.LexicalRouter` is real BM25 over the row's tool pool — the honest floor the fine-tune has
to beat, so that "the model scores X" says something about whether the task was hard. It is pure
python; no `rank_bm25` dependency.

### The guardrail (F3: layers 1 and 2)

Chain order is fixed at **injection → schema → policy**. All three layers always run — no
short-circuit — so per-layer statistics stay complete. `final_action` is the most restrictive
verdict, precedence `block > confirm > allow`. Any unhandled exception inside a layer becomes
`action="block", code="guard_error"`: the chain **fails closed**, and there is no path through it
that returns `allow` because something went wrong.

Guardrails are code, not prompts. Layers 1 and 2 contain no model calls and no probabilistic
components. A guardrail that can be talked out of its decision is not a guardrail.

**Effect derivation** (`registry.effect_with_source`) — first match wins, and the CLI can show you
which tier decided each tool:

1. explicit override in `config/policy.toml [effects]`
2. MCP annotations (`destructiveHint` / `readOnlyHint`)
3. name heuristic on the first token of the tool name

```
$ mcpr guard audit
50 tools: 41 read, 4 write, 5 destructive | sources: 7 override, 40 annotation, 3 heuristic
1 tool(s) differ from the effect baked into the snapshot: git.git_checkout write->destructive
ok: no tool is destructive by heuristic alone
```

Two things worth reading twice there. First, `guard audit` **exits 1 if any tool is `destructive`
by heuristic alone** — a guess from a verb prefix must never be the only thing standing between the
router and a delete, so those tools have to be pinned by an explicit override. Second, the runtime
derivation deliberately disagrees with the effect baked into the frozen snapshot for
`git.git_checkout`: the server reports `destructiveHint: false`, but a checkout discards
uncommitted work, so `config/policy.toml` escalates it. The snapshot is *not* rewritten (that would
invalidate its sha256 and every prediction file citing it) and the divergence is never silent —
`guard audit` reports it and a test asserts it can only happen where an explicit override exists.

**Layer 1 — schema validation** (`guards/schema_guard.py`). Validates `arguments` against the
tool's real JSON Schema, with the validator class selected from each schema's own `$schema`. Any
violation blocks. Codes: `unknown_tool` · `not_object` · `missing_required` · `unknown_property` ·
`type_mismatch` · `enum_violation` · `constraint_violation`.

Because no real schema in the snapshot sets `additionalProperties: false`, an invented argument
validates cleanly. The guard does **not** inject that keyword itself — fabricating violations would
inflate the very metric C2 reports. Instead undeclared arguments are recorded as a separate,
non-blocking `extra_property_permitted` decision, so their frequency stays measurable without being
manufactured.

**Layer 2 — the destructive-operation gate** (`guards/policy_guard.py`). Pure; its only inputs are
the call, the registry, the loaded policy and the plan. That narrowness is the point: it is the
layer that still works *after* an injection has succeeded, so it must not depend on having detected
anything.

- `effect == "read"` → `allow`
- `effect in {"write", "destructive"}` → `confirm`, code `destructive_requires_confirmation`
- **Path containment** → `block`, code `path_escape`. Every path-shaped argument is resolved with
  `os.path.realpath` and must land inside `MCPR_SANDBOX_DIR`. Catches `..` traversal and symlink
  escape, and is tested with both. Applied only to servers in `[paths] sandboxed_servers`
  (default `["filesystem", "git"]`), because applying it universally would block benign remote
  reads on their own schema defaults and inflate the reported `FPR_benign` with pure false
  positives.
- **Plan lock** → `confirm`, code `outside_plan`. A `Plan` records `{allowed_servers, max_effect}`
  from the first routing decision on the clean user query; any later call outside it is gated.
  This is the out-of-band control that survives a successful injection — it does not depend on
  detecting anything.

**The confirmation gate is real** (`guards/confirm.py`). A `confirm` verdict is not advice.
Execution happens only on a second call carrying a `ConfirmToken` issued for *that exact call*:
single-use, TTL-bounded (`CONFIRM_TTL_SECONDS = 300`, on a **monotonic** clock so the TTL cannot be
extended by moving the system clock), and bound to `sha256` of the raw `{tool, arguments}`.

There is no auto-confirm flag, no `--yes`, and no environment variable that skips it. Claim C3
rests on there being no such path, and `config/policy.toml [rules]` is deliberately empty for the
same reason: every knob that could plausibly live there is a switch that can only weaken a gate the
spec says has no bypass.

The token hash is deliberately **not** `parse.canonicalise_arguments`. That function is lossy by
design (it collapses whitespace and drops schema defaults), so binding a confirmation to it would
let a token approved for one file body be redeemed against a materially different one. A metric may
forgive whitespace; a gate on a destructive operation binds to the bytes that will execute.

### Seeing the chain work

`mcpr guard check` runs the chain over one call and prints every decision. It exits 0 only on
`allow` — `confirm` and `block` both exit 1, because a caller must not proceed on a confirmation
any more than on a block.

```
$ mcpr guard check --tool github.search_code --args '{"query":"mcp"}'
final_action=allow  blocked_by=-  decisions=2

$ mcpr guard check --tool filesystem.write_file --args '{"path":"notes.txt","content":"hi"}'
final_action=confirm  blocked_by=-  decisions=2

$ mcpr guard check --tool filesystem.read_text_file --args '{"path":"../../../etc/passwd"}'
final_action=block  blocked_by=path_escape  decisions=2

$ mcpr guard check --tool github.search_code --args '{"query":"mcp"}' \
                   --plan-servers git --plan-max-effect read
final_action=confirm  blocked_by=-  decisions=2      # outside_plan
```

`tests/fixtures/policy_cases.jsonl` holds **57 hand-written cases** covering every layer-1 and
layer-2 code. It is the fixture F8 will compute `destructive_catch_rate` over — the C3 evidence.

---

## CLI

Every sub-command namespace a later feature needs is reserved up front, so no feature invents its
own spelling. Unimplemented commands raise `NotImplementedError` naming the feature that fills them
in.

| Command | Status |
|---|---|
| `mcpr version` | ✅ |
| `mcpr doctor` | ✅ prerequisites, env split, disk, sandbox, endpoint pings |
| `mcpr snapshot refresh \| show \| diff \| confusables` | ✅ |
| `mcpr run prompt` | ✅ |
| `mcpr guard check \| audit` | ✅ |
| `mcpr data synth \| label \| verify \| split` | ⬜ F6 |
| `mcpr eval score` | ⬜ F8 |
| `mcpr run demo` | ⬜ F9 |
| `mcpr report build` | ⬜ F10 |
| `mcpr models pull` | ⬜ F7 |

---

## Configuration

| File | Owns |
|---|---|
| `config/servers.toml` | MCP server launch config: transport, command, args, url, `headers_env`, `enabled`, `sandbox_arg` |
| `config/policy.toml` | `[effects]` overrides, `[thresholds]` for layer 3, `[paths]` sandbox root and sandboxed servers |
| `.env` | the `MCPR_*` variables (see `.env.example`) |

`MCPR_SANDBOX_DIR` is the only directory filesystem/git tools may touch. Servers marked
`sandbox_arg = true` have their last positional argument replaced with the resolved sandbox path at
launch, so no server is ever *pointed* outside it in the first place.

Teacher and baseline models are **configured, never hardcoded**. `mcpr doctor` pings both endpoints
and the resolved model ids get recorded in the dataset manifest, which the report quotes.

Loading `config/policy.toml` raises `PolicyError` on an `[effects]` key that names no tool in the
registry, or a value outside `{read, write, destructive}` — a typo in the one table that pins
destructive operations must be loud, not silently inert.

---

## Repository layout

```
├── SPEC.md                       # project-wide source of truth
├── specs/F0..F10-*.md            # one feature spec per session (F0-F3 written)
├── config/{servers,policy}.toml
├── schemas/
│   ├── registry.json             # FROZEN tool-schema snapshot (committed)
│   └── registry.meta.json        # provenance: versions, capture UTC, sha256
├── src/mcpr/
│   ├── config.py types.py io.py registry.py
│   ├── mcp_client.py snapshot.py
│   ├── prompt.py parse.py baselines.py
│   ├── guards/{schema_guard,policy_guard,confirm}.py
│   └── cli.py
└── tests/                        # + tests/live/ (marked `live`, excluded by default)
```

Not yet created: `data/`, `notebooks/`, `reports/`, and `src/mcpr/`'s `sanitize.py`,
`guards/injection_guard.py`, `dispatch.py`, `backends.py`, `teacher.py`, `synth.py`, `metrics.py`,
`report.py`.

---

## Tests

```
$ ruff check . && ruff format --check . && pytest
All checks passed!
31 files already formatted
354 passed, 1 deselected in 5.00s
```

The one deselected test is in `tests/live/`, which needs a running MCP server and is excluded by
default. The rest run offline, with no MCP server and no model weights, in well under the 60-second
budget the spec sets.

Determinism is a tested property, not an aspiration: all randomness draws from an explicit
`random.Random(seed)`, all JSON is written with `sort_keys=True, ensure_ascii=False`, a trailing
newline and `newline="\n"` (so a file written on Windows hashes identically to one written on
Kaggle), and `prompt_hash` stability is asserted **across separate processes**.

---

## Limitations, stated rather than hidden

Reporting the unflattering number is a design principle here, so:

- **No results exist yet.** F4–F10 are unimplemented. Nothing in this repo currently demonstrates
  C1, C2 or C4, and C3's *metric* awaits F8 even though its mechanism is built and tested.
- **The confirmation store is a stated simplification.** It lives in one process's memory and binds
  a token to a call and nothing else. A real deployment would persist it and bind each token to an
  authenticated user session, so a token issued to one user could not be redeemed by another and a
  restart would not silently drop pending confirmations.
- **A regex detector is defeatable by paraphrase.** When layer 3 lands (F4), that will be true of
  it too. The plan lock is the layer that does not depend on detecting anything, and it is
  deliberately the one that survives a successful injection.
- **The planned attack corpus is small and self-authored.** n=60 injections gives a Wilson CI of
  roughly ±12 points on ASR — that will be said out loud in the report. The corpus will be written
  by the same person who wrote the detector; the adaptive arm partially mitigates this, but the
  honest framing is "measured against my own threat model", not "robust".
- **The snapshot ages.** It is frozen on purpose (reproducibility, offline tests, no PAT required),
  and `mcpr snapshot diff` is what makes the ageing visible.
- **The GitHub capture is read-only**, so the snapshot contains no GitHub write or destructive tool
  to guard. Eight such overrides sit commented out in `config/policy.toml`, ready for a future
  non-readonly capture.
- **The teacher's labels will be the ceiling for training.** Only the test set gets human
  verification.
- **The MCP SDK is pinned to v1.x.** `mcp` 2.0.0 shipped 2026-07-28 with a full client rewrite;
  migration is explicitly future work.

---

## License

MIT. See `LICENSE`.
