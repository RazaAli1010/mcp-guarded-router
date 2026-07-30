## F3 — Guardrail layers 1 and 2: schema validation and the destructive-operation gate

**Goal:** Every proposed tool call is validated against the real MCP JSON Schema before dispatch,
and every write or destructive call is stopped at a confirmation gate that cannot be bypassed.

**Depends on:** F2

**Runs on:** LAPTOP. Pure python, zero heavy deps, importable on Kaggle.

### Context digest

- `GuardDecision` / `GuardChainResult` shapes and the fixed chain order
  **injection → schema → policy** with "most restrictive wins" precedence
  `block > confirm > allow` — SPEC.md §6.6. All layers always run; no short-circuit.
- Layer 1 codes (SPEC.md §8.2): `unknown_tool, not_object, missing_required, unknown_property,
  type_mismatch, enum_violation, constraint_violation`. Validation via `jsonschema`, draft from
  the tool's `$schema`, default draft 2020-12. Any violation → `block`.
- Layer 2 rules (SPEC.md §8.3): `read` → allow; `write`/`destructive` → `confirm` with code
  `destructive_requires_confirmation`; path containment → `block` with code `path_escape`;
  plan lock → `confirm` with code `outside_plan`; confirmation token TTL
  `CONFIRM_TTL_SECONDS = 300`; **no auto-confirm flag exists**.
- `effect_for()` three-tier derivation — SPEC.md §8.1, implemented in F1's `registry.py`.
- `config/policy.toml` sections `[effects] [rules] [limits] [thresholds] [paths]` — created
  empty-ish in F0.
- Fail-closed principle (SPEC.md §3.7): an unhandled exception inside the chain yields
  `action="block", code="guard_error"`.
- `destructive_catch_rate` must be exactly `1.000` (SPEC.md §9.3, claim C3).

### Context deltas

- **`ConfirmToken` is added to `src/mcpr/types.py`**: `{token: str, call_hash: str,
  issued_at: float, expires_at: float}`. `call_hash` is sha256 of the canonicalised
  `{tool, arguments}` so a token cannot be replayed against a *different* call. Add to
  SPEC.md §6.6.
- **`config/policy.toml [effects]` is populated** in this feature with explicit overrides for
  every tool the derivation heuristic gets wrong. The populated table is a shared contract;
  add a pointer to it in SPEC.md §8.1.
- **`Plan` is added to `src/mcpr/types.py`**: `{allowed_servers: list[str],
  max_effect: Literal["read","write","destructive"], created_at: float}`. Add to SPEC.md §8.3.

### Scope

1. `src/mcpr/guards/__init__.py`:
   ```python
   class GuardChain:
       def __init__(self, registry, policy, injection_guard=None, plan: Plan | None = None): ...
       def check(self, call: ToolCall, untrusted: list[UntrustedBlock] | None = None) -> GuardChainResult
   ```
   Runs all layers, collects every `GuardDecision`, computes `final_action` by precedence, sets
   `blocked_by` to the code of the first `block` (in chain order) or `None`. Wraps each layer in
   `try/except Exception` → `GuardDecision(layer=…, action="block", code="guard_error",
   detail=repr(e))`. `injection_guard=None` (F4 has not landed yet) contributes no decisions.
2. `src/mcpr/guards/schema_guard.py::check(call, registry) -> GuardDecision`:
   - `call.tool == "none"` → `allow`, code `abstained`.
   - tool not in registry → `block`, `unknown_tool`.
   - `jsonschema` validation of `call.arguments` against `spec.input_schema`. Map the first
     `ValidationError` to one of the six codes by inspecting `error.validator`
     (`required`→`missing_required`, `additionalProperties`→`unknown_property`,
     `type`→`type_mismatch`, `enum`/`const`→`enum_violation`, everything else→
     `constraint_violation`). Put `error.json_path` and the validator value into `evidence`.
   - Build the validator once per tool and cache it (`functools.lru_cache` on the schema's
     sha256); constructing a validator per call is the obvious performance trap here.
3. `src/mcpr/guards/policy_guard.py::check(call, registry, policy, plan) -> GuardDecision`:
   - effect lookup via `registry.effect_for` with `policy["effects"]` overrides.
   - path containment: for each argument whose **name** is in
     `{"path","file_path","filepath","source","destination","dest","repository","repo_path",
     "directory","dir"}` *or* whose schema `description` matches `/absolute path|file path|
     directory/i`, resolve `Path(value).expanduser()`; if relative, resolve against
     `MCPR_SANDBOX_DIR`; then `os.path.realpath` and require
     `Path(real).is_relative_to(realpath(sandbox))`. Otherwise `block` / `path_escape` with the
     resolved path in `evidence`. Handle list-valued path arguments too.
   - plan lock: if `plan` is not `None` and (`spec.server not in plan.allowed_servers` or
     `effect_rank(effect) > effect_rank(plan.max_effect)`) → `confirm` / `outside_plan`.
   - otherwise: `read` → `allow` / `read_only`; `write`/`destructive` → `confirm` /
     `destructive_requires_confirmation`.
4. `src/mcpr/guards/confirm.py` — in-memory `ConfirmStore`:
   `issue(call) -> ConfirmToken`, `redeem(token: str, call: ToolCall) -> bool` (verifies the
   token exists, has not expired, has not been used, and `call_hash` matches), `purge()`.
   Single-use, TTL-bounded, hash-bound. Document plainly in the docstring that a real system
   would persist this and bind it to a user session — the in-memory store is a stated
   simplification, not an oversight.
5. Populate `config/policy.toml`:
   ```toml
   [effects]
   "github.create_or_update_file" = "write"
   "github.delete_file"           = "destructive"
   "github.push_files"            = "write"
   "github.merge_pull_request"    = "destructive"
   "github.issue_write"           = "write"
   "github.create_pull_request"   = "write"
   "github.fork_repository"       = "write"
   "github.unstar_repository"     = "destructive"
   "filesystem.write_file"        = "write"
   "filesystem.edit_file"         = "write"
   "filesystem.move_file"         = "destructive"
   "filesystem.create_directory"  = "write"
   "git.git_commit"               = "write"
   "git.git_reset"                = "destructive"
   "git.git_checkout"             = "destructive"
   ```
   Include only keys that exist in `schemas/registry.json`; the loader raises on an override for
   an unknown tool, so a typo cannot silently do nothing.
6. `mcpr guard check --tool <qn> --args '<json>'` — prints the full `GuardChainResult` as a
   table plus JSON. `mcpr guard audit` — walks the whole registry, prints the derived effect and
   the source of that derivation (`override` / `annotation` / `heuristic`) for every tool, and
   exits 1 if any tool's derivation source is `heuristic` **and** its effect is `destructive`
   (those must be pinned by an explicit override, because a heuristic must never be the only
   thing standing between the router and a delete).
7. `tests/fixtures/policy_cases.jsonl` — ≥ 40 hand-written `{call, expected_action,
   expected_code}` rows covering every code in both layers. This fixture is the C3 evidence and
   F8 consumes it for `destructive_catch_rate`.

### Out of scope

- Injection detection and untrusted-content sanitising — F4. `GuardChain` accepts an optional
  injection guard and works without one.
- Actually executing an approved call — F9.
- Computing `destructive_catch_rate` — F8 consumes the fixture from here.

### Implementation notes

- `jsonschema 4.26`: use `jsonschema.validators.validator_for(schema)` then `cls(schema)`, and
  `sorted(v.iter_errors(args), key=lambda e: list(e.absolute_path))` for a deterministic "first"
  error. `best_match` is *not* deterministic enough for a reported metric.
- Real MCP schemas often omit `additionalProperties`, which means unknown properties validate
  fine. Do **not** inject `additionalProperties: false` yourself — that would fabricate
  violations and inflate claim C2. Instead emit a separate non-blocking `GuardDecision` with
  code `extra_property_permitted` and `action="allow"` so the frequency is still measurable.
- `Path.is_relative_to` requires both paths resolved; a symlink inside the sandbox pointing
  outside must be caught, hence `os.path.realpath` on both sides. Test with an actual symlink
  created in a `tmp_path`.
- `effect_rank = {"read": 0, "write": 1, "destructive": 2}`.
- Timing: use `time.monotonic()` for TTL, `time.time()` only for display. Tests inject a clock.
- Layer 2 must never call a model, never read the network, and never consult the untrusted
  content. Its only inputs are the call, the registry, the policy file and the plan.

### Test plan

- `test_schema_guard_matrix` — one test per code, driven by `policy_cases.jsonl` plus targeted
  unit cases; includes a tool whose schema has an `enum` and one with `minimum`.
- `test_schema_guard_caches_validator` — two calls on the same tool build one validator
  (assert via `lru_cache.cache_info()`).
- `test_unknown_tool_blocks` — a hallucinated `github.search_everything` blocks.
- `test_abstain_allows` — `{"tool":"none"}` yields `allow`/`abstained`.
- `test_policy_read_allows` / `test_policy_write_confirms` / `test_policy_destructive_confirms`.
- `test_path_escape_dotdot` — `{"path": "../../etc/passwd"}` → `block`/`path_escape`.
- `test_path_escape_symlink` — real symlink in `tmp_path` pointing outside the sandbox →
  `block`/`path_escape`.
- `test_path_inside_sandbox_allows`.
- `test_plan_lock_other_server_confirms` — plan allows `["github"]`, call targets
  `filesystem.read_text_file` → `confirm`/`outside_plan`.
- `test_plan_lock_effect_escalation_confirms` — plan `max_effect="read"`, call is `write`.
- `test_confirm_token_single_use` / `test_confirm_token_expires` /
  `test_confirm_token_bound_to_call` (redeeming against a mutated `arguments` fails).
- `test_no_autoconfirm_exists` — grep the source tree for `auto_confirm|force|--yes` and assert
  no such flag reaches `ConfirmStore.redeem`. Crude, and worth it.
- `test_guard_error_fails_closed` — monkeypatch `schema_guard.check` to raise; chain returns
  `block`/`guard_error`.
- `test_all_destructive_have_overrides` — mirrors `mcpr guard audit`'s exit-1 condition.

### Verify

> **Note added during implementation.** `github.delete_file` is **not in the snapshot**. F1
> captured the GitHub server with `X-MCP-Readonly: true`, so all 23 `github.*` tools are `read`
> and there are no GitHub write/destructive tools to guard. The third command below therefore
> yields `block` / `unknown_tool`, not `confirm` — that is the correct answer, and it is left
> here unedited so the run is reproducible. The `confirm` and `outside_plan` paths are
> demonstrated by the two supplementary commands underneath. The same readonly capture is why
> only 7 of the 15 keys in Scope item 5's `[effects]` table exist.
>
> `mcpr guard check` exits 0 on `allow` and 1 on `confirm`/`block` (fail closed, SPEC.md 3.7),
> so commands 3 and 4 exit non-zero by design.

```bash
mcpr guard audit
mcpr guard check --tool github.search_code --args '{"query":"mcp"}'          # allow
mcpr guard check --tool github.delete_file --args '{"owner":"a","repo":"b","path":"x","branch":"main","message":"m"}'   # confirm
mcpr guard check --tool filesystem.read_text_file --args '{"path":"../../etc/passwd"}'   # block / path_escape

# Supplementary: the two paths the block above cannot reach with this snapshot.
mcpr guard check --tool git.git_reset --args '{"repo_path":"."}'             # confirm
mcpr guard check --tool filesystem.read_text_file --args '{"path":"a.txt"}' \
                 --plan-servers github --plan-max-effect read                # confirm / outside_plan

pytest -q tests/test_guards_schema.py tests/test_guards_policy.py tests/test_guards_chain.py
python -c "
import json; from mcpr.io import read_jsonl
rows=list(read_jsonl('tests/fixtures/policy_cases.jsonl'))
print(len(rows),'cases'); assert len(rows)>=40"
```

### Acceptance criteria

- [ ] Every one of the seven Layer-1 codes and the four Layer-2 codes is produced by at least
      one passing test.
- [ ] `mcpr guard audit` exits 0 and shows a derivation source for all tools; no tool has
      effect `destructive` derived from the heuristic alone.
- [ ] Path containment blocks both `..` traversal and a symlink escape, verified with a real
      symlink in a temp directory.
- [ ] A confirmation token is single-use, expires after `CONFIRM_TTL_SECONDS`, and cannot be
      redeemed against a modified `arguments` dict.
- [ ] No code path anywhere in the repo can execute a `write` or `destructive` call without a
      redeemed token — demonstrated by `test_no_autoconfirm_exists` plus reading `GuardChain`.
- [ ] `tests/fixtures/policy_cases.jsonl` has ≥ 40 rows and every row's expectation passes.
- [ ] An exception raised inside any layer yields `final_action == "block"`.
