## F0 — Repo scaffold, config, and environment-split enforcement

**Goal:** A laptop-installable package whose test suite proves that no GPU-class dependency can
leak into the laptop environment, plus a `mcpr doctor` command that reports whether every
external prerequisite is actually reachable.

**Depends on:** none

**Runs on:** LAPTOP only.

### Context digest

- Package name `mcpr`, layout per SPEC.md §5. CLI entry point `mcpr = "mcpr.cli:app"`.
- Laptop dependency list is exactly SPEC.md §4.1. Forbidden package list is SPEC.md §2.4:
  `torch transformers peft trl bitsandbytes accelerate datasets unsloth sentence-transformers
  vllm xformers flash-attn safetensors tokenizers`.
- Env vars per SPEC.md §7.2, prefix `MCPR_`. Constants per SPEC.md §7.1 live in
  `src/mcpr/config.py` as module-level uppercase names.
- Disk budget SPEC.md §2.5: venv ≤ 400 MB.
- Determinism principle SPEC.md §3.3: all JSON written `sort_keys=True, ensure_ascii=False`
  with a trailing newline.
- Global DoD SPEC.md §12: `pytest -q` under 60 s, no network, no MCP server.

### Context deltas

none

### Scope

1. `pyproject.toml` — hatchling backend, `[project]` block copied verbatim from SPEC.md §4.1,
   `[tool.ruff]` with `line-length = 100` and `select = ["E","F","I","UP","B","SIM"]`,
   `[tool.pytest.ini_options]` with `addopts = "-q -m 'not live'"` and
   `markers = ["live: needs a running MCP server or network"]`.
2. `src/mcpr/config.py` — every constant from SPEC.md §7.1 as a module-level literal, plus
   `class Settings(BaseSettings)`-style loading of the `MCPR_*` env vars via `python-dotenv`
   (`load_env() -> Settings`, a plain Pydantic model, no pydantic-settings dependency).
   Add `PROJECT_ROOT: Path` resolved from `__file__`, and
   `def resolve(p: str | Path) -> Path` that makes repo-relative paths absolute.
3. `src/mcpr/types.py` — all Pydantic models from SPEC.md §6: `ToolSpec`, `ToolCall`,
   `ParseResult`, `UntrustedBlock`, `GuardDecision`, `GuardChainResult`, plus
   `DatasetRow` (§6.4) and `PredictionRow` (§6.5). All with
   `model_config = ConfigDict(extra="forbid")`. This file has no imports outside pydantic +
   stdlib and is the shared vocabulary for every later feature.
4. `src/mcpr/io.py` — `read_jsonl(path) -> Iterator[dict]`, `write_jsonl(path, rows)`,
   `write_json(path, obj)`, `sha256_file(path) -> str`. All writes go through here so the
   determinism rule cannot be violated per-caller.
5. `config/servers.toml` and `config/policy.toml` — created with the structure later features
   fill in. `servers.toml` has one `[servers.<id>]` table per server with
   `transport = "stdio" | "http"`, `command`, `args`, `url`, `headers_env`, `enabled`.
   Populate all four servers from SPEC.md §4.4 with `enabled = false` except `fetch` and `git`.
   `policy.toml` gets `[effects]` (empty), `[rules]`, `[limits]`, `[thresholds]`
   (`injection_flag = 1.0`, `injection_block = 3.0`), `[paths] sandbox_root = "./sandbox"`.
6. `src/mcpr/cli.py` — Typer app with sub-commands stubbed as `NotImplementedError` except:
   `mcpr doctor` (implement now) and `mcpr version`. Sub-command namespaces to reserve so later
   features do not invent their own: `snapshot`, `data`, `guard`, `eval`, `run`, `report`,
   `models`.
7. `mcpr doctor` — prints a table (rich) with one row per check and PASS/WARN/FAIL:
   Python version in range · every laptop dep importable · **no forbidden package importable**
   · `uv` on PATH · `node`/`npx` on PATH (WARN, not FAIL — filesystem server is optional) ·
   `MCPR_SANDBOX_DIR` exists and is writable · free disk space on the repo's filesystem ≥ 2 GB ·
   teacher endpoint reachable (a 1-token completion) · baseline endpoint reachable ·
   `MCPR_GITHUB_PAT` present. Network checks are skipped with WARN when the env var is empty, so
   `doctor` never fails offline. Exits non-zero only on a FAIL.
8. `Makefile` — `install`, `check` (ruff check + ruff format --check + pytest),
   `doctor`, `clean`, and a placeholder `kaggle-bundle` target that currently errors with
   "implemented in F6". `.env.example`, `.gitignore` (ignore `models/`, `sandbox/`, `.env`,
   `dist/`, `data/predictions/`), and `CLAUDE.md` (five lines pointing at SPEC.md).

### Out of scope

- Any MCP connection or schema capture — F1.
- Prompt building, parsing — F2.
- Guardrail logic; `config/policy.toml` is created but its `[effects]` table stays empty — F3.
- `kaggle-bundle` implementation — F6.

### Implementation notes

- Verified 2026-07-29: `pydantic 2.13.4`, `jsonschema 4.26.0`, `typer 0.27.0`, `rich 15.0.0`,
  `mcp 1.29.0`, `pytest 9.1.1`, `ruff 0.16.0`. Pin ranges exactly as SPEC.md §4.1 writes them —
  in particular `mcp>=1.29,<2`, because an unbounded `mcp` now resolves to 2.0.0 (SPEC.md §13 D1).
- Typer 0.27 with `no_args_is_help=True` on the top-level app.
- The forbidden-package check in `doctor` must use `importlib.util.find_spec`, not a bare
  `import` inside `try`, so a partially-installed package is still detected.
- Do not add `pydantic-settings`; it is not in the dependency list and one more package is one
  more thing to break. Read env vars with `os.environ` + `dotenv.load_dotenv`.
- `write_json` must end the file with `\n`; several later diffs depend on it.

### Test plan

- `tests/test_env_split.py::test_no_forbidden_deps_declared` — parse `pyproject.toml` with
  `tomllib`, assert no name in the forbidden list appears in `project.dependencies` or in
  `optional-dependencies.dev`.
- `tests/test_env_split.py::test_no_forbidden_deps_importable` — `find_spec` returns `None`
  for every forbidden name. This is the test that fails loudly if someone runs
  `pip install torch` on the laptop.
- `tests/test_config.py` — every constant in SPEC.md §7.1 exists with the specified value;
  `resolve()` returns absolute paths; `HELD_OUT_TOOLS` has exactly 3 entries.
- `tests/test_types.py` — each model round-trips through `model_dump_json` →
  `model_validate_json`; `extra="forbid"` rejects an unknown key on `ToolCall`.
- `tests/test_io.py` — `write_jsonl` then `read_jsonl` round-trips; output bytes are stable
  across two writes of the same dict with keys inserted in different orders (determinism).

### Verify

```bash
make install && make check && mcpr doctor
```

`make check` must exit 0. `mcpr doctor` must print a table where the two forbidden-package rows
read PASS, and must exit 0 even with an empty `.env` (network rows show WARN).

### Acceptance criteria

- [ ] `pip install -e ".[dev]"` succeeds on Python 3.11–3.13 and the resulting venv is < 400 MB
      (`du -sh .venv` recorded in the session summary).
- [ ] `test_no_forbidden_deps_importable` fails if `torch` is installed and passes otherwise
      (demonstrate by reading the assertion; do not actually install torch).
- [ ] `mcpr doctor` exits 0 offline with an empty `.env` and exits non-zero when
      `MCPR_SANDBOX_DIR` points at a non-writable path.
- [ ] `pytest -q` completes in under 60 s with no network access.
- [ ] `mcpr --help` lists all seven reserved sub-command namespaces.
