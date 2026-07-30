## F1 — MCP client and the frozen tool-schema snapshot

**Goal:** Connect to real MCP servers, capture their `tools/list` output verbatim into a
committed `schemas/registry.json`, and make every downstream component depend on that file
rather than on a live server.

**Depends on:** F0

**Runs on:** LAPTOP only. `mcpr snapshot refresh` is the only command in the entire project
allowed to open an MCP connection.

### Context digest

- `ToolSpec` fields, exactly (SPEC.md §6.1): `server, name, qualified_name, title, description,
  input_schema, output_schema, annotations, effect`. `qualified_name = f"{server}.{name}"` and is
  the primary key everywhere.
- `schemas/registry.json` = `{"version": 1, "tools": [...]}`, sorted by `qualified_name`.
- `effect_for()` derivation rules, three tiers, first match wins — SPEC.md §8.1.
- Servers and launch commands — SPEC.md §4.4:
  `uvx mcp-server-fetch` (PyPI 2026.7.10) · `uvx mcp-server-git --repository <path>` (2026.7.10) ·
  `npx -y @modelcontextprotocol/server-filesystem <dir>` (npm 2026.7.10, optional) ·
  GitHub remote Streamable HTTP at `https://api.githubcopilot.com/mcp/` with
  `Authorization: Bearer $MCPR_GITHUB_PAT`, `X-MCP-Readonly: true`,
  `X-MCP-Toolsets: repos,issues,pull_requests,users`.
- SDK pin `mcp>=1.29,<2`, v1 client API (SPEC.md §13 D1).
- `HELD_OUT_TOOLS` default (SPEC.md §7.1):
  `["github.search_pull_requests", "filesystem.directory_tree", "git.git_diff"]`.
- Principle SPEC.md §3.1: the snapshot is the contract; tests never need a live server.

### Context deltas

- **`config/servers.toml` gains a `[servers.<id>].sandbox_arg` boolean** indicating that the
  server's last positional arg must be replaced with the resolved `MCPR_SANDBOX_DIR` at launch.
  Add this key to SPEC.md §5's description of `servers.toml`.
- **If any tool in `HELD_OUT_TOOLS` is absent from the captured snapshot**, replace it with a
  present tool of similar confusability and update SPEC.md §7.1 in the same commit. Record the
  substitution in the session summary.

### Scope

1. `src/mcpr/mcp_client.py`:
   ```python
   async def list_tools(server_id: str, cfg: ServerConfig) -> list[dict]
   async def call_tool(server_id: str, cfg: ServerConfig, name: str, args: dict) -> ToolResult
   ```
   Two transports. stdio: `stdio_client(StdioServerParameters(command=..., args=..., env=...))`
   then `ClientSession(read, write)` then `await session.initialize()` then
   `await session.list_tools()`. http: `streamablehttp_client(url, headers=...)` returning
   `(read, write, get_session_id)` then the same `ClientSession` dance. Both wrapped in a
   30 s timeout with one retry. `ToolResult` carries `content_text: str`, `is_error: bool`,
   `structured: dict | None`.
2. `src/mcpr/snapshot.py::build_registry(server_ids: list[str]) -> Registry` — calls
   `list_tools` per server, maps each MCP `Tool` to a `ToolSpec` preserving `inputSchema` and
   `outputSchema` **verbatim** (no reformatting, no key renaming, no schema "cleanup"; the
   messiness is the point of the project), computes `effect` via `registry.effect_for`, sorts
   by `qualified_name`.
3. `src/mcpr/registry.py`:
   ```python
   def load_registry(path: Path = REGISTRY_PATH) -> Registry     # cached
   def get(qualified_name: str) -> ToolSpec
   def effect_for(spec: ToolSpec, overrides: dict[str, str]) -> Literal["read","write","destructive"]
   def confusables(qualified_name: str, k: int) -> list[str]
   ```
   `confusables()` is deterministic and pure: rank other tools by
   `0.5 * jaccard(token_set(description)) + 0.3 * jaccard(token_set(name.split("_"))) +
   0.2 * (same server)`, ties broken by `qualified_name`. Same server + shared verb prefix
   (`search_*`, `list_*`, `get_*`) must rank highest — `github.search_code`,
   `github.search_repositories`, `github.search_issues` and `github.search_pull_requests` are
   the canonical confusable cluster this project exists to disambiguate.
4. `schemas/registry.meta.json` — `{"captured_utc", "sdk_version", "servers": [{"id",
   "transport", "package", "package_version", "tool_count", "annotation_coverage"}],
   "tool_count", "sha256_registry", "effect_counts": {"read": n, "write": n, "destructive": n}}`.
   `annotation_coverage` = fraction of that server's tools carrying any MCP annotation; it is
   reported because rule 3 of §8.1 exists precisely for servers that omit them.
5. CLI: `mcpr snapshot refresh [--servers fetch,git,filesystem,github]` writes both files;
   `mcpr snapshot show [--server X] [--effect destructive]` prints a table;
   `mcpr snapshot diff <old.json>` prints added/removed/changed tools and exits 1 on any change
   (so drift is detectable later); `mcpr snapshot confusables <qualified_name>` prints the top 8.
6. `tests/fixtures/registry_min.json` — a hand-trimmed 24-tool subset of the real snapshot,
   committed, spanning all four servers and including the full `search_*` cluster plus at least
   three `destructive` tools and three tools with **no** annotations. Every later feature's unit
   tests load this fixture, never the full registry, so tests stay fast and stable.
7. Commit `schemas/registry.json` and `schemas/registry.meta.json` to git. They are project
   data, not build output.

### Out of scope

- Executing tools in anger — `call_tool` is written and smoke-tested here but is only wired
  into the pipeline in F9.
- Any policy decision about effects beyond deriving the `effect` field — F3.
- Prompt rendering of tool schemas — F2.

### Implementation notes

- **mcp v1 API surface** (do not use the v2 `Client(target)` form):
  `from mcp import ClientSession, StdioServerParameters`,
  `from mcp.client.stdio import stdio_client`,
  `from mcp.client.streamable_http import streamablehttp_client`.
  `await session.initialize()` is required before `list_tools()` in v1.
- `session.list_tools()` returns `ListToolsResult` with `.tools`, each a `Tool` with
  **camelCase** `inputSchema` / `outputSchema` and `.annotations` (which may be `None`).
  Convert with `tool.model_dump(by_alias=True, exclude_none=False)` and pull the fields out;
  do not hand-build dicts field by field.
- `tools/list` is paginated (`nextCursor`). The GitHub server returns enough tools to paginate.
  Loop until `nextCursor` is `None` or you will silently capture a partial catalog.
- Launch each stdio server with a minimal env (`PATH`, `HOME`, plus the server's own vars) —
  do not forward the whole environment, which would leak `MCPR_TEACHER_API_KEY` into a
  subprocess.
- `uvx` downloads on first run; budget ~250 MB of cache (SPEC.md §2.5). If `uv` is missing,
  `snapshot refresh` must fail with an actionable message, not a traceback.
- If the GitHub PAT is absent or the remote server refuses, capture the other three servers and
  record `"github": {"status": "unavailable", "reason": ...}` in `registry.meta.json` rather
  than aborting. The project must be completable without a PAT, with a smaller confusable set.
- Target snapshot size: 55–90 tools total. If GitHub is available its `repos`+`issues`+
  `pull_requests`+`users` toolsets alone supply most of that.

### Test plan

- `test_registry_loads` — `registry_min.json` parses into `ToolSpec`s; `qualified_name`
  equals `f"{server}.{name}"` for every entry; the list is sorted.
- `test_effect_derivation` — table-driven over the three tiers: an explicit override wins over
  an annotation; `destructiveHint=True` → `destructive`; `readOnlyHint=True` → `read`; a tool
  with **no** annotations and name `search_code` → `read`; `delete_file` → `destructive`;
  `create_or_update_file` → `write`.
- `test_confusables_are_deterministic` — two calls return an identical list; calling with
  `k=8` on `github.search_code` returns a list whose first three entries are all
  `github.search_*`.
- `test_confusables_excludes_self`.
- `test_input_schema_verbatim` — for a fixture tool, the stored `input_schema` is byte-equal to
  the fixture's raw MCP payload (guards against helpful reformatting).
- `tests/live/test_snapshot_live.py` — marked `@pytest.mark.live`, connects to `fetch` only and
  asserts ≥ 1 tool. Excluded from `make check`.

### Verify

```bash
mcpr snapshot refresh --servers fetch,git
mcpr snapshot refresh --servers fetch,git,filesystem,github   # if node + PAT available
mcpr snapshot show | head -40
mcpr snapshot confusables github.search_code
python -c "import json;d=json.load(open('schemas/registry.meta.json'));print(d['tool_count'],d['effect_counts'])"
```

Expected: `registry.json` and `registry.meta.json` exist; `tool_count` ≥ 20 (fetch+git only) or
≥ 55 (all four); `effect_counts` has a non-zero `destructive` bucket; `mcpr snapshot confusables
github.search_code` lists `github.search_repositories` and `github.search_issues` in the top 3.

### Acceptance criteria

- [ ] `schemas/registry.json` is committed and contains ≥ 20 tools from ≥ 2 real servers, with
      `input_schema` byte-identical to what the servers returned.
- [ ] `registry.meta.json` records package versions, capture time, per-server
      `annotation_coverage`, and the registry sha256.
- [ ] `effect_for` is covered by a table-driven test hitting all three derivation tiers,
      including the no-annotations path.
- [ ] `confusables("github.search_code", 3)` returns three `github.search_*` tools,
      deterministically across runs.
- [ ] The full non-live test suite passes with **no MCP server running and no network**.
- [ ] `mcpr snapshot refresh` degrades gracefully (records `unavailable`, exits 0) when the
      GitHub PAT is missing.
