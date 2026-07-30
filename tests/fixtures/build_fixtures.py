"""Regenerate the test fixtures from the committed snapshot.

    mcpr snapshot refresh --servers fetch,git,filesystem,github --raw-dir <dir>
    python tests/fixtures/build_fixtures.py <dir>

Not a test - pytest ignores it, since collection only picks up `test_*.py`. It exists so the
fixtures are reproducible from a command rather than hand-curated once and never explainable
again (SPEC.md 3.4). Re-run it whenever `schemas/registry.json` is refreshed, then check the
diff: a change here means the real servers changed under the project.

The 24 tools are chosen to make every later feature's unit tests possible without loading the
full 50-tool registry:

- all four servers are represented;
- the whole `github.search_*` cluster is present - `search_code`, `search_repositories`,
  `search_issues` and `search_pull_requests` are the confusable set the project exists to
  disambiguate (SPEC.md 1);
- every `destructive` tool in the snapshot is included, so F3's policy gate has real targets;
- both non-github `HELD_OUT_TOOLS` entries are present.

One requirement of the F1 spec is *not* met and is deliberately not faked: it asks for at
least three tools carrying **no** annotations, to exercise SPEC.md 8.1 rule 3. The real
servers only supply one (`fetch.fetch`) - git, filesystem and the GitHub remote all annotate
100% of their tools. Stripping annotations from a real tool to manufacture the other two
would fabricate the exact condition the fixture exists to test, so the no-annotation path is
covered by hand-built ToolSpecs in tests/test_effects.py instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcpr.io import write_json
from mcpr.registry import load_registry
from mcpr.types import Registry

FIXTURES = Path(__file__).parent

KEEP = [
    "fetch.fetch",
    "filesystem.directory_tree",
    "filesystem.edit_file",
    "filesystem.list_directory",
    "filesystem.move_file",
    "filesystem.read_text_file",
    "filesystem.search_files",
    "filesystem.write_file",
    "git.git_add",
    "git.git_commit",
    "git.git_diff",
    "git.git_log",
    "git.git_reset",
    "git.git_status",
    "github.get_file_contents",
    "github.issue_read",
    "github.list_commits",
    "github.list_issues",
    "github.search_code",
    "github.search_commits",
    "github.search_issues",
    "github.search_pull_requests",
    "github.search_repositories",
    "github.search_users",
]


def main(raw_dir: Path) -> None:
    """Write registry_min.json and raw_tools_sample.json. Impure: reads and writes files."""
    by_name = {t.qualified_name: t for t in load_registry().tools}
    missing = [name for name in KEEP if name not in by_name]
    if missing:
        raise SystemExit(f"not in schemas/registry.json: {missing}")

    tools = sorted((by_name[name] for name in KEEP), key=lambda t: t.qualified_name)
    write_json(FIXTURES / "registry_min.json", Registry(version=1, tools=tools))

    # The untouched MCP payloads for exactly those tools, so test_input_schema_verbatim can
    # prove the stored schemas were never reformatted on the way in.
    keep = set(KEEP)
    raw: dict[str, dict] = {}
    for server in ("fetch", "filesystem", "git", "github"):
        payload = json.loads((raw_dir / f"{server}.json").read_text(encoding="utf-8"))
        for entry in payload:
            qualified_name = f"{server}.{entry['name']}"
            if qualified_name in keep:
                raw[qualified_name] = entry
    write_json(FIXTURES / "raw_tools_sample.json", raw)

    print(f"registry_min.json: {len(tools)} tools from {sorted({t.server for t in tools})}")
    print(f"raw_tools_sample.json: {len(raw)} verbatim payloads")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
