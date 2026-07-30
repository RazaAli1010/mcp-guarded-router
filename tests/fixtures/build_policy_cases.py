"""Regenerate `tests/fixtures/policy_cases.jsonl` (F3 scope 7).

Run with `python tests/fixtures/build_policy_cases.py`. The fixture is committed; this script
exists so the rows can be re-emitted byte-identically rather than hand-edited into drift.

The fixture is claim C3's evidence: F8 replays it against the real `schemas/registry.json` to
compute `destructive_catch_rate`, which must be exactly 1.000. So every `tool` here is a real
tool from that snapshot, except the three rows that deliberately name tools which exist nowhere.

`unknown_property` is the one SPEC.md 8.2 code with no row. No schema in the snapshot sets
`additionalProperties: false`, and F3's implementation notes forbid injecting it ourselves -
that would fabricate violations and inflate the very metric claim C2 reports. The code is
covered instead by a synthetic strict spec in `tests/test_guards_schema.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "policy_cases.jsonl"

# (tool, arguments, expected_action, expected_code, layer, note, plan)
# `plan` is None on all but the plan-lock rows.
CASES: list[tuple] = [
    # --- layer 2: reads pass straight through -------------------------------------------------
    ("github.search_code", {"query": "mcp"}, "allow", "read_only", "policy", "plain read", None),
    (
        "github.search_repositories",
        {"query": "mcp stars:>100"},
        "allow",
        "read_only",
        "policy",
        "read with a qualified query",
        None,
    ),
    (
        "github.list_issues",
        {"owner": "modelcontextprotocol", "repo": "servers"},
        "allow",
        "read_only",
        "policy",
        "read, only required args",
        None,
    ),
    (
        "github.issue_read",
        {"method": "get", "owner": "a", "repo": "b", "issue_number": 1},
        "allow",
        "read_only",
        "policy",
        "read with a valid enum member",
        None,
    ),
    (
        "github.get_file_contents",
        {"owner": "a", "repo": "b", "path": "/README.md"},
        "allow",
        "read_only",
        "policy",
        "a github path is NOT sandbox-checked: github is not in sandboxed_servers, and its "
        "path argument names a remote repo, not the local disk",
        None,
    ),
    (
        "filesystem.read_text_file",
        {"path": "notes.txt"},
        "allow",
        "read_only",
        "policy",
        "relative path inside the sandbox",
        None,
    ),
    (
        "filesystem.list_directory",
        {"path": "."},
        "allow",
        "read_only",
        "policy",
        "the sandbox root itself is inside the sandbox",
        None,
    ),
    (
        "filesystem.search_files",
        {"path": "sub/dir", "pattern": "*.md"},
        "allow",
        "read_only",
        "policy",
        "nested relative path",
        None,
    ),
    ("git.git_status", {"repo_path": "."}, "allow", "read_only", "policy", "git read", None),
    (
        "fetch.fetch",
        {"url": "https://example.com"},
        "allow",
        "read_only",
        "policy",
        "fetch has no annotations at all; read comes from the verb heuristic",
        None,
    ),
    # --- layer 2: every mutation is gated -----------------------------------------------------
    (
        "filesystem.write_file",
        {"path": "out.txt", "content": "hello"},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "destructiveHint: true, pinned by an override",
        None,
    ),
    (
        "filesystem.edit_file",
        {"path": "out.txt", "edits": [{"oldText": "a", "newText": "b"}]},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "destructive with a well-formed nested edits array",
        None,
    ),
    (
        "filesystem.move_file",
        {"source": "a.txt", "destination": "b.txt"},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "both paths inside the sandbox, still gated",
        None,
    ),
    (
        "filesystem.create_directory",
        {"path": "newdir"},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "write, not destructive - still confirms",
        None,
    ),
    (
        "git.git_reset",
        {"repo_path": "."},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "destructiveHint: true",
        None,
    ),
    (
        "git.git_add",
        {"repo_path": ".", "files": ["a.txt", "b.txt"]},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "write via the verb heuristic, list-valued paths all inside",
        None,
    ),
    (
        "git.git_commit",
        {"repo_path": ".", "message": "wip"},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "write, pinned by an override",
        None,
    ),
    (
        "git.git_checkout",
        {"repo_path": ".", "branch_name": "main"},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "the one tool the heuristic gets wrong: destructiveHint is false and 'checkout' is in "
        "neither verb list, so only the override makes this destructive",
        None,
    ),
    # --- layer 2: path containment ------------------------------------------------------------
    (
        "filesystem.read_text_file",
        {"path": "../../etc/passwd"},
        "block",
        "path_escape",
        "policy",
        "dotdot traversal out of the sandbox",
        None,
    ),
    (
        "filesystem.write_file",
        {"path": "../escape.txt", "content": "x"},
        "block",
        "path_escape",
        "policy",
        "a single dotdot is still an escape",
        None,
    ),
    (
        "filesystem.edit_file",
        {"path": "/etc/hosts", "edits": [{"oldText": "a", "newText": "b"}]},
        "block",
        "path_escape",
        "policy",
        "absolute path; note Path('/etc/hosts').is_absolute() is False on Windows, which is why "
        "the guard uses os.path.join rather than an absoluteness test",
        None,
    ),
    (
        "filesystem.directory_tree",
        {"path": "~"},
        "block",
        "path_escape",
        "policy",
        "expanduser runs before the join, so ~ becomes the home directory and escapes",
        None,
    ),
    (
        "filesystem.list_directory",
        {"path": "nested/../../../outside"},
        "block",
        "path_escape",
        "policy",
        "dotdots buried mid-path, normalised by realpath",
        None,
    ),
    (
        "filesystem.search_files",
        {"path": "../..", "pattern": "*"},
        "block",
        "path_escape",
        "policy",
        "escape on a read tool: containment does not care about the effect",
        None,
    ),
    (
        "filesystem.move_file",
        {"source": "a.txt", "destination": "../../out.txt"},
        "block",
        "path_escape",
        "policy",
        "escape via destination, the second of two path arguments",
        None,
    ),
    (
        "filesystem.move_file",
        {"source": "../../in.txt", "destination": "b.txt"},
        "block",
        "path_escape",
        "policy",
        "escape via source; neither argument is called 'path'",
        None,
    ),
    (
        "git.git_reset",
        {"repo_path": "../.."},
        "block",
        "path_escape",
        "policy",
        "repo_path is a path argument too",
        None,
    ),
    (
        "git.git_add",
        {"repo_path": ".", "files": ["ok.txt", "../../x"]},
        "block",
        "path_escape",
        "policy",
        "list-valued path argument; git_add.files has no description, so only the name list "
        "catches it",
        None,
    ),
    # --- layer 1: schema validation -----------------------------------------------------------
    (
        "github.search_code",
        {},
        "block",
        "missing_required",
        "schema",
        "required query absent",
        None,
    ),
    (
        "filesystem.write_file",
        {"path": "a.txt"},
        "block",
        "missing_required",
        "schema",
        "one of two required args absent",
        None,
    ),
    (
        "git.git_commit",
        {"repo_path": "."},
        "block",
        "missing_required",
        "schema",
        "missing message on a write tool - layer 1 blocks before layer 2 confirms",
        None,
    ),
    (
        "filesystem.edit_file",
        {"path": "a.txt", "edits": [{"oldText": "x"}]},
        "block",
        "missing_required",
        "schema",
        "required field missing two levels down, at $.edits[0]",
        None,
    ),
    (
        "github.issue_read",
        {"owner": "a", "repo": "b"},
        "block",
        "missing_required",
        "schema",
        "two required args absent; the deterministic sort decides which is reported",
        None,
    ),
    (
        "github.search_code",
        {"query": 123},
        "block",
        "type_mismatch",
        "schema",
        "number where a string is required",
        None,
    ),
    (
        "filesystem.read_text_file",
        {"path": ["a.txt"]},
        "block",
        "type_mismatch",
        "schema",
        "array where a string is required; layer 2 skips non-string path values so this is the "
        "only code produced",
        None,
    ),
    (
        "git.git_add",
        {"repo_path": ".", "files": "a.txt"},
        "block",
        "type_mismatch",
        "schema",
        "string where an array is required",
        None,
    ),
    (
        "fetch.fetch",
        {"url": "https://example.com", "max_length": "lots"},
        "block",
        "type_mismatch",
        "schema",
        "string where an integer is required",
        None,
    ),
    (
        "filesystem.edit_file",
        {"path": "a.txt", "edits": "replace a with b"},
        "block",
        "type_mismatch",
        "schema",
        "string where an array of objects is required",
        None,
    ),
    (
        "github.search_code",
        {"query": "mcp", "order": "sideways"},
        "block",
        "enum_violation",
        "schema",
        "not a member of [asc, desc]",
        None,
    ),
    (
        "github.issue_read",
        {"method": "delete", "owner": "a", "repo": "b", "issue_number": 1},
        "block",
        "enum_violation",
        "schema",
        "an invented enum member on a read tool",
        None,
    ),
    (
        "github.list_issues",
        {"owner": "a", "repo": "b", "state": "???"},
        "block",
        "enum_violation",
        "schema",
        "enum members here are uppercase, so a plausible lowercase guess still fails",
        None,
    ),
    (
        "github.search_code",
        {"query": "mcp", "page": 0},
        "block",
        "constraint_violation",
        "schema",
        "below minimum 1",
        None,
    ),
    (
        "github.search_code",
        {"query": "mcp", "perPage": 500},
        "block",
        "constraint_violation",
        "schema",
        "above maximum 100",
        None,
    ),
    (
        "fetch.fetch",
        {"url": "https://example.com", "start_index": -1},
        "block",
        "constraint_violation",
        "schema",
        "below minimum 0",
        None,
    ),
    (
        "fetch.fetch",
        {"url": ""},
        "block",
        "constraint_violation",
        "schema",
        "empty string violates minLength",
        None,
    ),
    (
        "filesystem.write_file",
        ["out.txt", "hello"],
        "block",
        "not_object",
        "schema",
        "arguments is an array, not an object. Unreachable through a validated ToolCall, but "
        "reachable through model_construct, which is how F8 builds prediction rows in bulk",
        None,
    ),
    (
        "github.search_everything",
        {"query": "mcp"},
        "block",
        "unknown_tool",
        "schema",
        "hallucinated github tool",
        None,
    ),
    (
        "filesystem.delete_everything",
        {"path": "."},
        "block",
        "unknown_tool",
        "schema",
        "a destructive-sounding tool that does not exist must not be derived an effect",
        None,
    ),
    (
        "git.git_push",
        {"repo_path": "."},
        "block",
        "unknown_tool",
        "schema",
        "plausible but absent from the snapshot",
        None,
    ),
    # --- the abstention and the non-blocking note ---------------------------------------------
    ("none", {}, "allow", "abstained", "schema", "the reserved abstention id", None),
    (
        "github.search_code",
        {"query": "mcp", "reason": "the user asked about code"},
        "allow",
        "extra_property_permitted",
        "schema",
        "an invented argument is legal because the schema omits additionalProperties: false; it "
        "is recorded, not blocked, so the rate stays measurable without being manufactured",
        None,
    ),
    (
        "filesystem.write_file",
        {"path": "a.txt", "content": "x", "mode": "append"},
        "confirm",
        "extra_property_permitted",
        "schema",
        "the note rides alongside the policy confirmation rather than replacing it",
        None,
    ),
    # --- layer 2: the plan lock ---------------------------------------------------------------
    (
        "filesystem.read_text_file",
        {"path": "a.txt"},
        "confirm",
        "outside_plan",
        "policy",
        "a read the plan did not authorise: the server is out of scope",
        {"allowed_servers": ["github"], "max_effect": "read"},
    ),
    (
        "filesystem.write_file",
        {"path": "a.txt", "content": "x"},
        "confirm",
        "outside_plan",
        "policy",
        "right server, effect above the plan's ceiling",
        {"allowed_servers": ["filesystem"], "max_effect": "read"},
    ),
    (
        "git.git_add",
        {"repo_path": ".", "files": ["a.txt"]},
        "confirm",
        "outside_plan",
        "policy",
        "write attempted under a read-only plan",
        {"allowed_servers": ["git"], "max_effect": "read"},
    ),
    (
        "git.git_commit",
        {"repo_path": ".", "message": "wip"},
        "confirm",
        "destructive_requires_confirmation",
        "policy",
        "inside the plan and still gated - the plan lock is an extra check, never a bypass",
        {"allowed_servers": ["github", "git"], "max_effect": "write"},
    ),
    (
        "github.search_code",
        {"query": "mcp"},
        "allow",
        "read_only",
        "policy",
        "control: a plan that does authorise the call changes nothing",
        {"allowed_servers": ["github"], "max_effect": "read"},
    ),
]


def build() -> list[dict]:
    """Materialise every case as a fixture row. Pure."""
    rows = []
    for index, (tool, arguments, action, code, layer, note, plan) in enumerate(CASES, start=1):
        row = {
            "call": {"arguments": arguments, "tool": tool},
            "expected_action": action,
            "expected_code": code,
            "id": f"pc_{index:03d}",
            "layer": layer,
            "note": note,
        }
        if plan is not None:
            row["plan"] = plan
        rows.append(row)
    return rows


def main() -> None:
    """Write the fixture. Keys are sorted so the file round-trips byte-identically."""
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build()]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(lines)} cases -> {OUT}")


if __name__ == "__main__":
    main()
