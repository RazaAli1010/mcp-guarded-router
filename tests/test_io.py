"""Determinism of the I/O layer (SPEC.md 3.3).

"Same inputs + same seed => byte-identical outputs" is only true if every writer sorts keys
and emits LF. These tests check the bytes, not the parsed values - a file that round-trips
through json but differs byte-wise still breaks the sha256 provenance chain of SPEC.md 10.3.
"""

from __future__ import annotations

from pathlib import Path

from mcpr.io import read_jsonl, sha256_file, write_json, write_jsonl
from mcpr.types import ToolCall


def test_write_then_read_jsonl_round_trips(tmp_path: Path) -> None:
    rows = [{"id": "q_1", "tool": "git.git_diff"}, {"id": "q_2", "tool": "none"}]
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, rows)
    assert list(read_jsonl(path)) == rows


def test_jsonl_bytes_are_stable_across_key_insertion_order(tmp_path: Path) -> None:
    """The determinism guarantee: key order in the source dict must not reach the file."""
    a = {"id": "q_1", "tool": "git.git_diff", "arguments": {"repo": ".", "a": 1}}
    b = {"arguments": {"a": 1, "repo": "."}, "tool": "git.git_diff", "id": "q_1"}

    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(first, [a])
    write_jsonl(second, [b])

    assert first.read_bytes() == second.read_bytes()
    assert sha256_file(first) == sha256_file(second)


def test_writes_use_lf_endings_and_a_trailing_newline(tmp_path: Path) -> None:
    """CRLF here would make Windows and Kaggle disagree on every sha256 in the manifest."""
    jsonl = tmp_path / "rows.jsonl"
    write_jsonl(jsonl, [{"a": 1}, {"b": 2}])
    raw = jsonl.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2

    doc = tmp_path / "meta.json"
    write_json(doc, {"version": 1})
    raw = doc.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")


def test_non_ascii_is_written_unescaped(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"query": "café - naïve - 東京"}])
    text = path.read_text(encoding="utf-8")
    assert "café" in text
    assert "\\u" not in text


def test_write_json_sorts_keys(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    write_json(path, {"z": 1, "a": 2, "m": 3})
    text = path.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"m"') < text.index('"z"')


def test_pydantic_models_are_written_directly(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    write_jsonl(path, [ToolCall(tool="none", arguments={})])
    assert list(read_jsonl(path)) == [{"arguments": {}, "tool": "none"}]


def test_parent_directories_are_created(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / "2026-08-02a" / "tuned.jsonl"
    write_jsonl(path, [{"id": "q_1"}])
    assert path.is_file()


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"b": 2}]


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"mcpr" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
