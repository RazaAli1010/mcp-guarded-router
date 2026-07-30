"""The only sanctioned way to read and write project data files.

Every write funnels through here so SPEC.md 3.3 determinism cannot be violated per-caller:
`sort_keys=True`, `ensure_ascii=False`, a trailing newline, and - critically on Windows -
`newline="\\n"`, because Python's default text mode would otherwise translate `\\n` to
`\\r\\n` and make `sha256_file` disagree with the byte-identical file written on Kaggle,
breaking the provenance checks of SPEC.md 10.1 and 10.3.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one dict per line of a JSONL file, skipping blank lines.

    Impure: reads from the filesystem. Lazy - the file stays open until the iterator is
    exhausted, so callers that need the file closed should wrap this in `list()`.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    """Write `rows` as JSONL, one object per line, keys sorted.

    Impure: creates parent directories and writes to the filesystem. Pydantic models are
    accepted directly and dumped via `model_dump(mode="json")`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False))
            fh.write("\n")


def write_json(path: str | Path, obj: Any) -> None:
    """Write a single JSON document, keys sorted, 2-space indent, trailing newline.

    Impure: creates parent directories and writes to the filesystem. The trailing newline is
    load-bearing - several later diffs depend on it.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(_jsonable(obj), sort_keys=True, ensure_ascii=False, indent=2))
        fh.write("\n")


def sha256_file(path: str | Path) -> str:
    """Return the hex sha256 of a file's bytes.

    Impure: reads from the filesystem. Read in binary in 1 MiB chunks, so the digest is the
    true on-disk digest and matches what a Kaggle notebook computes for the same file.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(obj: Any) -> Any:
    """Convert a Pydantic model to a plain dict; pass anything else through unchanged.

    Pure. Kept private so callers never have to think about which of the two they hold.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return obj
