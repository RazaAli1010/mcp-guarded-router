"""Enforcement of the laptop/Kaggle dependency split (SPEC.md 2.4).

This is the test that keeps the parity guarantee real: because `src/mcpr/` has zero heavy
dependencies, the Kaggle notebooks install the same package and reuse the same prompt and
guardrail code, so training, evaluation and production prompts are identical by construction.
One `pip install torch` on the laptop and that argument stops being true.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SPEC.md 2.4, verbatim.
FORBIDDEN = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "unsloth",
    "sentence-transformers",
    "vllm",
    "xformers",
    "flash-attn",
    "safetensors",
    "tokenizers",
]


def _requirement_name(spec: str) -> str:
    """Distribution name from a PEP 508 requirement, without extras or version specifiers."""
    name = spec.strip()
    for sep in ("[", "(", ";", "=", "<", ">", "!", "~", " "):
        name = name.split(sep)[0]
    return name.strip().lower()


def test_no_forbidden_deps_declared() -> None:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    project = data["project"]
    declared = list(project["dependencies"])
    declared += list(project["optional-dependencies"]["dev"])

    names = {_requirement_name(spec) for spec in declared}
    leaked = sorted(names & {f.lower() for f in FORBIDDEN})
    assert leaked == [], f"forbidden package(s) declared in the laptop dependency set: {leaked}"


def test_kaggle_extra_is_documentation_only() -> None:
    """The kaggle extra must stay empty; its contents live in the notebook install cells."""
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["optional-dependencies"]["kaggle"] == []


def test_no_forbidden_deps_importable() -> None:
    """Fails loudly if someone runs `pip install torch` in the laptop venv.

    `find_spec`, not `try: import` - a partially installed distribution still has a spec and
    still counts as a leak.
    """
    installed = [
        name for name in FORBIDDEN if importlib.util.find_spec(name.replace("-", "_")) is not None
    ]
    assert installed == [], (
        f"forbidden package(s) installed in the laptop environment: {installed}. "
        "These belong on Kaggle only (SPEC.md 2.4)."
    )


def test_mcp_pin_excludes_v2() -> None:
    """SPEC.md 13 D1: `mcp` 2.x is a client-API rewrite. The pin must stay `>=1.29,<2`."""
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    pins = [d for d in data["project"]["dependencies"] if _requirement_name(d) == "mcp"]
    assert pins == ["mcp>=1.29,<2"], f"unexpected mcp pin: {pins}"


def test_package_imports_no_forbidden_module() -> None:
    """No module under src/mcpr/ may import from the forbidden list (SPEC.md 2.4 corollary)."""
    forbidden_modules = {name.replace("-", "_") for name in FORBIDDEN}
    offenders: list[str] = []
    for path in (PROJECT_ROOT / "src" / "mcpr").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            head = stripped.split()[1].split(".")[0]
            if head in forbidden_modules:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {stripped}")
    assert offenders == [], f"forbidden import(s) inside src/mcpr: {offenders}"
