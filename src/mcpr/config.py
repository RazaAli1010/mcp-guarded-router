"""Project constants (SPEC.md 7.1) and environment loading (SPEC.md 7.2).

Every constant here is a module-level uppercase literal imported by name elsewhere; none of
them may be renamed, because later features depend on the exact spelling. Environment
variables are read with ``os.environ`` after ``dotenv.load_dotenv()`` - deliberately not
``pydantic-settings``, which is not in the dependency list.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

# --- Paths ----------------------------------------------------------------------------------

# src/mcpr/config.py -> src/mcpr -> src -> repo root. Correct for an editable install, which is
# the only install this project uses (laptop: `uv pip install -e .`; Kaggle: `pip install -e`).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Repo-relative path of the frozen tool-schema snapshot. Pass through `resolve()` to read it.
REGISTRY_PATH = "schemas/registry.json"

# --- Prompt and catalog shape ---------------------------------------------------------------

MAX_PROMPT_TOKENS = 3072
MAX_UNTRUSTED_CHARS = 4000
TOOLS_PER_PROMPT_MIN = 8
TOOLS_PER_PROMPT_MAX = 24

#: Embedded in ROUTER_SYSTEM_PROMPT (F2). Bumping it invalidates every labelled dataset.
PROMPT_VERSION = "v1"

#: Hard negatives forced into every sampled catalog.
MIN_CONFUSABLES = 3

# --- Guardrail thresholds (config/policy.toml carries the tunable copies) --------------------

INJECTION_FLAG_THRESHOLD = 1.0
INJECTION_BLOCK_THRESHOLD = 3.0
CONFIRM_TTL_SECONDS = 300

# --- Evaluation -----------------------------------------------------------------------------

#: Never placed in a training catalog; the zero-shot arm of the test set. F1 must adjust this
#: via a Context delta - not a silent edit - if a listed tool is absent from the real snapshot.
HELD_OUT_TOOLS = [
    "github.search_pull_requests",
    "filesystem.directory_tree",
    "git.git_diff",
]

SEED = 3407
MAX_GEN_TOKENS = 256


def resolve(p: str | Path) -> Path:
    """Return `p` as an absolute path, resolving repo-relative paths against PROJECT_ROOT.

    Pure: no I/O, no clock, no globals beyond the module-level PROJECT_ROOT constant. The
    path is not required to exist.
    """
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


class Settings(BaseModel):
    """The MCPR_* environment variables of SPEC.md 7.2, as a plain Pydantic model.

    Teacher and baseline models are configured, never hardcoded (SPEC.md 7.2): `mcpr doctor`
    resolves them at runtime and later features record the resolved ids in the dataset
    manifest. Empty strings mean "not configured" and cause network checks to be skipped.
    """

    model_config = ConfigDict(extra="forbid")

    sandbox_dir: str = "./sandbox"
    github_pat: str = ""
    github_mcp_url: str = "https://api.githubcopilot.com/mcp/"
    teacher_base_url: str = ""
    teacher_model: str = ""
    teacher_api_key: str = ""
    baseline_base_url: str = ""
    baseline_model: str = ""
    baseline_api_key: str = ""
    router_backend: str = "replay"
    router_base_url: str = ""
    router_model: str = ""
    router_api_key: str = ""
    hf_repo: str = ""
    run_id: str = ""


def load_env(env_file: str | Path | None = None) -> Settings:
    """Load `.env` (if present) and read every MCPR_* variable into a Settings.

    Impure: touches the filesystem and os.environ. An unset or empty variable falls back to
    the Settings field default, so this never raises on a fresh checkout with no `.env`.
    """
    load_dotenv(dotenv_path=env_file or resolve(".env"), override=False)
    defaults = Settings()
    values = {
        name: os.environ.get(f"MCPR_{name.upper()}", "") or getattr(defaults, name)
        for name in Settings.model_fields
    }
    return Settings(**values)
