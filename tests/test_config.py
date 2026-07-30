"""The SPEC.md 7.1 constants must exist with exactly these values and these names.

Later features import them by name and other features' data is keyed on their values, so a
"harmless" edit here silently invalidates datasets. The table below is the spec transcribed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcpr import config


def test_constants_match_spec() -> None:
    expected = {
        "REGISTRY_PATH": "schemas/registry.json",
        "MAX_PROMPT_TOKENS": 3072,
        "MAX_UNTRUSTED_CHARS": 4000,
        "TOOLS_PER_PROMPT_MIN": 8,
        "TOOLS_PER_PROMPT_MAX": 24,
        "PROMPT_VERSION": "v1",
        "MIN_CONFUSABLES": 3,
        "INJECTION_FLAG_THRESHOLD": 1.0,
        "INJECTION_BLOCK_THRESHOLD": 3.0,
        "CONFIRM_TTL_SECONDS": 300,
        "SEED": 3407,
        "MAX_GEN_TOKENS": 256,
    }
    for name, value in expected.items():
        assert hasattr(config, name), f"missing constant {name}"
        assert getattr(config, name) == value, f"{name} != {value!r}"


def test_held_out_tools() -> None:
    assert config.HELD_OUT_TOOLS == [
        "github.search_pull_requests",
        "filesystem.directory_tree",
        "git.git_diff",
    ]
    assert len(config.HELD_OUT_TOOLS) == 3


def test_thresholds_ordered() -> None:
    """Flag before block, or the flag tier is unreachable."""
    assert config.INJECTION_FLAG_THRESHOLD < config.INJECTION_BLOCK_THRESHOLD


def test_project_root_is_the_repo_root() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / "SPEC.md").is_file()


@pytest.mark.parametrize("raw", ["schemas/registry.json", "config/policy.toml", "./sandbox"])
def test_resolve_returns_absolute_paths(raw: str) -> None:
    resolved = config.resolve(raw)
    assert isinstance(resolved, Path)
    assert resolved.is_absolute()
    assert resolved.is_relative_to(config.PROJECT_ROOT)


def test_resolve_passes_absolute_paths_through(tmp_path: Path) -> None:
    assert config.resolve(tmp_path) == tmp_path.resolve()


def test_registry_path_resolves_under_the_repo() -> None:
    assert config.resolve(config.REGISTRY_PATH) == config.PROJECT_ROOT / "schemas" / "registry.json"


def test_load_env_reads_mcpr_prefixed_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCPR_TEACHER_MODEL", "some-teacher")
    monkeypatch.setenv("MCPR_RUN_ID", "2026-08-02a")
    settings = config.load_env(env_file=tmp_path / "absent.env")
    assert settings.teacher_model == "some-teacher"
    assert settings.run_id == "2026-08-02a"


def test_load_env_falls_back_to_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty or unset variable must not blank out a documented default."""
    for name in config.Settings.model_fields:
        monkeypatch.delenv(f"MCPR_{name.upper()}", raising=False)
    monkeypatch.setenv("MCPR_SANDBOX_DIR", "")
    settings = config.load_env(env_file=tmp_path / "absent.env")
    assert settings.sandbox_dir == "./sandbox"
    assert settings.router_backend == "replay"
    assert settings.teacher_api_key == ""


def test_settings_covers_every_documented_env_var() -> None:
    """Every MCPR_* name in .env.example must have a Settings field, and vice versa."""
    lines = (config.PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip().removeprefix("MCPR_").lower()
        for line in lines
        if line.strip().startswith("MCPR_")
    }
    assert documented == set(config.Settings.model_fields)


def test_settings_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        config.Settings(not_a_real_setting="x")
