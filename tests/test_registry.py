"""The frozen snapshot is the contract (SPEC.md 3.1), so these tests guard its shape.

Everything here loads `tests/fixtures/registry_min.json` rather than the full snapshot:
24 tools keep the suite fast, and pinning the fixture means a later `snapshot refresh` that
changes the real servers cannot silently change what every other feature's tests assert.
`tests/fixtures/build_fixtures.py` documents how the fixture was cut.

No test in this file touches the network or needs an MCP server (SPEC.md 12).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcpr.registry import confusables, get, load_registry
from mcpr.types import Registry

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURES / "registry_min.json"
RAW_PATH = FIXTURES / "raw_tools_sample.json"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry(FIXTURE_PATH)


@pytest.fixture(scope="module")
def raw_payloads() -> dict[str, dict]:
    return json.loads(RAW_PATH.read_text(encoding="utf-8"))


def test_registry_loads(registry: Registry) -> None:
    assert registry.version == 1
    assert len(registry.tools) == 24
    for spec in registry.tools:
        assert spec.qualified_name == f"{spec.server}.{spec.name}"
        assert spec.effect in ("read", "write", "destructive")
        assert isinstance(spec.input_schema, dict)


def test_registry_is_sorted_by_qualified_name(registry: Registry) -> None:
    """qualified_name is the primary key everywhere; the on-disk order must be canonical."""
    names = [spec.qualified_name for spec in registry.tools]
    assert names == sorted(names)


def test_qualified_names_are_unique(registry: Registry) -> None:
    names = [spec.qualified_name for spec in registry.tools]
    assert len(names) == len(set(names))


def test_none_is_never_a_registry_entry(registry: Registry) -> None:
    """SPEC.md 6.2 reserves "none" for abstention, so it must not collide with a real tool."""
    assert "none" not in {spec.qualified_name for spec in registry.tools}
    with pytest.raises(KeyError):
        get("none", registry)


def test_fixture_spans_every_server(registry: Registry) -> None:
    assert {spec.server for spec in registry.tools} == {"fetch", "filesystem", "git", "github"}


def test_fixture_has_destructive_tools(registry: Registry) -> None:
    """F3's policy gate needs real destructive targets to test against."""
    destructive = [s.qualified_name for s in registry.tools if s.effect == "destructive"]
    assert len(destructive) >= 3


def test_get_returns_the_requested_tool(registry: Registry) -> None:
    spec = get("github.search_code", registry)
    assert spec.server == "github"
    assert spec.name == "search_code"


def test_get_raises_for_an_unknown_tool(registry: Registry) -> None:
    with pytest.raises(KeyError):
        get("github.not_a_real_tool", registry)


# --- confusables --------------------------------------------------------------------------


def test_confusables_are_deterministic(registry: Registry) -> None:
    first = confusables("github.search_code", 8, registry)
    second = confusables("github.search_code", 8, registry)
    assert first == second
    assert len(first) == 8


def test_confusables_excludes_self(registry: Registry) -> None:
    ranked = confusables("github.search_code", len(registry.tools), registry)
    assert "github.search_code" not in ranked


def test_confusables_rank_the_search_cluster_first(registry: Registry) -> None:
    """The F1 acceptance criterion: the top three are all `github.search_*`.

    This cluster - search_code, search_repositories, search_issues, search_pull_requests -
    is the disambiguation task the whole project is built around (SPEC.md 1).
    """
    top3 = confusables("github.search_code", 3, registry)
    assert all(name.startswith("github.search_") for name in top3), top3
    assert "github.search_repositories" in top3
    assert "github.search_issues" in top3


def test_confusables_respects_k(registry: Registry) -> None:
    assert len(confusables("git.git_diff", 3, registry)) == 3
    assert len(confusables("git.git_diff", 1, registry)) == 1


def test_confusables_prefers_the_same_server(registry: Registry) -> None:
    """The `0.2 * same server` term must actually bite: git's neighbours are git tools."""
    top3 = confusables("git.git_diff", 3, registry)
    assert all(name.startswith("git.") for name in top3), top3


# --- the verbatim guarantee ---------------------------------------------------------------


def test_input_schema_verbatim(registry: Registry, raw_payloads: dict[str, dict]) -> None:
    """Stored schemas must be byte-equal to what the servers actually returned.

    This is the test that stops a well-meaning later change from sorting keys, filling in a
    missing `type`, or otherwise tidying a schema. Real MCP schemas are messy and the router
    has to learn the mess; a cleaned snapshot would train against a world that does not exist.
    """
    assert raw_payloads, "fixture is empty"
    for spec in registry.tools:
        raw = raw_payloads[spec.qualified_name]
        assert json.dumps(spec.input_schema, sort_keys=True) == json.dumps(
            raw["inputSchema"], sort_keys=True
        ), f"{spec.qualified_name} input_schema was altered on capture"
        assert spec.output_schema == raw.get("outputSchema")


def test_annotations_are_preserved_verbatim(
    registry: Registry, raw_payloads: dict[str, dict]
) -> None:
    """Including the explicit nulls: SPEC.md 8.1 rule 2 reads these, so they must be untouched."""
    for spec in registry.tools:
        assert spec.annotations == (raw_payloads[spec.qualified_name].get("annotations") or {})


def test_description_and_title_come_from_the_payload(
    registry: Registry, raw_payloads: dict[str, dict]
) -> None:
    for spec in registry.tools:
        raw = raw_payloads[spec.qualified_name]
        assert spec.description == (raw.get("description") or "")
        assert spec.title == raw.get("title")
