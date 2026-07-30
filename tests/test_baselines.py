"""The lexical baseline is the floor F8 reports beside the model (F2 scope 7).

It only has to be honest and deterministic. These tests pin the properties the comparison
depends on - it stays inside the pool, it never fakes abstention, and its ranking does not
depend on the order the caller happened to pass the tools in.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from mcpr.baselines import LexicalRouter, _idf, _tokenise
from mcpr.registry import load_registry
from mcpr.types import Registry

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURES / "registry_min.json"

QUERIES = [
    "find repos that mention edge runtime in code",
    "show me the git commit log",
    "read a text file from disk",
    "fetch a url and convert it to markdown",
    "list the open issues on a repository",
    "what changed between two branches",
    "search github for python projects about mcp",
    "create a new directory for the project",
    "who are the collaborators on this repo",
    "move a file to another folder",
    "get the contents of README.md",
    "stage my changes for commit",
    "list every tag in the repository",
    "show the diff of what is staged",
    "find files matching a glob pattern",
    "look up a user by their handle",
    "what is the latest release",
    "write some text into a file",
    "list the allowed directories",
    "show the status of the working tree",
]


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry(FIXTURE_PATH)


def test_lexical_baseline_runs(registry: Registry) -> None:
    """Every query must yield a tool that is actually in the pool it was given."""
    router = LexicalRouter()
    pool_names = {spec.qualified_name for spec in registry.tools}
    for query in QUERIES:
        call = router.route(query, registry.tools)
        assert call.tool in pool_names, query
        assert call.arguments == {}


def test_ranking_is_independent_of_caller_order(registry: Registry) -> None:
    """A shuffled pool must rank identically, or the baseline's score depends on the prompt seed."""
    router = LexicalRouter()
    shuffled = list(registry.tools)
    random.Random(3407).shuffle(shuffled)
    assert router.rank(QUERIES[0], registry.tools) == router.rank(QUERIES[0], shuffled)


def test_ranking_covers_the_pool_exactly(registry: Registry) -> None:
    ranked = LexicalRouter().rank("read a file", registry.tools)
    assert [name for name, _ in ranked] != sorted(name for name, _ in ranked)  # actually ranked
    assert sorted(name for name, _ in ranked) == sorted(
        spec.qualified_name for spec in registry.tools
    )


def test_never_abstains_on_an_unmatched_query(registry: Registry) -> None:
    """A zero-score pool still returns a tool, deterministically, and never fakes `none`.

    Abstaining here would inflate `abstain_recall` for a router that is not doing anything.
    """
    call = LexicalRouter().route("zzzqqq wwwvvv", registry.tools)
    assert call.tool != "none"
    assert call.tool == min(spec.qualified_name for spec in registry.tools)


def test_empty_pool_is_survivable() -> None:
    assert LexicalRouter().route("anything", []).tool == "none"


def test_idf_is_never_negative() -> None:
    """The textbook Robertson IDF goes negative past df > N/2, inverting the ranking.

    With a pool of at most TOOLS_PER_PROMPT_MAX documents, common words clear that bar easily,
    so this is the property that keeps the baseline a fair floor rather than a strawman.
    """
    assert all(_idf(24, df) >= 0.0 for df in range(25))
    assert _idf(24, 1) > _idf(24, 20)


def test_tokenise_splits_snake_case_and_keeps_duplicates() -> None:
    assert _tokenise("github.search_code") == ["github", "search", "code"]
    assert _tokenise("Read read READ") == ["read", "read", "read"]
