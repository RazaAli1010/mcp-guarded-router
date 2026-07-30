"""The trivial retrieval baseline the fine-tune has to beat (F2 scope 7).

This exists to be beaten. F8 reports `LexicalRouter`'s `exact_tool_acc` beside the model's so a
reader can see for themselves that MCP tool routing is not keyword matching - without that
number, "the fine-tune scores X" says nothing about whether the task was hard.

Which means the baseline has to be a fair floor rather than a strawman (SPEC.md 3.5). It is
real BM25, tuned the way a small corpus needs, not a deliberately weak stand-in.

Pure python, no `rank_bm25` dependency: SPEC.md 2.5 budgets the laptop venv at 400 MB and this
is forty lines.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from mcpr.types import ToolCall, ToolSpec

#: Standard BM25 saturation and length-normalisation parameters. Local tunables, deliberately
#: not SPEC.md 7.1 constants - adding a row to that table is a Context delta, and the baseline's
#: exact configuration is not a project-wide contract.
BM25_K1 = 1.5
BM25_B = 0.75

#: Same character class as `registry._TOKEN_RE`, so the baseline and the confusability ranking
#: agree on what a word is. `_` is excluded, so `search_code` splits into `search`, `code`.
_WORD_RE = re.compile(r"[a-z0-9]+")


class LexicalRouter:
    """BM25 over `qualified_name + " " + description`, scored within one row's tool pool.

    The corpus is the pool passed to each call, never the whole registry. SPEC.md 6.4 gives
    every dataset row its own `tool_pool` and SPEC.md 9.1 defines `hallucinated_tool_rate`
    against that row's pool, so a registry-wide index could return an out-of-pool tool and make
    the baseline's accuracy incomparable to the model's - destroying the one thing it is for.
    With at most `TOOLS_PER_PROMPT_MAX` documents there is nothing to amortise anyway.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b

    def route(self, query: str, tools: Sequence[ToolSpec]) -> ToolCall:
        """Pick the highest-scoring tool in `tools`. Pure.

        Returns empty `arguments`: this baseline models tool selection only, and inventing
        argument values would make `full_call_acc` measure something other than what it claims.

        Never returns `"none"`, even when the query shares no term with any tool. The baseline
        has no abstention capability, and faking one would inflate `abstain_recall` for a router
        that is not doing anything. F8 reports the honest number instead.
        """
        ranked = self.rank(query, tools)
        return ToolCall(tool=ranked[0][0] if ranked else "none", arguments={})

    def rank(self, query: str, tools: Sequence[ToolSpec]) -> list[tuple[str, float]]:
        """Score every tool in the pool, best first. Pure.

        Sorted by score descending, ties broken by `qualified_name` ascending - the same rule as
        `registry.confusable_scores`, so an all-zero pool still ranks deterministically.
        """
        # Sorting the pool before scoring fixes the float accumulation order, making the result
        # independent of whatever shuffle the caller applied.
        ordered = sorted(tools, key=lambda spec: spec.qualified_name)
        if not ordered:
            return []

        docs = [_tokenise(f"{spec.qualified_name} {spec.description}") for spec in ordered]
        lengths = [len(doc) for doc in docs]
        avgdl = sum(lengths) / len(docs) or 1.0
        doc_freq = Counter(term for doc in docs for term in set(doc))
        total = len(docs)

        # Query terms are deduplicated rather than weighted by query-term frequency. Textbook
        # BM25 multiplies by qtf, but for a one-sentence request that only rewards accidental
        # repetition ("repos that mention repos").
        terms = list(dict.fromkeys(_tokenise(query)))

        scored: list[tuple[float, str]] = []
        for spec, doc, length in zip(ordered, docs, lengths, strict=True):
            freqs = Counter(doc)
            score = 0.0
            for term in terms:
                freq = freqs[term]
                if not freq:
                    continue
                score += (
                    _idf(total, doc_freq[term])
                    * (freq * (self.k1 + 1))
                    / (freq + self.k1 * (1 - self.b + self.b * length / avgdl))
                )
            # Negated so one ascending sort gives score desc, then name asc.
            scored.append((-score, spec.qualified_name))
        return [(name, -score) for score, name in sorted(scored)]


def _tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, in order and with duplicates. Pure.

    Returns a list, not the set `registry._tokens` returns: the duplicates *are* the term
    frequency BM25 is built on.
    """
    return _WORD_RE.findall(text.lower())


def _idf(total: int, doc_freq: int) -> float:
    """Inverse document frequency, in the `log(1 + x)` form. Pure.

    Not the textbook `log((N - df + 0.5) / (df + 0.5))`, which goes **negative** once a term
    appears in more than half the corpus. A tool pool holds at most `TOOLS_PER_PROMPT_MAX`
    documents, so ordinary words - "file", "path", "repository", "search" - clear that bar
    easily, and a negative weight would make a tool containing the query term score *worse*
    than one that does not. On a corpus this small that inverts the ranking rather than nudging
    it. This is the form Lucene ships, and it is strictly positive for every `df` in `[0, N]`.
    """
    return math.log(1 + (total - doc_freq + 0.5) / (doc_freq + 0.5))
