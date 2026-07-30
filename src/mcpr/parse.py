"""The strict reader for whatever the router actually emitted (SPEC.md 6.2).

`parse_router_output` never raises. It is the first thing that touches model output, and
SPEC.md 3.7 says the guard chain fails closed - a parser that threw would turn a malformed
generation into a crashed evaluation run rather than a recorded data point.

It is also deliberately diagnostic rather than boolean. The distribution over
`ParseResult.error_code` *is* claim C2 - fine-tuning fixes format adherence, which is where
small models actually fail - so collapsing six distinct failure modes into "invalid" would
delete the result this module exists to produce.
"""

from __future__ import annotations

import json
import re

from mcpr.types import ParseResult, ToolCall

#: Matches an opening markdown fence with an optional language tag. The tag class excludes `{`
#: so the pattern can never swallow the object itself, and the newline is optional because
#: models truncated at MAX_GEN_TOKENS emit one-line and unclosed fences too.
_FENCE_OPEN = re.compile(r"\A```[A-Za-z0-9_+.-]*[ \t]*\r?\n?")

#: How many stray `{` positions to probe when the whole string is not valid JSON. Capped, and
#: the cap is load-bearing: `json.decoder.errmsg` rescans the document to compute line/column
#: for every JSONDecodeError it raises, so an uncapped scan over a pathological input is O(n^2)
#: and blows the 60-second suite budget of SPEC.md 12. A model that emits more than eight stray
#: braces before its answer has produced garbage, not a recoverable object.
_MAX_PREFIX_PROBES = 8

_DECODER = json.JSONDecoder()

_MISSING = object()


def parse_router_output(raw: str) -> ParseResult:
    """Parse one router generation into a `ToolCall`, or say precisely why it is not one. Pure.

    Never raises, for any input, including non-UTF-8-decodable noise and inputs deep enough to
    exhaust the C recursion limit.

    Checks the envelope before the content. SPEC.md 6.2 states the contract as "exactly one line
    containing exactly one JSON object, no prose" - the envelope clause comes first and is the
    coarser failure - so `[1,2] and here is why` is `extra_text`, not `not_object`.

    A markdown code fence is stripped and the result still counts as `ok`; models emit them
    constantly and it is a formatting tic, not a routing error. `ParseResult.raw` keeps the
    original string un-stripped so F8 can report fence frequency as its own number.
    """
    try:
        text = _strip_fence(raw)
        if not text:
            return ParseResult(ok=False, error_code="invalid_json", raw=raw)

        value, end, offset = _find_first_value(text)
        if value is _MISSING:
            return ParseResult(ok=False, error_code="invalid_json", raw=raw)
        if offset > 0:
            return ParseResult(ok=False, error_code="extra_text", raw=raw)

        rest = text[end:].strip()
        if rest:
            code = "multiple_objects" if _decodes(rest) else "extra_text"
            return ParseResult(ok=False, error_code=code, raw=raw)

        if not isinstance(value, dict):
            return ParseResult(ok=False, error_code="not_object", raw=raw)
        if "tool" not in value or "arguments" not in value:
            return ParseResult(ok=False, error_code="missing_keys", raw=raw)
        tool, arguments = value["tool"], value["arguments"]
        if isinstance(tool, bool) or not isinstance(tool, str) or not isinstance(arguments, dict):
            return ParseResult(ok=False, error_code="bad_types", raw=raw)

        # Constructed with explicit keywords, never ToolCall(**value): ToolCall forbids extras,
        # so a model that helpfully added "reasoning" would make this raise. Extra keys are
        # accepted and reported ok - there is no ParseErrorCode for them (the Literal is closed,
        # and widening it is a Context delta), folding them into bad_types would poison the C2
        # error-code distribution with something that is not a type error, and F3's schema guard
        # validates `arguments` regardless. A correct call with a stray key is a correct call.
        return ParseResult(ok=True, call=ToolCall(tool=tool, arguments=arguments), raw=raw)
    except Exception:  # noqa: BLE001 - SPEC.md 3.7: fail closed, never propagate.
        return ParseResult(ok=False, error_code="invalid_json", raw=raw)


# --- internals --------------------------------------------------------------------------------


def _strip_fence(raw: str) -> str:
    """Remove a surrounding markdown code fence and outer whitespace. Pure, never raises."""
    text = raw.strip()
    match = _FENCE_OPEN.match(text)
    if not match:
        return text
    text = text[match.end() :].rstrip()
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _find_first_value(text: str) -> tuple[object, int, int]:
    """Decode the first JSON value in `text`, returning `(value, end_index, start_offset)`.

    Pure. Returns `(_MISSING, 0, 0)` when nothing decodes. Tries the whole string first, then
    probes at successive `{` positions so that leading prose is diagnosed as `extra_text`
    rather than collapsing to `invalid_json`.

    Probes `{` only, never `[`: the contract is a JSON *object*, and rescuing the `[1]` out of
    `use tool [1] here` would blur the error-code taxonomy for no benefit.

    `RecursionError` is caught alongside `ValueError` at every decode site because deeply
    nested input raises it straight out of the C scanner, and it is not a `ValueError`.
    """
    try:
        value, end = _DECODER.raw_decode(text)
        return value, end, 0
    except (ValueError, RecursionError):
        pass

    probes = 0
    index = text.find("{", 1)
    while index != -1 and probes < _MAX_PREFIX_PROBES:
        try:
            value, end = _DECODER.raw_decode(text, index)
            return value, end, index
        except (ValueError, RecursionError):
            probes += 1
        index = text.find("{", index + 1)
    return _MISSING, 0, 0


def _decodes(text: str) -> bool:
    """Whether `text` is itself a complete JSON value. Pure, never raises."""
    try:
        json.loads(text)
    except (ValueError, RecursionError):
        return False
    return True
