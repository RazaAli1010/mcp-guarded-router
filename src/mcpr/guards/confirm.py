"""The confirmation gate behind every write and destructive call (SPEC.md 8.3).

A `confirm` verdict from `policy_guard` is not advice. Execution happens only on a second call
carrying a token issued for *that exact call*, and the token is single-use and TTL-bounded.
There is no auto-confirm flag, no `--yes`, and no environment variable that skips this. Claim C3
rests on there being no such path.

**Stated simplification.** This store lives in one process's memory and binds a token to a call
and nothing else. A real deployment would persist it and bind each token to an authenticated
user session, so that a token issued to one user cannot be redeemed by another, and so that a
restart does not silently drop pending confirmations. That is out of scope here, but it is a
simplification rather than an oversight, and it belongs in the README's limitations section.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable

from mcpr.config import CONFIRM_TTL_SECONDS
from mcpr.types import ConfirmToken, ToolCall

#: Bytes of entropy behind each token. `secrets`, never `random`: SPEC.md 3.3's seeded-RNG rule
#: is about reproducible *data*, and a reproducible confirmation token would be a forgeable one.
TOKEN_BYTES = 32


def call_hash(call: ToolCall) -> str:
    """Hex sha256 binding a token to one exact call. Pure.

    Serialises the **raw** `{tool, arguments}` with sorted keys and compact separators. Sorting
    is the only normalisation applied, and it is safe because JSON object key order carries no
    meaning.

    This deliberately does **not** use `parse.canonicalise_arguments`. That function serves
    `arg_exact_acc` (SPEC.md 9.1) and is lossy by design: it collapses whitespace inside strings
    and drops keys whose value equals a schema default. Binding a confirmation to that form would
    mean a token approved for

        content = "cleanup()\\n\\n# rm -rf / is commented out"

    could be redeemed against

        content = "cleanup() # rm -rf / is commented out"

    because the two canonicalise identically - which is exactly the substitution `call_hash`
    exists to prevent. It would also be unable to tell "the user explicitly passed dryRun: false"
    from "the key was absent". A metric may forgive whitespace; a gate on a destructive operation
    binds to the bytes that will execute. The two notions of equality must never be unified.
    """
    payload = json.dumps(
        {"tool": call.tool, "arguments": call.arguments},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConfirmStore:
    """In-memory store of outstanding confirmations. Single-use, TTL-bounded, hash-bound.

    Impure by nature - it holds state and reads a clock - which is why the clock is injectable
    rather than called directly, so TTL behaviour can be tested without sleeping.
    """

    def __init__(
        self,
        ttl_seconds: float = CONFIRM_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build an empty store.

        `clock` defaults to `time.monotonic`, not `time.time`: a wall clock can be stepped
        backwards by NTP or a daylight-saving change, and a TTL that gates a destructive
        operation must not be extendable by moving the system clock.
        """
        self._ttl = float(ttl_seconds)
        self._clock = clock
        self._pending: dict[str, ConfirmToken] = {}

    def issue(self, call: ToolCall) -> ConfirmToken:
        """Mint a token for one call. Impure."""
        self.purge()
        now = self._clock()
        token = ConfirmToken(
            token=secrets.token_urlsafe(TOKEN_BYTES),
            call_hash=call_hash(call),
            issued_at=now,
            expires_at=now + self._ttl,
        )
        self._pending[token.token] = token
        return token

    def redeem(self, token: str, call: ToolCall) -> bool:
        """Spend a token against a call. Impure. `True` only if it may proceed.

        Four conditions, all required: the token exists, it has not already been spent, it has
        not expired, and its `call_hash` matches the call being made.

        A hash mismatch returns `False` **without** consuming the entry. Burning it would let
        anyone who can submit one wrong call destroy a pending confirmation the user is about to
        approve - turning a mis-typed argument, or an injected call, into a denial of service
        against the gate itself. Only a genuine success spends the token.
        """
        pending = self._pending.get(token)
        if pending is None:
            return False
        if self._clock() >= pending.expires_at:
            del self._pending[token]
            return False
        if not secrets.compare_digest(pending.call_hash, call_hash(call)):
            return False
        del self._pending[token]
        return True

    def purge(self) -> int:
        """Drop every expired token and return how many went. Impure.

        Called on each `issue`, so an abandoned confirmation cannot accumulate indefinitely.
        """
        now = self._clock()
        expired = [key for key, tok in self._pending.items() if now >= tok.expires_at]
        for key in expired:
            del self._pending[key]
        return len(expired)

    def __len__(self) -> int:
        """How many tokens are currently held, expired ones included until the next purge."""
        return len(self._pending)
