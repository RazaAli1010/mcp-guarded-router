"""The three-layer guardrail (SPEC.md 8).

Layer 1 (`schema_guard`) validates a proposed call against the real MCP JSON Schema, layer 2
(`policy_guard`) classifies its effect and gates every mutation behind a confirmation, and
layer 3 (`injection_guard`, F4) scores untrusted tool output. `GuardChain` runs all three.

Nothing in this package calls a model, opens a socket, or reads a clock - `confirm.ConfirmStore`
excepted, and its clock is injected. Guardrails are code, not prompts: a guardrail that can be
talked out of its decision is not a guardrail (SPEC.md 3.2).
"""

from __future__ import annotations

from typing import Protocol

from mcpr.types import (
    GuardAction,
    GuardChainResult,
    GuardDecision,
    GuardLayer,
    Plan,
    Registry,
    ToolCall,
    UntrustedBlock,
)

# Submodules are imported as module objects, never as `from .schema_guard import check`. A
# from-import binds the original function at import time, which would silently defeat the
# monkeypatching that `test_guard_error_fails_closed` relies on to prove the chain fails closed.
from . import policy_guard, schema_guard

#: SPEC.md 6.6's precedence: the most restrictive decision wins.
ACTION_RANK: dict[str, int] = {"allow": 0, "confirm": 1, "block": 2}

__all__ = ["ACTION_RANK", "GuardChain", "InjectionGuard", "policy_guard", "schema_guard"]


class InjectionGuard(Protocol):
    """What layer 3 must provide (F4).

    Kept structural rather than imported so that F4's `injection_guard.py` depends only on
    `types` and `config`. Importing this package from there would be a cycle.
    """

    def check(self, untrusted: list[UntrustedBlock]) -> list[GuardDecision]:
        """Score untrusted blocks and return one decision per finding."""
        ...


class GuardChain:
    """Runs every layer over one proposed call and combines their verdicts.

    Chain order is fixed at **injection -> schema -> policy** (SPEC.md 6.6): untrusted content is
    inspected before the model's output is trusted enough to validate. No layer short-circuits
    the others, because per-layer statistics have to be complete for the F8 metrics to mean
    anything - a call that fails schema validation still gets a policy verdict.
    """

    def __init__(
        self,
        registry: Registry,
        policy: dict,
        injection_guard: InjectionGuard | None = None,
        plan: Plan | None = None,
    ) -> None:
        """Hold the inputs every layer needs. Pure; nothing is read from disk or the environment.

        `injection_guard` is `None` until F4 lands, and contributes no decisions in that state.
        `policy` is the mapping from `registry.load_policy`, which has already resolved the
        sandbox root - that resolution is what lets `policy_guard.check` stay pure.
        """
        self.registry = registry
        self.policy = policy
        self.injection_guard = injection_guard
        self.plan = plan

    def check(
        self, call: ToolCall, untrusted: list[UntrustedBlock] | None = None
    ) -> GuardChainResult:
        """Run all three layers and combine them. Pure given the constructor arguments.

        `final_action` is the most restrictive decision; `blocked_by` is the code of the first
        block in chain order, or `None`. Every layer is wrapped so that an unhandled exception
        becomes `block` / `guard_error` rather than propagating - SPEC.md 3.7, fail closed. There
        is no path through this method that returns `allow` because something went wrong.
        """
        decisions: list[GuardDecision] = []

        if self.injection_guard is not None:
            decisions += self._layer(
                "injection", lambda: self.injection_guard.check(untrusted or [])
            )

        # One wrapper around both schema calls, not two: a single failure inside layer 1 should
        # produce a single guard_error, not one per function.
        decisions += self._layer("schema", lambda: self._schema(call))
        decisions += self._layer(
            "policy",
            lambda: [policy_guard.check(call, self.registry, self.policy, self.plan)],
        )

        return GuardChainResult(
            decisions=decisions,
            final_action=self._final_action(decisions),
            blocked_by=next((d.code for d in decisions if d.action == "block"), None),
        )

    # --- internals ------------------------------------------------------------------------

    def _schema(self, call: ToolCall) -> list[GuardDecision]:
        """Layer 1's verdict, plus the non-blocking undeclared-argument note when there is one."""
        found = [schema_guard.check(call, self.registry)]
        extra = schema_guard.extra_properties(call, self.registry)
        if extra is not None:
            found.append(extra)
        return found

    @staticmethod
    def _layer(layer: GuardLayer, run) -> list[GuardDecision]:
        """Run one layer, converting any escape into a blocking `guard_error`. Impure via `run`."""
        try:
            return list(run())
        except Exception as exc:  # noqa: BLE001 - SPEC.md 3.7: never fall through to execution
            return [
                GuardDecision(
                    layer=layer,
                    action="block",
                    code="guard_error",
                    detail=repr(exc),
                )
            ]

    @staticmethod
    def _final_action(decisions: list[GuardDecision]) -> GuardAction:
        """Most restrictive action across every decision. Pure.

        `max` returns the first maximal element, so ties resolve in chain order and the result is
        deterministic. An empty list cannot occur in practice - schema and policy always speak -
        but it defaults to `allow` rather than raising, and the caller sees an empty `decisions`
        list to explain it.
        """
        if not decisions:
            return "allow"
        return max(decisions, key=lambda d: ACTION_RANK[d.action]).action
