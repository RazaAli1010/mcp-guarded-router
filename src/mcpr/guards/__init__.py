"""The three-layer guardrail (SPEC.md 8).

Layer 1 (`schema_guard`) validates a proposed call against the real MCP JSON Schema, layer 2
(`policy_guard`) classifies its effect and gates every mutation behind a confirmation, and
layer 3 (`injection_guard`, F4) scores untrusted tool output. `GuardChain` runs all three.

Nothing in this package calls a model, opens a socket, or reads a clock. Guardrails are code,
not prompts: a guardrail that can be talked out of its decision is not a guardrail
(SPEC.md 3.2).

TODO(F3): `GuardChain` lands with the chain wiring.
"""

from __future__ import annotations
