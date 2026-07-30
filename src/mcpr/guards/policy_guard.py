"""Layer 2: the destructive-operation gate (SPEC.md 8.3).

Not a model. This layer never calls one, never opens a socket, and never reads the untrusted
content - its only inputs are the call, the registry, the loaded policy and the plan. That
narrowness is the point: it is the layer that still works after an injection has succeeded
(SPEC.md 13 D5), so it must not depend on having detected anything.

`check` is pure. The sandbox root arrives already resolved from `MCPR_SANDBOX_DIR` inside
`policy["paths"]["sandbox_root"]`, which is `registry.load_policy`'s job; doing that lookup here
would put an env var read inside a guard.

There is no auto-confirm. `write` and `destructive` both return `confirm`, and only a redeemed
`ConfirmToken` (see `confirm.py`) may follow.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from mcpr.registry import effect_with_source
from mcpr.registry import get as registry_get
from mcpr.types import Effect, GuardDecision, Plan, Registry, ToolCall

#: SPEC.md 8.3's ordering of effect classes. Used by the plan lock to decide "above".
EFFECT_RANK: dict[str, int] = {"read": 0, "write": 1, "destructive": 2}

#: Argument names that denote a filesystem location. The ten from the F3 spec, plus `paths` and
#: `files`: `git.git_add.files` is an array of repo-relative paths carrying no description at
#: all, so neither the name list nor `PATH_DESC_RE` would catch it, and it is the only path-shaped
#: argument of a tool that writes. This lives in code rather than config on purpose - SPEC.md 3.2.
PATH_ARG_NAMES = frozenset(
    {
        "path",
        "paths",
        "file_path",
        "filepath",
        "files",
        "source",
        "destination",
        "dest",
        "repository",
        "repo_path",
        "directory",
        "dir",
    }
)

#: Fallback for servers that name a path argument something else but describe it plainly.
PATH_DESC_RE = re.compile(r"absolute path|file path|directory", re.I)


def check(
    call: ToolCall,
    registry: Registry,
    policy: dict,
    plan: Plan | None = None,
) -> GuardDecision:
    """Classify a call's effect and gate it. Pure.

    Order is deliberate: abstention, then an unknown tool, then path containment (a `block`,
    which outranks everything below), then the plan lock, then the effect verdict.
    """
    if call.tool == "none":
        return GuardDecision(
            layer="policy",
            action="allow",
            code="abstained",
            detail="the router declined to call a tool",
        )

    try:
        spec = registry_get(call.tool, registry)
    except KeyError:
        # Layer 1 reports this too, but layer 2 must not fall through to an effect it cannot
        # derive. This is the one place the chain could leak, so both layers close it.
        return GuardDecision(
            layer="policy",
            action="block",
            code="unknown_tool",
            detail=f"{call.tool!r} is not in the registry, so its effect cannot be derived",
            evidence=[call.tool],
        )

    # The runtime derivation, never `spec.effect`: the baked field was computed by snapshot.py at
    # capture time, before config/policy.toml had any overrides in it (SPEC.md 8.1 as amended).
    effect, source = effect_with_source(spec, policy.get("effects") or {})

    paths = policy.get("paths") or {}
    if spec.server in (paths.get("sandboxed_servers") or ()):
        sandbox_root = str(paths.get("sandbox_root", ""))
        escape = _first_escape(call.arguments, spec.input_schema, sandbox_root)
        if escape is not None:
            argument, value, resolved, root = escape
            return GuardDecision(
                layer="policy",
                action="block",
                code="path_escape",
                detail=(
                    f"{argument}={value!r} resolves outside the sandbox; "
                    f"{spec.qualified_name} may only touch {root}"
                ),
                evidence=[f"{argument}={value}", f"resolved={resolved}", f"sandbox={root}"],
            )

    if plan is not None and (
        spec.server not in plan.allowed_servers
        or EFFECT_RANK[effect] > EFFECT_RANK[plan.max_effect]
    ):
        return GuardDecision(
            layer="policy",
            action="confirm",
            code="outside_plan",
            detail=(
                f"{spec.qualified_name} ({effect}) is outside the plan taken from the original "
                f"request: servers {plan.allowed_servers}, max effect {plan.max_effect}"
            ),
            evidence=[
                f"server={spec.server}",
                f"effect={effect}",
                f"plan_servers={','.join(plan.allowed_servers)}",
                f"plan_max_effect={plan.max_effect}",
            ],
        )

    if effect == "read":
        return GuardDecision(
            layer="policy",
            action="allow",
            code="read_only",
            detail=f"{spec.qualified_name} is read-only (by {source})",
            evidence=["effect=read", f"source={source}"],
        )

    return GuardDecision(
        layer="policy",
        action="confirm",
        code="destructive_requires_confirmation",
        detail=(
            f"{spec.qualified_name} has effect {effect} (by {source}) and needs a confirmation "
            "token before it may run"
        ),
        evidence=[f"effect={effect}", f"source={source}"],
    )


def effect_rank(effect: Effect) -> int:
    """Position of an effect class in SPEC.md 8.3's ordering. Pure."""
    return EFFECT_RANK[effect]


# --- internals --------------------------------------------------------------------------------


def _is_path_argument(name: str, schema: dict) -> bool:
    """Whether an argument names a filesystem location. Pure."""
    if name in PATH_ARG_NAMES:
        return True
    description = ((schema.get("properties") or {}).get(name) or {}).get("description") or ""
    return bool(PATH_DESC_RE.search(description))


def _first_escape(
    arguments: dict, schema: dict, sandbox_root: str
) -> tuple[str, str, str, str] | None:
    """First path argument resolving outside the sandbox, or `None`. Pure.

    Returns `(argument_name, raw_value, resolved_path, sandbox_root)`. Arguments are visited in
    sorted order so the reported escape does not depend on dict insertion order.
    """
    if not sandbox_root or not isinstance(arguments, dict):
        return None
    root = os.path.realpath(sandbox_root)

    for name in sorted(arguments):
        if not _is_path_argument(name, schema):
            continue
        raw = arguments[name]
        for value in raw if isinstance(raw, list) else [raw]:
            # Non-strings are layer 1's problem (`type_mismatch`), and the chain does not
            # short-circuit, so skipping here cannot let one through. Resolving them would only
            # add a misleading second code.
            if not isinstance(value, str):
                continue
            resolved = _resolve(value, root)
            if resolved is None:
                return (name, value, "<unresolvable>", root)
            if not _inside(resolved, root):
                return (name, value, resolved, root)
    return None


def _resolve(value: str, root: str) -> str | None:
    """Resolve one path argument against the sandbox root. Pure. `None` if it cannot be resolved.

    `os.path.join` is what makes this correct on both platforms, and it replaces any
    `is_absolute()` test. On Windows `Path("/etc/passwd").is_absolute()` is `False` because the
    path carries no drive, so an "if relative, resolve against the sandbox" branch would place
    `/etc/passwd` *inside* the sandbox and allow it. `os.path.isabs` is no better - its handling
    of single-slash paths changed in 3.13 and this project supports 3.11 through 3.13. `join`
    needs no such test: an absolute value replaces the root, a relative one lands under it.

    `expanduser` runs *before* the join so `~` becomes an absolute home path and escapes; after
    the join it would leave a literal `~` directory sitting inside the sandbox.
    """
    try:
        return os.path.realpath(os.path.join(root, os.path.expanduser(value)))
    except (OSError, ValueError):
        # Embedded NULs and over-long paths raise on Windows. Treating that as an escape keeps
        # the failure diagnosable; letting it propagate would surface as `guard_error`, which is
        # still fail-closed but says nothing about which argument was at fault.
        return None


def _inside(candidate: str, root: str) -> bool:
    """Whether a resolved path lies within a resolved root. Pure.

    Both sides are `realpath`ed by the callers, which is what catches a symlink inside the
    sandbox pointing out of it, and what canonicalises Windows 8.3 short names and macOS's
    `/tmp` -> `/private/tmp`. `normcase` is applied because `is_relative_to` is a case-sensitive
    string comparison while Windows preserves the input's casing for a path tail that does not
    exist yet; on POSIX it is a no-op.
    """
    return Path(os.path.normcase(candidate)).is_relative_to(Path(os.path.normcase(root)))
