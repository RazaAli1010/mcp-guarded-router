"""Capture real MCP `tools/list` output into the frozen snapshot.

This is the only writer of `schemas/registry.json` and `schemas/registry.meta.json`, and -
through `mcp_client` - the only path in the project that reaches a live server (SPEC.md 3.1).

The central rule is **verbatim**: `inputSchema` and `outputSchema` are stored exactly as the
servers returned them. No reformatting, no key renaming, no filling in of missing `type`
fields, no "cleanup" of schemas that look wrong. Real MCP servers emit messy, inconsistent
schemas, and teaching a router to cope with the mess is the point of the project - a tidied
snapshot would train and evaluate against a world that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from mcpr.config import REGISTRY_PATH, resolve
from mcpr.io import sha256_file, write_json
from mcpr.mcp_client import (
    McpUnavailable,
    list_tools_with_info,
    load_server_configs,
    package_name,
    resolve_package_version,
)
from mcpr.registry import effect_from_parts, load_effect_overrides
from mcpr.types import Registry, ServerConfig, ToolSpec

#: Provenance file that sits beside the snapshot.
REGISTRY_META_PATH = "schemas/registry.meta.json"


@dataclass
class ServerReport:
    """What happened with one server during a capture. Ordinary dataclass, not a contract."""

    id: str
    transport: str
    package: str
    status: str  # "ok" | "unavailable"
    package_version: str | None = None
    tool_count: int = 0
    annotation_coverage: float = 0.0
    reason: str | None = None


@dataclass
class CaptureResult:
    """The registry plus the per-server provenance needed to write the meta file."""

    registry: Registry
    reports: list[ServerReport] = field(default_factory=list)

    @property
    def ok_servers(self) -> list[ServerReport]:
        """Reports for servers that actually answered. Pure."""
        return [r for r in self.reports if r.status == "ok"]


@dataclass
class RegistryDiff:
    """Added, removed and changed `qualified_name`s between two snapshots."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    @property
    def any_change(self) -> bool:
        """True when the two snapshots differ at all. Pure."""
        return bool(self.added or self.removed or self.changed)


async def build_registry(server_ids: list[str]) -> Registry:
    """Capture the named servers and return the assembled registry.

    Impure: contacts live MCP servers. Servers that cannot be reached are skipped silently
    here; use `capture` when the per-server reasons matter, which they do for the meta file.
    """
    return (await capture(server_ids)).registry


async def capture(server_ids: list[str], raw_dir: Path | None = None) -> CaptureResult:
    """Contact each server, convert its tools, and record what happened to each.

    Impure. A server that raises `McpUnavailable` becomes a `status="unavailable"` report and
    the capture continues - SPEC.md's F1 notes require the project to be completable with no
    GitHub PAT, just with a smaller confusable set.

    When `raw_dir` is given, each server's untouched `tools/list` payload is also written
    there. That dump is what `test_input_schema_verbatim` compares the stored schemas
    against, so the "no helpful reformatting" rule is checked against real bytes.

    Tools are sorted by `qualified_name`, which is the primary key everywhere downstream.
    """
    configs = load_server_configs()
    overrides = load_effect_overrides()
    specs: list[ToolSpec] = []
    reports: list[ServerReport] = []

    for server_id in server_ids:
        cfg = configs.get(server_id)
        if cfg is None:
            reports.append(
                ServerReport(
                    id=server_id,
                    transport="?",
                    package="?",
                    status="unavailable",
                    reason="not defined in config/servers.toml",
                )
            )
            continue
        reports.append(await _capture_one(server_id, cfg, overrides, specs, raw_dir))

    specs.sort(key=lambda s: s.qualified_name)
    return CaptureResult(registry=Registry(version=1, tools=specs), reports=reports)


def to_tool_spec(server: str, raw: dict, overrides: dict[str, str]) -> ToolSpec:
    """Convert one verbatim MCP tool payload into a `ToolSpec`. Pure.

    `input_schema` and `output_schema` are the very objects the server sent - assigned, not
    rebuilt - so a byte-comparison against the raw payload succeeds. `description` defaults to
    the empty string because MCP marks it optional while `ToolSpec` requires it, and an
    absent description is itself a fact worth training against.
    """
    name = raw["name"]
    qualified_name = f"{server}.{name}"
    annotations = raw.get("annotations") or {}
    return ToolSpec(
        server=server,
        name=name,
        qualified_name=qualified_name,
        title=raw.get("title"),
        description=raw.get("description") or "",
        input_schema=raw["inputSchema"],
        output_schema=raw.get("outputSchema"),
        annotations=annotations,
        effect=effect_from_parts(qualified_name, name, annotations, overrides),
    )


def build_meta(result: CaptureResult, registry_path: str | Path = REGISTRY_PATH) -> dict:
    """Assemble `schemas/registry.meta.json`. Impure: hashes the registry file on disk.

    Must be called *after* the registry has been written, since `sha256_registry` is the
    digest of the real bytes - that is the number SPEC.md 10.3 makes every metric depend on.

    `annotation_coverage` is reported per server because SPEC.md 8.1 rule 3 exists precisely
    for servers that ship no annotations; without the number, a reader cannot tell whether an
    effect was derived from a hint or guessed from a verb.
    """
    return {
        "captured_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sdk_version": version("mcp"),
        "servers": [
            {
                "id": r.id,
                "transport": r.transport,
                "package": r.package,
                "package_version": r.package_version,
                "tool_count": r.tool_count,
                "annotation_coverage": r.annotation_coverage,
                **({} if r.status == "ok" else {"status": r.status, "reason": r.reason}),
            }
            for r in result.reports
        ],
        "tool_count": len(result.registry.tools),
        "sha256_registry": sha256_file(resolve(registry_path)),
        "effect_counts": effect_counts(result.registry),
    }


def effect_counts(registry: Registry) -> dict[str, int]:
    """Count tools per effect class, always with all three keys present. Pure."""
    counts = {"read": 0, "write": 0, "destructive": 0}
    for spec in registry.tools:
        counts[spec.effect] += 1
    return counts


def write_snapshot(result: CaptureResult, registry_path: str | Path = REGISTRY_PATH) -> dict:
    """Write both snapshot files and return the meta dict. Impure.

    Order matters: the registry is written first so the meta file can hash the real bytes.
    Both go through `io.write_json`, which is what guarantees sorted keys, LF newlines and a
    trailing newline on every platform (SPEC.md 3.3).
    """
    write_json(resolve(registry_path), result.registry)
    meta = build_meta(result, registry_path)
    write_json(resolve(REGISTRY_META_PATH), meta)
    return meta


def diff_registries(old: Registry, new: Registry) -> RegistryDiff:
    """Compare two snapshots by `qualified_name`. Pure.

    A tool counts as changed when anything about it moved - schema, description, annotations
    or derived effect - because SPEC.md 13 D3 accepts that a frozen snapshot ages and asks
    `snapshot diff` to make that drift visible rather than silent.
    """
    old_by_name = {s.qualified_name: s for s in old.tools}
    new_by_name = {s.qualified_name: s for s in new.tools}
    return RegistryDiff(
        added=sorted(set(new_by_name) - set(old_by_name)),
        removed=sorted(set(old_by_name) - set(new_by_name)),
        changed=sorted(
            name
            for name in set(old_by_name) & set(new_by_name)
            if old_by_name[name].model_dump() != new_by_name[name].model_dump()
        ),
    )


# --- internals --------------------------------------------------------------------------------


async def _capture_one(
    server_id: str,
    cfg: ServerConfig,
    overrides: dict[str, str],
    specs: list[ToolSpec],
    raw_dir: Path | None = None,
) -> ServerReport:
    """Capture one server, appending its tools to `specs`. Impure; never raises."""
    package = package_name(cfg)
    try:
        info, raw_tools = await list_tools_with_info(server_id, cfg)
    except McpUnavailable as exc:
        return ServerReport(
            id=server_id,
            transport=cfg.transport,
            package=package,
            status="unavailable",
            reason=exc.reason,
        )

    if raw_dir is not None:
        write_json(raw_dir / f"{server_id}.json", raw_tools)

    tools = [to_tool_spec(server_id, raw, overrides) for raw in raw_tools]
    specs.extend(tools)
    annotated = sum(1 for t in tools if t.annotations)
    return ServerReport(
        id=server_id,
        transport=cfg.transport,
        package=package,
        status="ok",
        # The Python servers report the MCP SDK version in serverInfo, so ask uvx for the
        # real distribution version and fall back to serverInfo only where it is meaningful.
        package_version=resolve_package_version(cfg) or info.version,
        tool_count=len(tools),
        annotation_coverage=round(annotated / len(tools), 4) if tools else 0.0,
    )
