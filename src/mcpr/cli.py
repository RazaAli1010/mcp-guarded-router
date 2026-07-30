"""The `mcpr` command line.

Every sub-command namespace later features need is reserved here up front, so no feature
invents its own spelling. Only `version` and `doctor` are implemented in F0; the rest raise
NotImplementedError naming the feature that fills them in.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import shutil
import sys
import time
import tomllib
from collections import Counter
from dataclasses import dataclass

import httpx
import typer
from rich.console import Console
from rich.table import Table

from mcpr import __version__
from mcpr.config import PROJECT_ROOT, PROMPT_VERSION, REGISTRY_PATH, Settings, load_env, resolve
from mcpr.guards import GuardChain
from mcpr.mcp_client import load_server_configs
from mcpr.prompt import (
    PromptTooLong,
    build_router_prompt,
    estimate_tokens,
    sample_tool_pool,
)
from mcpr.registry import (
    POLICY_PATH,
    PolicyError,
    confusable_scores,
    effect_with_source,
    load_policy,
    load_registry,
)
from mcpr.snapshot import capture, diff_registries, write_snapshot
from mcpr.types import Effect, EffectSource, Plan, Registry, ToolCall, ToolSpec

#: Shared colour vocabulary, so a `destructive` effect and a `block` action read the same way in
#: every table this module prints.
EFFECT_STYLES: dict[str, str] = {"read": "green", "write": "yellow", "destructive": "red"}
ACTION_STYLES: dict[str, str] = {"allow": "green", "confirm": "yellow", "block": "red"}

# Import names of the SPEC.md 2.4 forbidden set. Dashes become underscores; these are the
# names `find_spec` is asked about, not the distribution names on PyPI.
FORBIDDEN_IMPORTS = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "unsloth",
    "sentence_transformers",
    "vllm",
    "xformers",
    "flash_attn",
    "safetensors",
    "tokenizers",
]

# Distribution names as they would appear in pyproject.toml.
FORBIDDEN_DISTS = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "unsloth",
    "sentence-transformers",
    "vllm",
    "xformers",
    "flash-attn",
    "safetensors",
    "tokenizers",
]

# Import name -> the dependency it comes from, for a useful failure message.
LAPTOP_IMPORTS = {
    "mcp": "mcp",
    "pydantic": "pydantic",
    "jsonschema": "jsonschema",
    "typer": "typer",
    "rich": "rich",
    "dotenv": "python-dotenv",
    "httpx": "httpx",
    "tomli_w": "tomli-w",
}

MIN_FREE_DISK_BYTES = 2 * 1024**3

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A fine-tuned MCP tool router with a three-layer guardrail.",
)


def _stub(feature: str) -> None:
    """Raise for a sub-command whose implementation belongs to a later feature. Pure."""
    raise NotImplementedError(f"implemented in {feature}")


# --- Reserved sub-command namespaces --------------------------------------------------------

snapshot_app = typer.Typer(no_args_is_help=True, help="Tool-schema snapshot management (F1).")
data_app = typer.Typer(no_args_is_help=True, help="Dataset synthesis, labelling, splits (F6).")
guard_app = typer.Typer(no_args_is_help=True, help="Run the guardrail chain over inputs (F3/F4).")
eval_app = typer.Typer(no_args_is_help=True, help="Scoring of prediction files (F8).")
run_app = typer.Typer(no_args_is_help=True, help="End-to-end dispatch and demo (F9).")
report_app = typer.Typer(no_args_is_help=True, help="Metrics, figures, results.md (F10).")
models_app = typer.Typer(no_args_is_help=True, help="Optional local GGUF weights (SPEC.md 10.4).")

app.add_typer(snapshot_app, name="snapshot")
app.add_typer(data_app, name="data")
app.add_typer(guard_app, name="guard")
app.add_typer(eval_app, name="eval")
app.add_typer(run_app, name="run")
app.add_typer(report_app, name="report")
app.add_typer(models_app, name="models")


@snapshot_app.command("refresh")
def snapshot_refresh(
    servers: str = typer.Option(
        "",
        "--servers",
        help="Comma-separated server ids. Defaults to every server enabled in servers.toml.",
    ),
    raw_dir: str = typer.Option(
        "",
        "--raw-dir",
        help="Also dump each server's verbatim tools/list payload here, for fixture building.",
    ),
) -> None:
    """Contact the configured MCP servers and rewrite schemas/registry.json.

    The only command in the project allowed to open an MCP connection (SPEC.md 3.1). Exits 0
    when at least one server answered, recording the others as unavailable; exits 1 only when
    none did.
    """
    configs = load_server_configs()
    ids = (
        [s.strip() for s in servers.split(",") if s.strip()]
        if servers
        else [sid for sid, cfg in configs.items() if cfg.enabled]
    )
    console = Console()
    if not ids:
        console.print("[red]no servers selected and none enabled in config/servers.toml[/]")
        raise typer.Exit(code=1)

    result = asyncio.run(capture(ids, raw_dir=resolve(raw_dir) if raw_dir else None))
    meta = write_snapshot(result)

    table = Table(title="mcpr snapshot refresh", title_justify="left")
    table.add_column("server", no_wrap=True)
    table.add_column("transport", no_wrap=True)
    # The GitHub remote reports a build SHA as its version, so both of these are capped -
    # an unbounded column squeezes every other one into illegibility.
    table.add_column("package", overflow="ellipsis", max_width=34)
    table.add_column("version", overflow="ellipsis", max_width=22)
    table.add_column("tools", no_wrap=True)
    table.add_column("annotated", no_wrap=True)
    table.add_column("status", overflow="fold")
    for report in result.reports:
        ok = report.status == "ok"
        table.add_row(
            report.id,
            report.transport,
            report.package,
            report.package_version or "-",
            str(report.tool_count),
            f"{report.annotation_coverage:.0%}",
            "[green]ok[/]" if ok else f"[yellow]unavailable[/]: {report.reason}",
        )
    console.print(table)
    console.print(
        f"{meta['tool_count']} tools -> {REGISTRY_PATH} "
        f"(sha256 {meta['sha256_registry'][:12]}...), effects {meta['effect_counts']}"
    )
    if not result.ok_servers:
        console.print("[red]no server could be reached; the snapshot was not updated[/]")
        raise typer.Exit(code=1)


@snapshot_app.command("show")
def snapshot_show(
    server: str = typer.Option("", "--server", help="Only tools from this server."),
    effect: str = typer.Option("", "--effect", help="Only tools with this derived effect."),
) -> None:
    """Print the frozen snapshot as a table. Reads the file only - never the network."""
    tools = load_registry().tools
    if server:
        tools = [t for t in tools if t.server == server]
    if effect:
        tools = [t for t in tools if t.effect == effect]

    table = Table(title=f"{len(tools)} tools", title_justify="left")
    table.add_column("qualified_name", no_wrap=True)
    table.add_column("effect", no_wrap=True)
    table.add_column("description", overflow="ellipsis", max_width=70)
    styles = EFFECT_STYLES
    for spec in tools:
        first_line = spec.description.strip().splitlines()
        table.add_row(
            spec.qualified_name,
            f"[{styles[spec.effect]}]{spec.effect}[/]",
            first_line[0] if first_line else "",
        )
    Console().print(table)


@snapshot_app.command("diff")
def snapshot_diff(old: str = typer.Argument(..., help="Path to an earlier registry.json.")) -> None:
    """Report drift between an earlier snapshot and the current one. Exits 1 on any change.

    SPEC.md 13 D3 accepts that a frozen snapshot ages; this is what makes the ageing visible
    rather than silent, so a later session notices that a server changed under it.
    """
    console = Console()
    current = load_registry()
    previous = Registry.model_validate_json(resolve(old).read_text(encoding="utf-8"))
    delta = diff_registries(previous, current)

    for label, names, colour in (
        ("added", delta.added, "green"),
        ("removed", delta.removed, "red"),
        ("changed", delta.changed, "yellow"),
    ):
        for name in names:
            console.print(f"[{colour}]{label:8}[/] {name}")
    if not delta.any_change:
        console.print("[green]no drift[/]: the two snapshots are identical")
        return
    console.print(
        f"{len(delta.added)} added, {len(delta.removed)} removed, {len(delta.changed)} changed"
    )
    raise typer.Exit(code=1)


@snapshot_app.command("confusables")
def snapshot_confusables(
    qualified_name: str = typer.Argument(..., help="e.g. github.search_code"),
    k: int = typer.Option(8, "-k", help="How many neighbours to print."),
) -> None:
    """Print the tools most easily mistaken for this one, with their scores."""
    console = Console()
    try:
        ranked = confusable_scores(qualified_name, k)
    except KeyError:
        console.print(f"[red]{qualified_name} is not in {REGISTRY_PATH}[/]")
        raise typer.Exit(code=1) from None

    table = Table(title=f"confusables for {qualified_name}", title_justify="left")
    table.add_column("#", no_wrap=True)
    table.add_column("qualified_name", no_wrap=True)
    table.add_column("score", no_wrap=True)
    for rank, (name, score) in enumerate(ranked, start=1):
        table.add_row(str(rank), name, f"{score:.4f}")
    Console().print(table)


@data_app.command("synth")
def data_synth() -> None:
    """Synthesise raw queries from the tool registry. TODO(F6)."""
    _stub("F6")


@data_app.command("label")
def data_label() -> None:
    """Label raw queries with the configured teacher model. TODO(F6)."""
    _stub("F6")


@data_app.command("verify")
def data_verify() -> None:
    """Human verification CLI for the test split. TODO(F6)."""
    _stub("F6")


@data_app.command("split")
def data_split() -> None:
    """Write train/val/test splits. TODO(F6)."""
    _stub("F6")


@dataclass(frozen=True)
class AuditRow:
    """One tool's effect derivation, as `mcpr guard audit` reports it.

    `baked_effect` is what `schemas/registry.json` stores; `effect` is what the guards actually
    use. They differ wherever an override was added after the snapshot was captured, which is
    expected and reported rather than repaired (SPEC.md 8.1).
    """

    qualified_name: str
    server: str
    effect: Effect
    source: EffectSource
    why: str
    baked_effect: Effect

    @property
    def unpinned(self) -> bool:
        """A destructive tool resting on nothing but a verb guess. This is what exits 1."""
        return self.effect == "destructive" and self.source == "heuristic"


def audit_rows(registry: Registry, overrides: dict[str, str]) -> list[AuditRow]:
    """Derive every tool's effect and its provenance, sorted by `qualified_name`. Pure.

    Factored out of the command so the test for the exit-1 condition asserts the same
    computation the CLI exits on, rather than a parallel reimplementation of it.
    """
    rows = []
    for spec in registry.tools:
        effect, source = effect_with_source(spec, overrides)
        rows.append(
            AuditRow(
                qualified_name=spec.qualified_name,
                server=spec.server,
                effect=effect,
                source=source,
                why=_derivation_reason(spec, effect, source),
                baked_effect=spec.effect,
            )
        )
    return sorted(rows, key=lambda r: r.qualified_name)


def _derivation_reason(spec: ToolSpec, effect: Effect, source: EffectSource) -> str:
    """Human-readable account of which signal decided an effect. Pure."""
    if source == "override":
        return "config/policy.toml [effects]"
    if source == "annotation":
        return "destructiveHint" if effect == "destructive" else "readOnlyHint"
    verb = spec.name.split("_")[0].lower()
    return f"verb {verb!r}" if effect != "write" else f"verb {verb!r} in neither list"


@guard_app.command("check")
def guard_check(
    tool: str = typer.Option(..., "--tool", help="qualified_name, or 'none' to abstain."),
    args: str = typer.Option("{}", "--args", help="Arguments as a JSON object."),
    plan_servers: str = typer.Option(
        "", "--plan-servers", help="Comma-separated servers the plan allows; enables the lock."
    ),
    plan_max_effect: str = typer.Option(
        "read", "--plan-max-effect", help="Highest effect the plan allows: read|write|destructive."
    ),
    policy_path: str = typer.Option(POLICY_PATH, "--policy", help="Path to policy.toml."),
) -> None:
    """Run the injection -> schema -> policy chain over one call and print every decision.

    Exits 0 only when the chain allows the call. `confirm` and `block` both exit 1: a caller must
    not proceed on a confirmation any more than on a block, and SPEC.md 3.7 says fail closed.
    Usage errors exit 2, which Click supplies.
    """
    console = Console()
    try:
        arguments = json.loads(args)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--args is not valid JSON: {exc}", param_hint="--args") from None
    if not isinstance(arguments, dict):
        raise typer.BadParameter("--args must be a JSON object", param_hint="--args")

    registry = load_registry()
    try:
        policy = load_policy(policy_path, registry=registry)
    except PolicyError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None

    plan = None
    if plan_servers:
        if plan_max_effect not in EFFECT_STYLES:
            raise typer.BadParameter(
                f"--plan-max-effect must be one of {sorted(EFFECT_STYLES)}",
                param_hint="--plan-max-effect",
            )
        plan = Plan(
            allowed_servers=[s.strip() for s in plan_servers.split(",") if s.strip()],
            max_effect=plan_max_effect,  # type: ignore[arg-type]
            created_at=time.time(),
        )

    result = GuardChain(registry, policy, plan=plan).check(ToolCall(tool=tool, arguments=arguments))

    table = Table(title=f"guard chain for {tool}", title_justify="left")
    table.add_column("layer", no_wrap=True)
    table.add_column("action", no_wrap=True)
    table.add_column("code", no_wrap=True)
    table.add_column("detail", overflow="fold")
    table.add_column("evidence", overflow="fold")
    for decision in result.decisions:
        style = ACTION_STYLES[decision.action]
        table.add_row(
            decision.layer,
            f"[{style}]{decision.action}[/]",
            decision.code,
            decision.detail,
            "\n".join(decision.evidence),
        )
    console.print(table)

    console.print_json(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    console.print(
        f"[{ACTION_STYLES[result.final_action]}]final_action={result.final_action}[/]  "
        f"blocked_by={result.blocked_by or '-'}  decisions={len(result.decisions)}"
    )
    if result.final_action != "allow":
        raise typer.Exit(code=1)


@guard_app.command("audit")
def guard_audit(
    policy_path: str = typer.Option(POLICY_PATH, "--policy", help="Path to policy.toml."),
    server: str = typer.Option("", "--server", help="Only tools from this server."),
    source: str = typer.Option("", "--source", help="Only override|annotation|heuristic."),
) -> None:
    """Show every tool's effect and which of SPEC.md 8.1's three tiers decided it.

    Exits 1 if any tool is `destructive` by heuristic alone. A guess from a verb prefix must
    never be the only thing standing between the router and a delete: those tools have to be
    pinned by an explicit `[effects]` override.
    """
    console = Console()
    registry = load_registry()
    try:
        policy = load_policy(policy_path, registry=registry)
    except PolicyError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None

    rows = audit_rows(registry, policy["effects"])
    shown = [
        r for r in rows if (not server or r.server == server) and (not source or r.source == source)
    ]

    table = Table(title=f"effect derivation ({REGISTRY_PATH})", title_justify="left")
    table.add_column("qualified_name", no_wrap=True)
    table.add_column("effect", no_wrap=True)
    table.add_column("source", no_wrap=True)
    table.add_column("why", overflow="fold")
    for row in shown:
        table.add_row(
            row.qualified_name,
            f"[{EFFECT_STYLES[row.effect]}]{row.effect}[/]",
            f"[red]{row.source}[/]" if row.unpinned else row.source,
            row.why,
        )
    console.print(table)

    counts = Counter(r.effect for r in rows)
    sources = Counter(r.source for r in rows)
    console.print(
        f"{len(rows)} tools: "
        + ", ".join(f"{counts[e]} {e}" for e in ("read", "write", "destructive"))
        + " | sources: "
        + ", ".join(f"{sources[s]} {s}" for s in ("override", "annotation", "heuristic"))
    )

    drifted = [r for r in rows if r.effect != r.baked_effect]
    if drifted:
        # Not a failure. snapshot.py bakes ToolSpec.effect at capture time, so an override added
        # afterwards necessarily disagrees with the frozen file. SPEC.md 8.1 says the runtime
        # derivation wins and the snapshot is not rewritten - but the divergence is never silent.
        console.print(
            f"[yellow]{len(drifted)} tool(s) differ from the effect baked into the snapshot[/]: "
            + ", ".join(f"{r.qualified_name} {r.baked_effect}->{r.effect}" for r in drifted)
        )

    unpinned = [r for r in rows if r.unpinned]
    if unpinned:
        console.print(
            f"[red]{len(unpinned)} destructive tool(s) derived from the name heuristic alone[/]: "
            + ", ".join(r.qualified_name for r in unpinned)
        )
        console.print("pin these with an explicit [effects] override in config/policy.toml")
        raise typer.Exit(code=1)
    console.print("[green]ok[/]: no tool is destructive by heuristic alone")


@eval_app.command("score")
def eval_score() -> None:
    """Compute the SPEC.md 9 metrics from a predictions file. TODO(F8)."""
    _stub("F8")


@run_app.command("prompt")
def run_prompt(
    query: str = typer.Option(..., "--query", help="The natural-language request to route."),
    gold: str = typer.Option(
        "", "--gold", help="Anchor the sampled catalog on this tool. Omit for an abstain pool."
    ),
    seed: int = typer.Option(0, "--seed", help="Seeds both the catalog draw and the tool order."),
) -> None:
    """Render one router prompt and print it with its hash.

    The fastest way for a human to eyeball drift: if the render changes, this output and its
    hash change with it. Reads the frozen snapshot only - never the network.
    """
    console = Console()
    registry = load_registry()
    rng = random.Random(seed)
    try:
        tools = sample_tool_pool(gold or "none", registry, rng)
    except KeyError:
        console.print(f"[red]{gold} is not in {REGISTRY_PATH}[/]")
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None

    try:
        prompt = build_router_prompt(query, tools, seed=seed)
    except PromptTooLong as exc:
        console.print(f"[red]{exc}[/]")
        console.print("[yellow]try a different --seed, which draws a smaller catalog[/]")
        raise typer.Exit(code=1) from None

    # soft_wrap keeps rich from folding the tool lines at terminal width. The whole point of
    # this command is to show the render as the model receives it, and SPEC.md 6.3 puts exactly
    # one compact JSON line per tool - a wrapped copy would misrepresent that.
    console.print(prompt.system, highlight=False, soft_wrap=True)
    console.print()
    console.print(prompt.user, highlight=False, soft_wrap=True)
    console.print(
        f"[green]tools[/] {len(prompt.tool_names)}  "
        f"[green]est_tokens[/] {estimate_tokens(prompt.system + prompt.user)}  "
        f"[green]prompt_hash[/] {prompt.prompt_hash}"
    )


@run_app.command("demo")
def run_demo() -> None:
    """End-to-end routing + dispatch against the configured backend. TODO(F9)."""
    _stub("F9")


@report_app.command("build")
def report_build() -> None:
    """Generate reports/results.md and figures from reports/metrics/*.json. TODO(F10)."""
    _stub("F10")


@models_app.command("pull")
def models_pull() -> None:
    """Download the optional router GGUF into models/. TODO(F7)."""
    _stub("F7")


# --- Implemented in F0 ----------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the package version and the prompt-format version."""
    Console().print(f"mcpr {__version__} (prompt format {PROMPT_VERSION})")


@dataclass
class Check:
    """One row of the doctor table."""

    name: str
    status: str
    detail: str


def _check_python() -> Check:
    """Verify the running interpreter is inside requires-python. Impure: reads sys."""
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    ok = (major, minor) >= (3, 11) and (major, minor) < (3, 14)
    return Check(
        "python >=3.11,<3.14",
        PASS if ok else FAIL,
        version_str if ok else f"{version_str} is outside the supported range",
    )


def _check_laptop_deps() -> Check:
    """Verify every laptop dependency is importable. Impure: inspects the import system."""
    missing = [
        dist for mod, dist in LAPTOP_IMPORTS.items() if importlib.util.find_spec(mod) is None
    ]
    if missing:
        return Check("laptop deps importable", FAIL, f"missing: {', '.join(missing)}")
    return Check("laptop deps importable", PASS, f"all {len(LAPTOP_IMPORTS)} present")


def _check_forbidden_declared() -> Check:
    """Verify no forbidden package is declared in pyproject.toml. Impure: reads the file."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return Check("no forbidden pkg declared", WARN, "pyproject.toml not found")
    with open(pyproject, "rb") as fh:
        data = tomllib.load(fh)
    project = data.get("project", {})
    declared = list(project.get("dependencies", []))
    declared += list(project.get("optional-dependencies", {}).get("dev", []))
    names = {_requirement_name(spec) for spec in declared}
    leaked = sorted(names & set(FORBIDDEN_DISTS))
    if leaked:
        return Check("no forbidden pkg declared", FAIL, f"declared: {', '.join(leaked)}")
    return Check("no forbidden pkg declared", PASS, f"{len(declared)} laptop deps clean")


def _check_forbidden_importable() -> Check:
    """Verify no forbidden package is installed.

    Impure: inspects the import system. Uses `find_spec` rather than `try: import` so a
    half-installed distribution - which still has a spec but may fail to import - is caught.
    """
    found = [name for name in FORBIDDEN_IMPORTS if importlib.util.find_spec(name) is not None]
    if found:
        return Check("no forbidden pkg importable", FAIL, f"installed: {', '.join(found)}")
    return Check("no forbidden pkg importable", PASS, f"none of {len(FORBIDDEN_IMPORTS)} present")


def _check_on_path(exe: str, *, required: bool, why: str) -> Check:
    """Verify an executable is on PATH. Impure: reads PATH."""
    found = shutil.which(exe)
    if found:
        return Check(f"{exe} on PATH", PASS, found)
    return Check(f"{exe} on PATH", FAIL if required else WARN, why)


def _check_sandbox(settings: Settings) -> Check:
    """Verify MCPR_SANDBOX_DIR exists and is actually writable.

    Impure: creates the directory if absent and writes a probe file. A write probe, not a
    permission-bit read: it is the only check that behaves correctly under Windows ACLs, and
    creating the directory keeps `doctor` green on a fresh clone where sandbox/ is gitignored.
    """
    path = resolve(settings.sandbox_dir)
    probe = path / ".mcpr_write_probe"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("sandbox dir writable", FAIL, f"{path}: {exc.strerror or exc}")
    return Check("sandbox dir writable", PASS, str(path))


def _check_disk() -> Check:
    """Verify free space on the repo's filesystem. Impure: stats the filesystem."""
    free = shutil.disk_usage(PROJECT_ROOT).free
    gb = free / 1024**3
    ok = free >= MIN_FREE_DISK_BYTES
    return Check("free disk >= 2 GB", PASS if ok else FAIL, f"{gb:.1f} GB free")


def _check_endpoint(label: str, base_url: str, model: str, api_key: str) -> Check:
    """Ping an OpenAI-compatible endpoint with a 1-token completion.

    Impure: makes a network request. Skipped with WARN when unconfigured, so `doctor` never
    fails offline on a fresh checkout with an empty .env.
    """
    name = f"{label} endpoint"
    if not base_url or not model:
        return Check(name, WARN, "skipped: base URL or model not configured")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=15.0)
    except httpx.HTTPError as exc:
        return Check(name, FAIL, f"{type(exc).__name__}: {exc}")
    if response.is_success:
        return Check(name, PASS, f"{model} responded {response.status_code}")
    return Check(name, FAIL, f"HTTP {response.status_code} from {url}")


def _check_github_pat(settings: Settings) -> Check:
    """Report whether a GitHub PAT is configured. Impure: reads loaded settings."""
    if settings.github_pat:
        return Check("MCPR_GITHUB_PAT set", PASS, "present")
    return Check("MCPR_GITHUB_PAT set", WARN, "unset: the github MCP server cannot be snapshotted")


def _requirement_name(spec: str) -> str:
    """Extract the distribution name from a PEP 508 requirement string.

    Pure. Strips extras and version specifiers so `torch>=2.0` cannot slip past a name match.
    """
    name = spec.strip()
    for sep in ("[", "(", ";", "=", "<", ">", "!", "~", " "):
        name = name.split(sep)[0]
    return name.strip().lower()


@app.command()
def doctor() -> None:
    """Report whether every external prerequisite is reachable. Exits non-zero only on FAIL."""
    settings = load_env()
    checks = [
        _check_python(),
        _check_laptop_deps(),
        _check_forbidden_declared(),
        _check_forbidden_importable(),
        _check_on_path("uv", required=True, why="required to launch the uvx MCP servers"),
        _check_on_path("node", required=False, why="optional: the filesystem MCP server needs it"),
        _check_on_path("npx", required=False, why="optional: the filesystem MCP server needs it"),
        _check_sandbox(settings),
        _check_disk(),
        _check_endpoint(
            "teacher",
            settings.teacher_base_url,
            settings.teacher_model,
            settings.teacher_api_key,
        ),
        _check_endpoint(
            "baseline",
            settings.baseline_base_url,
            settings.baseline_model,
            settings.baseline_api_key,
        ),
        _check_github_pat(settings),
    ]

    styles = {PASS: "green", WARN: "yellow", FAIL: "red"}
    table = Table(title="mcpr doctor", title_justify="left")
    table.add_column("check", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail", overflow="fold")
    for check in checks:
        table.add_row(check.name, f"[{styles[check.status]}]{check.status}[/]", check.detail)

    console = Console()
    console.print(table)

    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]
    console.print(
        f"{len(checks) - len(failures) - len(warnings)} pass, "
        f"{len(warnings)} warn, {len(failures)} fail"
    )
    if failures:
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point wrapper. Impure: runs the CLI."""
    app()


if __name__ == "__main__":
    main()
