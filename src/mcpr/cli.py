"""The `mcpr` command line.

Every sub-command namespace later features need is reserved here up front, so no feature
invents its own spelling. Only `version` and `doctor` are implemented in F0; the rest raise
NotImplementedError naming the feature that fills them in.
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sys
import tomllib
from dataclasses import dataclass

import httpx
import typer
from rich.console import Console
from rich.table import Table

from mcpr import __version__
from mcpr.config import PROJECT_ROOT, PROMPT_VERSION, REGISTRY_PATH, Settings, load_env, resolve
from mcpr.mcp_client import load_server_configs
from mcpr.registry import confusable_scores, load_registry
from mcpr.snapshot import capture, diff_registries, write_snapshot
from mcpr.types import Registry

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
    styles = {"read": "green", "write": "yellow", "destructive": "red"}
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


@guard_app.command("check")
def guard_check() -> None:
    """Run the injection -> schema -> policy chain over a call. TODO(F3/F4)."""
    _stub("F3/F4")


@eval_app.command("score")
def eval_score() -> None:
    """Compute the SPEC.md 9 metrics from a predictions file. TODO(F8)."""
    _stub("F8")


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
