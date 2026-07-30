"""The only module in the project that opens a network or subprocess connection.

SPEC.md 3.1 makes the frozen snapshot the contract: `mcpr snapshot refresh` is the single
command allowed to reach a live MCP server, and it is the only caller of this module. Every
other component reads `schemas/registry.json`, which is what lets the whole test suite run
offline with nothing installed and nothing listening.

The v1 client API is used deliberately (SPEC.md 13 D1): `stdio_client` /
`streamablehttp_client` wrapped in a `ClientSession` with an explicit `initialize()`. `mcp`
2.x ships a rewritten `Client(target)` surface; the project pins `mcp>=1.29,<2` and must not
be "helpfully" migrated.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO, Any, TypeVar

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Implementation, TextContent

from mcpr.config import Settings, load_env, resolve
from mcpr.types import ServerConfig, ToolResult

#: Launch configuration. Not a SPEC.md 7.1 constant - kept local so config.py stays an exact
#: transcription of the spec's constant table.
SERVERS_PATH = "config/servers.toml"

#: Placeholder in `args` replaced with the resolved MCPR_SANDBOX_DIR (SPEC.md 5).
SANDBOX_TOKEN = "<MCPR_SANDBOX_DIR>"

#: Per-attempt ceiling, and the number of attempts. F1 asks for a 30 s timeout and one retry.
TIMEOUT_SECONDS = 30.0
ATTEMPTS = 2

#: `${VAR}` interpolation inside `headers_env` values.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Launcher flags that consume the following argument, so `package_name` can skip past them.
_VALUE_FLAGS = {"--with", "--from", "--python", "-p", "--index", "--constraints"}

#: Actionable hints for a missing stdio launcher, keyed by executable name.
_INSTALL_HINTS = {
    "uvx": "install uv: https://docs.astral.sh/uv/getting-started/installation/",
    "uv": "install uv: https://docs.astral.sh/uv/getting-started/installation/",
    "npx": "install Node.js: https://nodejs.org/",
    "node": "install Node.js: https://nodejs.org/",
}

T = TypeVar("T")


class McpUnavailable(RuntimeError):
    """A server could not be reached, for a reason worth printing to a human.

    Raised instead of letting a traceback escape, so `snapshot refresh` can record
    `{"status": "unavailable", "reason": ...}` for one server and carry on with the rest
    (F1 acceptance criterion 6).
    """

    def __init__(self, server_id: str, reason: str) -> None:
        super().__init__(f"{server_id}: {reason}")
        self.server_id = server_id
        self.reason = reason


def load_server_configs(
    path: str | Path = SERVERS_PATH,
    settings: Settings | None = None,
) -> dict[str, ServerConfig]:
    """Read `config/servers.toml` and resolve every launch-time placeholder.

    Impure: reads the file, `os.environ` and (via `load_env`) `.env`. Returns every table,
    enabled or not; filtering on `enabled` is the caller's decision.

    Two substitutions happen here so that nothing downstream has to know about them:
    `${VAR}` inside a header value is replaced with that environment variable, and a server
    with `sandbox_arg = true` has its last positional argument replaced with the resolved
    `MCPR_SANDBOX_DIR`. An unset `${VAR}` is deliberately left intact rather than blanked -
    the surviving placeholder is what `_require_reachable` detects to report a missing PAT
    as an actionable reason instead of an authentication failure thirty seconds later.
    """
    settings = settings or load_env()
    with open(resolve(path), "rb") as fh:
        data = tomllib.load(fh)

    sandbox = str(resolve(settings.sandbox_dir))
    configs: dict[str, ServerConfig] = {}
    for server_id, table in sorted(data.get("servers", {}).items()):
        args = [str(a).replace(SANDBOX_TOKEN, sandbox) for a in table.get("args", [])]
        sandbox_arg = bool(table.get("sandbox_arg", False))
        if sandbox_arg and args:
            args[-1] = sandbox
        headers = {
            str(name): _expand(str(value))
            for name, value in (table.get("headers_env") or {}).items()
        }
        configs[server_id] = ServerConfig(
            id=server_id,
            transport=table["transport"],
            command=str(table.get("command", "")),
            args=args,
            url=_server_url(server_id, str(table.get("url", "")), settings),
            headers=headers,
            enabled=bool(table.get("enabled", True)),
            sandbox_arg=sandbox_arg,
        )
    return configs


async def list_tools(server_id: str, cfg: ServerConfig) -> list[dict]:
    """Return every tool the server advertises, as verbatim MCP payload dicts.

    Impure: launches a subprocess or makes an HTTP request. Raises `McpUnavailable` rather
    than a transport traceback. The result is what `snapshot.build_registry` freezes, so the
    dicts keep MCP's camelCase spelling (`inputSchema`, `outputSchema`) untouched.
    """
    _, tools = await list_tools_with_info(server_id, cfg)
    return tools


async def list_tools_with_info(
    server_id: str, cfg: ServerConfig
) -> tuple[Implementation, list[dict]]:
    """`list_tools`, plus the server's own `serverInfo` from the initialize handshake.

    Impure. `serverInfo.version` is the only trustworthy source for the `package_version`
    field of `registry.meta.json` - reading it from the handshake beats shelling out to
    `uvx --version` or hardcoding what SPEC.md 4.4 says was current on capture day.
    """

    async def work(session: ClientSession, info: Implementation) -> tuple[Implementation, list]:
        tools: list[dict] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(cursor=cursor)
            tools.extend(t.model_dump(by_alias=True, exclude_none=False) for t in page.tools)
            cursor = page.nextCursor
            if cursor is None:
                return info, tools

    return await _run(server_id, cfg, work)


async def call_tool(server_id: str, cfg: ServerConfig, name: str, args: dict) -> ToolResult:
    """Execute one MCP tool and flatten its result.

    Impure. Written and smoke-tested in F1 but not wired into the pipeline until F9, so no
    guardrail runs before it - do not call it from anything but `tests/live/` yet.
    """

    async def work(session: ClientSession, _info: Implementation) -> ToolResult:
        result = await session.call_tool(name, args)
        return ToolResult(
            content_text="\n".join(_block_text(b) for b in result.content),
            is_error=bool(result.isError),
            structured=result.structuredContent,
        )

    return await _run(server_id, cfg, work)


def package_name(cfg: ServerConfig) -> str:
    """Return the distribution the server is launched from. Pure.

    Skips launcher flags and their values, so `uvx --with mcp<2 mcp-server-git --repository X`
    yields `mcp-server-git` and `npx -y @scope/pkg DIR` yields `@scope/pkg`.
    """
    if cfg.transport == "http":
        return cfg.url
    skip = False
    for arg in cfg.args:
        if skip:
            skip = False
            continue
        if arg in _VALUE_FLAGS:
            skip = True
            continue
        if arg.startswith("-"):
            continue
        return arg
    return cfg.command


def resolve_package_version(cfg: ServerConfig) -> str | None:
    """Ask a uvx-launched server's own environment what version of the package it holds.

    Impure: runs a short subprocess against the already-warm uvx cache. Returns None for any
    other launcher.

    This exists because `serverInfo.version` is not the package version for the Python
    servers: both mcp-server-fetch and mcp-server-git report the *MCP SDK* version there
    (1.29.0), which would put a plainly wrong number into the provenance file that SPEC.md
    10.3 makes metrics depend on. The npm and remote servers do report their own version, so
    `snapshot.py` falls back to `serverInfo` for those.
    """
    if cfg.command != "uvx":
        return None
    pkg = package_name(cfg)
    prefix: list[str] = []
    skip = False
    for arg in cfg.args:
        if skip:
            prefix.append(arg)
            skip = False
            continue
        if arg in _VALUE_FLAGS:
            prefix.append(arg)
            skip = True
            continue
        break
    code = f"from importlib.metadata import version;print(version({pkg!r}))"
    try:
        done = subprocess.run(  # noqa: S603 - argv is built from the checked-in config
            [cfg.command, *prefix, "--from", pkg, "python", "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS * 2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


# --- internals --------------------------------------------------------------------------------


def _expand(value: str) -> str:
    """Substitute `${VAR}` from the environment, leaving unset names in place. Impure."""
    return _VAR_RE.sub(lambda m: os.environ.get(m.group(1)) or m.group(0), value)


def _server_url(server_id: str, url: str, settings: Settings) -> str:
    """Return the endpoint for an http server. Pure given `settings`.

    SPEC.md 7.2 makes `MCPR_GITHUB_MCP_URL` configurable, so the env var beats the checked-in
    default when it is set to something other than that default.
    """
    if server_id == "github" and settings.github_mcp_url:
        return settings.github_mcp_url
    return _expand(url)


def _block_text(block: Any) -> str:
    """Render one MCP content block as text. Pure.

    Non-text blocks are summarised, never dropped: an image or resource result is still
    untrusted content that F4's injection guard has to be able to see the existence of.
    """
    if isinstance(block, TextContent):
        return block.text
    return f"[{getattr(block, 'type', 'unknown')} content]"


def _minimal_env() -> dict[str, str]:
    """Build the environment for a stdio server. Impure: reads `os.environ`.

    Deliberately *not* `os.environ.copy()`. `get_default_environment()` returns only the
    handful of variables a subprocess needs to run (PATH, HOME/USERPROFILE, TEMP, ...), which
    is what keeps `MCPR_TEACHER_API_KEY` and `MCPR_GITHUB_PAT` out of a third-party process.
    """
    return dict(get_default_environment())


def _require_reachable(server_id: str, cfg: ServerConfig) -> None:
    """Fail fast with an actionable reason before spending 30 s on a doomed connection.

    Impure: reads PATH. Raises `McpUnavailable`; never returns a value.
    """
    if not cfg.enabled:
        raise McpUnavailable(server_id, "disabled in config/servers.toml")
    if cfg.transport == "stdio":
        if not cfg.command:
            raise McpUnavailable(server_id, "no command set for a stdio server")
        if shutil.which(cfg.command) is None:
            hint = _INSTALL_HINTS.get(cfg.command, "check your PATH")
            raise McpUnavailable(server_id, f"{cfg.command} is not on PATH - {hint}")
        return
    if not cfg.url:
        raise McpUnavailable(server_id, "no url set for an http server")
    unset = sorted(
        {name for value in cfg.headers.values() for name in _VAR_RE.findall(value)},
    )
    if unset:
        raise McpUnavailable(server_id, f"{', '.join(unset)} is not set in the environment")


@asynccontextmanager
async def _connect(
    cfg: ServerConfig, errlog: IO[str]
) -> AsyncIterator[tuple[ClientSession, Implementation]]:
    """Open a session over whichever transport the config names. Impure.

    `await session.initialize()` is mandatory in the v1 API before any request; skipping it
    makes `list_tools()` hang rather than fail. A stdio server's stderr is captured into
    `errlog` instead of the terminal so a launch failure can be reported as the server's own
    message - "sandbox is not a valid Git repository" beats "Connection closed".
    """
    if cfg.transport == "stdio":
        params = StdioServerParameters(
            command=cfg.command,
            args=list(cfg.args),
            env=_minimal_env(),
        )
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as session,
        ):
            init = await session.initialize()
            yield session, init.serverInfo
    else:
        async with (
            streamablehttp_client(cfg.url, headers=cfg.headers) as (read, write, _session_id),
            ClientSession(read, write) as session,
        ):
            init = await session.initialize()
            yield session, init.serverInfo


async def _run(
    server_id: str,
    cfg: ServerConfig,
    work: Callable[[ClientSession, Implementation], Awaitable[T]],
) -> T:
    """Connect, run `work`, and clean up - under a timeout, with one retry. Impure.

    Every failure is converted to `McpUnavailable`. `BaseExceptionGroup` is caught explicitly
    because anyio's task groups - which both transports are built on - wrap the real error;
    letting a bare group escape would print a traceback where F1 asks for a reason.
    """
    _require_reachable(server_id, cfg)
    last = ""
    for attempt in range(ATTEMPTS):
        # A real file, not StringIO: stdio_client hands errlog to the subprocess, so it must
        # own a file descriptor. TemporaryFile is deleted on close.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errlog:
            try:
                async with (
                    asyncio.timeout(TIMEOUT_SECONDS),
                    _connect(cfg, errlog) as (session, info),
                ):
                    return await work(session, info)
            except TimeoutError:
                last = f"timed out after {TIMEOUT_SECONDS:.0f}s"
            except BaseExceptionGroup as group:
                last = _describe(group)
            except Exception as exc:  # noqa: BLE001 - every transport error becomes a reason
                last = _describe(exc)
            errlog.seek(0)
            stderr = _last_line(errlog.read())
        if stderr:
            last = f"{last} [server said: {stderr}]"
        if attempt + 1 < ATTEMPTS:
            await asyncio.sleep(1.0)
    raise McpUnavailable(server_id, f"{last} (after {ATTEMPTS} attempts)")


def _last_line(text: str) -> str:
    """Return the last non-blank line of captured stderr, truncated. Pure."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1][:200] if lines else ""


def _describe(exc: BaseException) -> str:
    """Flatten an exception - or an anyio exception group - into one short line. Pure."""
    if isinstance(exc, BaseExceptionGroup):
        inner = [_describe(e) for e in exc.exceptions]
        return "; ".join(dict.fromkeys(inner)) or type(exc).__name__
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0]}" if text else type(exc).__name__
