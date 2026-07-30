"""The one test in the project that is allowed to contact a real MCP server.

Marked `live` and excluded by `addopts = "-q -m 'not live'"`, so `make check` never needs a
server, a network connection or a warm uvx cache (SPEC.md 12). Run it deliberately:

    pytest -m live

It exists to catch the case the frozen snapshot cannot: that `mcp_client` still speaks a
protocol the real servers accept. Only `fetch` is contacted - it is the smallest server and
needs no repository, no Node and no PAT.
"""

from __future__ import annotations

import pytest

from mcpr.mcp_client import list_tools_with_info, load_server_configs

pytestmark = pytest.mark.live


async def test_fetch_server_lists_at_least_one_tool() -> None:
    cfg = load_server_configs()["fetch"]
    info, tools = await list_tools_with_info("fetch", cfg)

    assert len(tools) >= 1
    assert info.name
    # Verbatim MCP spelling: the payload must still be camelCase when it reaches us.
    assert all("inputSchema" in tool for tool in tools)
    assert all(tool["name"] for tool in tools)
