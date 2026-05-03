from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable


try:  # pragma: no cover - exercised when dependency is installed
    from fastmcp import FastMCP as _FastMCP  # type: ignore
except ImportError:  # pragma: no cover - default path in this environment
    _FastMCP = None


@dataclass
class RegisteredTool:
    name: str
    description: str
    func: Callable[..., Any]
    metadata: dict[str, Any]


class RegisteredMCPServer:
    """Thin wrapper that preserves a FastMCP integration path and local execution fallback."""

    def __init__(self, name: str):
        self.name = name
        self._app = _FastMCP(name=name) if _FastMCP is not None else None
        self._tool_registry: dict[str, RegisteredTool] = {}

    def tool(self, *, name: str, description: str, metadata: dict[str, Any] | None = None):
        meta = metadata or {}

        def decorator(func: Callable[..., Any]):
            self._tool_registry[name] = RegisteredTool(
                name=name,
                description=description,
                func=func,
                metadata=meta,
            )
            if self._app is not None:
                self._app.tool(name=name, description=description, meta=meta)(func)
            return func

        return decorator

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def list_tools(self) -> list[RegisteredTool]:
        if self._app is not None:
            tools = self._run_async(self._app.list_tools())
            out: list[RegisteredTool] = []
            for tool in tools:
                meta = tool.meta or {}
                if tool.name in self._tool_registry:
                    meta = {**self._tool_registry[tool.name].metadata, **meta}
                out.append(
                    RegisteredTool(
                        name=tool.name,
                        description=tool.description or "",
                        func=self._tool_registry.get(tool.name, RegisteredTool(tool.name, tool.description or "", lambda **_: None, meta)).func,
                        metadata=meta,
                    )
                )
            return out
        return list(self._tool_registry.values())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._app is not None:
            result = self._run_async(self._app.call_tool(name, arguments))
            if hasattr(result, "structured_content") and result.structured_content is not None:
                return result.structured_content
            return result
        return self._tool_registry[name].func(**arguments)

    @property
    def fastmcp_app(self):  # pragma: no cover - passthrough helper
        return self._app
