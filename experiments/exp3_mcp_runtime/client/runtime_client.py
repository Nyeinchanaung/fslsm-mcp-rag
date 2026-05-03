from __future__ import annotations

from typing import Any

from experiments.exp3_mcp_runtime.server.compat import RegisteredMCPServer


class MCPRuntimeClient:
    def __init__(self, server: RegisteredMCPServer):
        self.server = server

    def list_tools(self) -> list[dict[str, Any]]:
        out = []
        for tool in self.server.list_tools():
            out.append(
                {
                    "name": tool.name,
                    "display_name": tool.metadata.get("display_name", tool.name),
                    "description": tool.description,
                    **tool.metadata,
                }
            )
        return out

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.server.call_tool(tool_name, arguments)
        return result.to_dict() if hasattr(result, "to_dict") else result
