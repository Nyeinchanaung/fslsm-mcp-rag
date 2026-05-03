from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.client.runtime_client import MCPRuntimeClient
from experiments.exp3_mcp_runtime.config import CANONICAL_PROFILES_PATH, CORE_QUESTIONS_PATH
from experiments.exp3_mcp_runtime.server.app import create_mcp_server


if __name__ == "__main__":
    server = create_mcp_server()
    client = MCPRuntimeClient(server)
    tools = client.list_tools()
    print(f"Registered tools: {len(tools)}")
    for tool in tools[:5]:
        print(tool)
    print(f"Core dataset present: {CORE_QUESTIONS_PATH.exists()}")
    print(f"Canonical profiles present: {CANONICAL_PROFILES_PATH.exists()}")
