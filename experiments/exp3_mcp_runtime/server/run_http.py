from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.server.app import create_mcp_server


if __name__ == "__main__":
    server = create_mcp_server()
    app = server.fastmcp_app
    if app is None:
        raise SystemExit("FastMCP is not installed in the current interpreter.")
    app.run(transport="http", host="127.0.0.1", port=8001, show_banner=True)
