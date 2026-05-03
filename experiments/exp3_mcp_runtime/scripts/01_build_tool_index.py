from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.exp3_mcp_runtime.tools.tool_index import ToolIndex


if __name__ == "__main__":
    idx = ToolIndex()
    idx.build()
    idx.save()
    print("Tool index built.")
