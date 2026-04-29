#!/bin/bash
# Run all 5 diagnostic scripts in sequence from the repo root

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/exp3_mcp_tool_selection"

echo "═══════════════════════════════════════════════════"
echo "  Experiment 3 Diagnostic Data Collection"
echo "═══════════════════════════════════════════════════"
echo ""

cd "$REPO_ROOT"

echo "[1/5] Exporting S1b session sample (50 sessions)..."
python experiments/exp3_mcp_tool_selection/scripts/diagnostic_01_session_sample.py
echo ""

echo "[2/5] Analyzing FAISS retrieval quality (30 S1a sessions)..."
python experiments/exp3_mcp_tool_selection/scripts/diagnostic_02_faiss_quality.py
echo ""

echo "[3/5] Exporting Experiment 2 session sample (100 sessions)..."
python experiments/exp3_mcp_tool_selection/scripts/diagnostic_03_exp2_sample.py
echo ""

echo "[4/5] Exporting tool descriptions and overlap analysis..."
python experiments/exp3_mcp_tool_selection/scripts/diagnostic_04_tool_descriptions.py
echo ""

echo "[5/5] Analyzing ground truth coverage..."
python experiments/exp3_mcp_tool_selection/scripts/diagnostic_05_ground_truth_coverage.py
echo ""

echo "═══════════════════════════════════════════════════"
echo "  ✓ All diagnostics complete"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Output files in diagnostics/:"
ls -lh "$EXP_DIR/diagnostics/"
echo ""
echo "To zip and share:"
echo "  cd $EXP_DIR && zip -r exp3_diagnostics.zip diagnostics/"
