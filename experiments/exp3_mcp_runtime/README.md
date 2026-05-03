# Exp3 MCP Runtime

This package is the new canonical Experiment 3 runtime for the thesis.

It replaces the older `exp3_revised` simulation-first setup with a real FastMCP-backed tool layer while preserving the thesis ablation framing:

- `S0`: prompt-bloat baseline
- `S1a`: unconditioned RAG-MCP
- `S1b`: FSLSM-conditioned RAG-MCP

The benchmark split is now:

- `Exp3-Core`: new 60-question tool-balanced benchmark, 15 tools × 4 questions each
- `R2a-Transfer`: old Exp2 question set rerun only as a secondary transfer benchmark

Exp3-Core separates public question data from ground truth:

- `data/exp3_core_questions.json`: question text, fixtures, and grounding metadata
- `data/exp3_core_answer_key.json`: task `target_tool_id` labels and secondary profile-conditioned labels for evaluation only

## Thesis Positioning

This implementation uses:

- `FastMCP` as the MCP server framework
- the Python `mcp` ecosystem underneath FastMCP
- `Tavily` as the external backend for the web-search tool
- `Streamlit` as the presentation/demo layer

This is a valid MCP implementation for the thesis because tools are exposed and invoked through an actual MCP server API, not only through local prompt routing.

Selector semantics:

- `S0` selects from all tool schemas with no profile signal.
- `S1a` retrieves top-5 candidate tools from the raw question and applies query-visible intent reranking.
- `S1b` uses the same task-intent reranker plus a small FSLSM profile signal; explicit user task intent takes precedence over profile preference.
- Ground-truth labels are loaded only after selection for `TSA` scoring.
- `Task-TSA` is the primary accuracy metric. `Profile-TSA` is reported separately as a secondary personalization analysis.

## Tool Naming

There are two tool identifiers:

- `display_name`: human-readable label used in the thesis/UI
- `mcp_name`: MCP-safe tool identifier used for registration and calling

Example:

- display name: `Concept Explainer`
- MCP name: `concept_explainer`

This keeps the thesis wording readable while satisfying MCP naming constraints.

## Running

Use the repo venv so the installed `fastmcp`, `mcp`, and `tavily-python` packages are available.

### Build tool index

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/01_build_tool_index.py
```

### Verify registration

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/02_verify_setup.py
```

### Generate the Exp3-Core benchmark artifacts

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/09_generate_core_dataset.py
```

### Validate the Exp3-Core benchmark

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/10_validate_core_dataset.py
```

### Run the core dry run

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/03_dry_run.py --run-id dry_exp3_core_YYYYMMDD_HHMMSS
```

### Run the main Exp3-Core benchmark

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/11_full_run_core.py --run-id exp3_core_YYYYMMDD_HHMMSS --log-passive
```

### Run the R2a transfer benchmark

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/12_full_run_transfer.py --run-id r2a_transfer_YYYYMMDD_HHMMSS
```

### Compute benchmark-separated metrics

```bash
./venv/bin/python experiments/exp3_mcp_runtime/scripts/07_compute_metrics.py --run-id exp3_core_YYYYMMDD_HHMMSS
./venv/bin/python experiments/exp3_mcp_runtime/scripts/08_generate_report.py --run-id exp3_core_YYYYMMDD_HHMMSS
```

Timestamped thesis runs are written under `results/runs/<run_id>/` and do not append to the shared demo DB.

### Run FastMCP server over stdio

```bash
./venv/bin/python experiments/exp3_mcp_runtime/server/run_stdio.py
```

### Run FastMCP server over HTTP

```bash
./venv/bin/python experiments/exp3_mcp_runtime/server/run_http.py
```

### Run Streamlit demo

```bash
./venv/bin/streamlit run experiments/exp3_mcp_runtime/demo/app.py
```

## Notes

- The package still contains graceful fallbacks for environments where outbound API calls fail.
- Tool 15 requires `TAVILY_API_KEY` in `.env`.
- Live LLM-backed tools require `OPENAI_API_KEY` in `.env`.
- The generated `Exp3-Core` dataset is intentionally marked as draft-oriented and still requires manual review before it should be treated as the final thesis benchmark.
