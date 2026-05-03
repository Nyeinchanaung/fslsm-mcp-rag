from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from experiments.exp3_mcp_runtime.core.profile_decoder import profile_to_label
from experiments.exp3_mcp_runtime.core.retriever import D2LRetriever
from experiments.exp3_mcp_runtime.server.compat import RegisteredMCPServer
from experiments.exp3_mcp_runtime.tools.tavily_search import web_search
from experiments.exp3_mcp_runtime.tools.tool_prompts import get_tool_prompt
from experiments.exp3_mcp_runtime.tools.tool_registry import TOOL_REGISTRY, get_tool_by_id
from experiments.exp3_mcp_runtime.runtime_types import ToolExecutionResult

_LLM_CLIENT: OpenAI | None = None
_LLM_UNAVAILABLE = False

load_dotenv()


def _get_llm_client() -> OpenAI | None:
    global _LLM_CLIENT
    if _LLM_UNAVAILABLE:
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenAI(api_key=api_key)
    return _LLM_CLIENT


def _format_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "No external evidence provided."
    blocks = []
    for idx, item in enumerate(evidence, 1):
        blocks.append(
            f"[{idx}] {item.get('chunk_id', item.get('source', 'source'))}\n"
            f"{item.get('text', item.get('content', ''))}"
        )
    return "\n\n".join(blocks[:3])


def _fallback_output(tool_name: str, question: str, profile_label: str, evidence: list[dict[str, Any]]) -> str:
    evidence_hint = evidence[0]["text"][:240] if evidence else "No evidence available."
    return (
        f"{tool_name} response for {profile_label}\n\n"
        f"Question: {question}\n\n"
        f"Evidence snippet: {evidence_hint}"
    )


def _llm_generate(tool_id: int, question: str, profile: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    global _LLM_UNAVAILABLE
    prompt = get_tool_prompt(
        tool_id,
        style_description=profile_to_label(profile),
        target_style=profile_to_label(profile),
    )
    client = _get_llm_client()
    if client is None:
        return _fallback_output(get_tool_by_id(tool_id).name, question, profile_to_label(profile), evidence)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt or "You are a helpful tutor."},
                {
                    "role": "user",
                    "content": (
                        f"Student profile: {profile_to_label(profile)}\n"
                        f"Question: {question}\n\n"
                        f"Evidence:\n{_format_evidence(evidence)}"
                    ),
                },
            ],
            max_tokens=700,
            timeout=60,
        )
        return response.choices[0].message.content
    except Exception:
        _LLM_UNAVAILABLE = True
        return _fallback_output(get_tool_by_id(tool_id).name, question, profile_to_label(profile), evidence)


def create_mcp_server() -> RegisteredMCPServer:
    app = RegisteredMCPServer(name="exp3-mcp-runtime")
    retriever = D2LRetriever()

    for tool in TOOL_REGISTRY:
        meta = {
            "tool_id": tool.tool_id,
            "category": tool.category,
            "fslsm_dims": tool.fslsm_dims,
            "token_cost": tool.token_cost,
            "display_name": tool.name,
            "mcp_name": tool.mcp_name,
        }

        if tool.tool_id == 14:

            @app.tool(name=tool.mcp_name, description=tool.description, metadata=meta)
            def content_retriever(
                question: str,
                fslsm_profile: dict[str, Any] | None = None,
                question_type: str = "",
                k: int = 5,
            ) -> ToolExecutionResult:
                start = time.perf_counter()
                retrieval = retriever.retrieve(question, k=k)
                return ToolExecutionResult(
                    tool_id=14,
                    tool_name="Content Retriever",
                    tool_output=retrieval["combined_text"],
                    evidence=retrieval["evidence"],
                    sources=retrieval["chunk_ids"],
                    latency_ms=(time.perf_counter() - start) * 1000,
                    token_cost_estimate=get_tool_by_id(14).token_cost,
                    execution_success=bool(retrieval["evidence"]),
                    metadata={"question_type": question_type, "profile_used_post_selection": bool(fslsm_profile)},
                )

            continue

        if tool.tool_id == 15:

            @app.tool(name=tool.mcp_name, description=tool.description, metadata=meta)
            def web_search_tool(
                question: str,
                fslsm_profile: dict[str, Any] | None = None,
                question_type: str = "",
                max_results: int = 3,
            ) -> ToolExecutionResult:
                start = time.perf_counter()
                raw = web_search(question, max_results=max_results)
                evidence = [{"source": "tavily", "content": raw, "text": raw}]
                return ToolExecutionResult(
                    tool_id=15,
                    tool_name="Web Search Tool",
                    tool_output=raw,
                    evidence=evidence,
                    sources=["tavily"],
                    latency_ms=(time.perf_counter() - start) * 1000,
                    token_cost_estimate=get_tool_by_id(15).token_cost,
                    execution_success=raw != "Search unavailable.",
                    metadata={"question_type": question_type, "profile_used_post_selection": bool(fslsm_profile)},
                )

            continue

        def _make_generic_tool(current_tool_id: int, current_tool_name: str, current_mcp_name: str):
            @app.tool(
                name=current_mcp_name,
                description=get_tool_by_id(current_tool_id).description,
                metadata={
                    "tool_id": current_tool_id,
                    "category": get_tool_by_id(current_tool_id).category,
                    "fslsm_dims": get_tool_by_id(current_tool_id).fslsm_dims,
                    "token_cost": get_tool_by_id(current_tool_id).token_cost,
                    "display_name": current_tool_name,
                    "mcp_name": current_mcp_name,
                },
            )
            def _generic_tool(
                question: str,
                fslsm_profile: dict[str, Any],
                question_type: str = "",
                evidence: list[dict[str, Any]] | None = None,
                source_text: str = "",
            ) -> ToolExecutionResult:
                start = time.perf_counter()
                evidence_list = evidence or []
                if source_text and not evidence_list:
                    evidence_list = [{"source": "source_text", "text": source_text}]
                output = _llm_generate(current_tool_id, question, fslsm_profile, evidence_list)
                return ToolExecutionResult(
                    tool_id=current_tool_id,
                    tool_name=current_tool_name,
                    tool_output=output,
                    evidence=evidence_list,
                    sources=[
                        item.get("chunk_id", item.get("source", "inline"))
                        for item in evidence_list
                    ],
                    latency_ms=(time.perf_counter() - start) * 1000,
                    token_cost_estimate=get_tool_by_id(current_tool_id).token_cost,
                    execution_success=True,
                    metadata={"question_type": question_type, "profile_used_post_selection": True},
                )

            return _generic_tool

        _make_generic_tool(tool.tool_id, tool.name, tool.mcp_name)

    return app
