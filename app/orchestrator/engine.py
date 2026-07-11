"""The request loop. Provider-agnostic — no `if model == "ollama"`
branching anywhere in here. Never imports a specific model SDK, MCP
client, or vector DB library directly, only interfaces/registries."""
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app import glossary
from app.auth import permissions as auth
from app.logging import request_logger as logger
from app.mcp_clients import tool_router
from app.mcp_clients.registry import McpRegistry
from app.mcp_clients.registry import get_registry as get_mcp_registry
from app.model_providers.base import ModelProvider
from app.model_providers.registry import get_active as get_model_provider
from app.orchestrator import grounding
from app.orchestrator.session_store import SessionStore
from app.orchestrator.session_store import get_store as get_session_store
from app.rag import registry as rag_registry
from app.rag import router as rag_router

_NO_DOCS_ANSWER = "I don't have documentation on this — you may want to check with the team."


def _build_system_prompt(chunks: list[dict[str, Any]]) -> str:
    lines = [
        "You are a logistics operations assistant. Answer only using the "
        "provided context and available tools. If the context doesn't cover "
        "the question, say you don't have documentation on it rather than "
        "guessing.",
        "",
        glossary.as_system_prompt_block(),
    ]
    if chunks:
        lines.append("\nRelevant context:")
        for chunk in chunks:
            lines.append(f"- [{chunk['domain']}:{chunk['doc_id']}] {chunk['text']}")
    return "\n".join(lines)


async def handle_question(
    user_id: str,
    question: str,
    tags: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    request_id = request_id or str(uuid.uuid4())
    tags = tags or {}
    role = auth.get_role(user_id)

    # 1. Decide which knowledge domain(s) apply
    decision = rag_router.route(question, tags)
    decision = rag_router.RoutingDecision(
        domains=auth.filter_domains(role, decision.domains), confident=decision.confident
    )

    # 2. Retrieve relevant knowledge from each applicable domain (RAG)
    chunks: list[dict[str, Any]] = []
    for domain in decision.domains:
        result = await rag_registry.get(domain).search(question, filters=tags)
        chunks += result
        logger.log_rag_query(request_id, domain, question, result)

    # 3. Grounding check — if a domain we're confident applies has zero hits, say so explicitly
    if rag_router.expected_domain_but_no_hits(decision, chunks):
        return {"type": "final_answer", "text": _NO_DOCS_ANSWER, "sources": [], "request_id": request_id}

    # 4. Build message history
    session_store = get_session_store()
    history = await session_store.get_history(user_id)
    messages = [*history, {"role": "system", "content": _build_system_prompt(chunks)}, {"role": "user", "content": question}]

    # 5. Only the tools this role is allowed to use, further narrowed to
    #    the ones relevant to this question — a small local model becomes
    #    unreliable once too many tool schemas are attached at once.
    mcp_registry = await get_mcp_registry()
    tools = auth.filter_tools(role, mcp_registry.all_tool_schemas())
    tools = tool_router.filter_tools(tool_router.route(question), tools)

    # 6. Call the active model provider (swappable)
    provider = get_model_provider()
    return await _run_loop(provider, mcp_registry, session_store, messages, tools, chunks, user_id, question, request_id)


async def resume_after_confirmation(request_id: str, approved: bool) -> dict[str, Any]:
    session_store = get_session_store()
    pending = await session_store.get_pending(request_id)
    if pending is None:
        raise ValueError(f"No pending confirmation for request_id={request_id!r}")
    await session_store.clear_pending(request_id)

    messages: list[dict[str, Any]] = pending["messages"]
    tool = pending["tool"]
    args = pending["args"]
    call_id = pending["call_id"]
    user_id = pending["user_id"]
    question = pending["question"]
    chunks = pending["chunks"]
    tools = pending["tools"]

    mcp_registry = await get_mcp_registry()
    messages.append({"role": "assistant", "content": "", "tool_call": {"id": call_id, "name": tool, "args": args}})

    if approved:
        client = mcp_registry.get_owner(tool)
        tool_result = await client.call_tool(tool, args)
        logger.log_tool_call(request_id, tool, args, tool_result)
    else:
        tool_result = {"cancelled": True, "reason": "user declined confirmation"}

    messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})

    provider = get_model_provider()
    return await _run_loop(provider, mcp_registry, session_store, messages, tools, chunks, user_id, question, request_id)


async def _run_loop(
    provider: ModelProvider,
    mcp_registry: McpRegistry,
    session_store: SessionStore,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    user_id: str,
    question: str,
    request_id: str,
) -> dict[str, Any]:
    while True:
        result = await provider.generate(messages, tools)
        logger.log_model_call(request_id, messages, result)

        if result["type"] == "final_answer":
            grounding.verify(result["text"], chunks)
            await session_store.append(user_id, {"role": "user", "content": question})
            await session_store.append(user_id, {"role": "assistant", "content": result["text"]})
            return {"type": "final_answer", "text": result["text"], "sources": chunks, "request_id": request_id}

        # result["type"] == "tool_call"
        tool_schema = mcp_registry.get_schema(result["tool"])
        if tool_schema["safety"] == "needs_approval":
            call_id = result.get("id") or str(uuid.uuid4())
            await session_store.save_pending(
                request_id,
                {
                    "user_id": user_id,
                    "question": question,
                    "messages": messages,
                    "tool": result["tool"],
                    "args": result["args"],
                    "call_id": call_id,
                    "chunks": chunks,
                    "tools": tools,
                },
            )
            return {"type": "confirmation_required", "tool": result["tool"], "args": result["args"], "request_id": request_id}

        client = mcp_registry.get_owner(result["tool"])
        tool_result = await client.call_tool(result["tool"], result["args"])
        logger.log_tool_call(request_id, result["tool"], result["args"], tool_result)

        call_id = result.get("id") or str(uuid.uuid4())
        messages.append({"role": "assistant", "content": "", "tool_call": {"id": call_id, "name": result["tool"], "args": result["args"]}})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
        # loop again — model sees tool result, may call another tool or return final answer


async def handle_question_stream(
    user_id: str,
    question: str,
    tags: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Same steps as handle_question, but yields incremental
    {"type": "text_delta", "text": str} events as the final answer is
    generated, followed by exactly one terminal event shaped like
    handle_question's return value. Tool-calling turns are not streamed
    token-by-token (their content is empty until the decision is made) —
    only the model's concluding turn streams to the client."""
    request_id = request_id or str(uuid.uuid4())
    tags = tags or {}
    role = auth.get_role(user_id)

    decision = rag_router.route(question, tags)
    decision = rag_router.RoutingDecision(
        domains=auth.filter_domains(role, decision.domains), confident=decision.confident
    )

    chunks: list[dict[str, Any]] = []
    for domain in decision.domains:
        result = await rag_registry.get(domain).search(question, filters=tags)
        chunks += result
        logger.log_rag_query(request_id, domain, question, result)

    if rag_router.expected_domain_but_no_hits(decision, chunks):
        yield {"type": "final_answer", "text": _NO_DOCS_ANSWER, "sources": [], "request_id": request_id}
        return

    session_store = get_session_store()
    history = await session_store.get_history(user_id)
    messages = [*history, {"role": "system", "content": _build_system_prompt(chunks)}, {"role": "user", "content": question}]

    mcp_registry = await get_mcp_registry()
    tools = auth.filter_tools(role, mcp_registry.all_tool_schemas())
    tools = tool_router.filter_tools(tool_router.route(question), tools)

    provider = get_model_provider()
    async for event in _run_loop_stream(
        provider, mcp_registry, session_store, messages, tools, chunks, user_id, question, request_id
    ):
        yield event


async def _run_loop_stream(
    provider: ModelProvider,
    mcp_registry: McpRegistry,
    session_store: SessionStore,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    user_id: str,
    question: str,
    request_id: str,
) -> AsyncIterator[dict[str, Any]]:
    while True:
        stream = await provider.generate(messages, tools, stream=True)
        terminal: dict[str, Any] | None = None
        async for event in stream:
            if event["type"] == "text_delta":
                yield event
            else:
                terminal = event
        logger.log_model_call(request_id, messages, terminal)

        if terminal["type"] == "final_answer":
            grounding.verify(terminal["text"], chunks)
            await session_store.append(user_id, {"role": "user", "content": question})
            await session_store.append(user_id, {"role": "assistant", "content": terminal["text"]})
            yield {"type": "final_answer", "text": terminal["text"], "sources": chunks, "request_id": request_id}
            return

        # terminal["type"] == "tool_call"
        tool_schema = mcp_registry.get_schema(terminal["tool"])
        if tool_schema["safety"] == "needs_approval":
            call_id = terminal.get("id") or str(uuid.uuid4())
            await session_store.save_pending(
                request_id,
                {
                    "user_id": user_id,
                    "question": question,
                    "messages": messages,
                    "tool": terminal["tool"],
                    "args": terminal["args"],
                    "call_id": call_id,
                    "chunks": chunks,
                    "tools": tools,
                },
            )
            yield {"type": "confirmation_required", "tool": terminal["tool"], "args": terminal["args"], "request_id": request_id}
            return

        client = mcp_registry.get_owner(terminal["tool"])
        tool_result = await client.call_tool(terminal["tool"], terminal["args"])
        logger.log_tool_call(request_id, terminal["tool"], terminal["args"], tool_result)

        call_id = terminal.get("id") or str(uuid.uuid4())
        messages.append({"role": "assistant", "content": "", "tool_call": {"id": call_id, "name": terminal["tool"], "args": terminal["args"]}})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
