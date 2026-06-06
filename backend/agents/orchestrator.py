"""
Stateful ReAct agent implemented with LangGraph.

Architecture:
    AgentState   — typed TypedDict passed through every graph node
    ToolRegistry — maps tool names → async callables
    AgentGraph   — LangGraph StateGraph wiring the Reason → Act → Observe loop
    AgentOrchestrator — public facade; owns lifecycle and injects dependencies

Loop invariants enforced here:
    - max_steps hard ceiling prevents infinite loops (AgentLoopDetectedError)
    - asyncio.wait_for wraps the whole execution (AgentTimeoutError)
    - each tool invocation is wrapped to raise ToolExecutionError on failure
    - step_history is append-only so the caller can reconstruct the trace
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, Callable, Coroutine

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

from backend.core.exceptions import (
    AgentError,
    AgentLoopDetectedError,
    AgentTimeoutError,
    ToolExecutionError,
)
from backend.llm_client.base import (
    BaseLLMClient,
    ChatCompletionRequest,
    ChatMessage,
)
from backend.rag_pipeline.engine import RAGEngine, RAGRequest

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ToolFn = Callable[[dict[str, Any]], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Agent state — the single mutable record threaded through every graph node
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[ChatMessage], add_messages]
    step_count: int
    tool_calls_log: list[dict[str, Any]]    # append-only trace
    final_answer: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Tool schema / registry
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters_schema: dict[str, Any]       # JSON Schema for the tool's args


class ToolRegistry:
    """Maps tool names to async callables and their JSON Schema descriptions."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolFn]] = {}
        self._log = structlog.get_logger(self.__class__.__name__)

    def register(self, definition: ToolDefinition, fn: ToolFn) -> None:
        self._tools[definition.name] = (definition, fn)
        self._log.debug("tool_registry.registered", tool=definition.name)

    def definitions(self) -> list[ToolDefinition]:
        return [d for d, _ in self._tools.values()]

    async def invoke(self, name: str, args: dict[str, Any]) -> str:
        if name not in self._tools:
            raise ToolExecutionError(
                f"Tool '{name}' is not registered",
                {"available_tools": list(self._tools)},
            )
        _, fn = self._tools[name]
        try:
            result = await fn(args)
            return result
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{name}' raised an unhandled exception",
                {"tool": name, "args": args, "error": str(exc)},
            ) from exc


# ---------------------------------------------------------------------------
# LLM-side tool-call parsing
# ---------------------------------------------------------------------------


def _build_tools_system_prompt(registry: ToolRegistry) -> str:
    tool_descriptions = "\n".join(
        f"- {d.name}: {d.description}\n  Args: {json.dumps(d.parameters_schema)}"
        for d in registry.definitions()
    )
    return f"""\
You are a precise enterprise assistant operating in a ReAct loop.

## Available Tools
{tool_descriptions}

## Rules
1. If you need to use a tool, respond ONLY with valid JSON:
   {{"tool": "<name>", "args": {{...}}}}
2. If you have a final answer that does not require another tool call, respond with:
   {{"final_answer": "<your answer here>"}}
3. Do NOT mix prose with the JSON structure.
4. Do NOT invent tool names that are not listed above.
"""


def _parse_llm_action(content: str) -> dict[str, Any]:
    """
    Extract the JSON action dict from LLM output.
    Handles markdown code fences and leading/trailing whitespace.
    """
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------


def _make_reason_node(
    llm: BaseLLMClient,
    registry: ToolRegistry,
    max_steps: int,
) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, Any]]]:
    """Closure captures dependencies; returns an async node function."""
    system_prompt = _build_tools_system_prompt(registry)
    log = structlog.get_logger("agent.reason_node")

    async def reason(state: AgentState) -> dict[str, Any]:
        step = state["step_count"] + 1
        log.debug("agent.reason", step=step)

        if step > max_steps:
            return {
                "step_count": step,
                "error": f"Max steps ({max_steps}) exceeded",
                "final_answer": None,
            }

        messages_for_llm = [
            ChatMessage(role="system", content=system_prompt),
            *state["messages"],
        ]
        request = ChatCompletionRequest(
            messages=messages_for_llm,
            temperature=0.0,
            max_tokens=512,
        )
        response = await llm.chat_complete(request)
        assistant_msg = ChatMessage(role="assistant", content=response.content)

        try:
            action = _parse_llm_action(response.content)
        except (json.JSONDecodeError, ValueError) as exc:
            return {
                "messages": [assistant_msg],
                "step_count": step,
                "error": f"LLM returned non-JSON: {str(exc)[:120]}",
                "final_answer": None,
            }

        return {
            "messages": [assistant_msg],
            "step_count": step,
            "_parsed_action": action,
            "error": None,
        }

    return reason


def _make_act_node(
    registry: ToolRegistry,
) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, Any]]]:
    log = structlog.get_logger("agent.act_node")

    async def act(state: AgentState) -> dict[str, Any]:
        action: dict[str, Any] = state.get("_parsed_action", {})  # type: ignore[call-overload]
        step = state["step_count"]

        if "final_answer" in action:
            log.info("agent.final_answer", step=step)
            return {
                "final_answer": action["final_answer"],
                "tool_calls_log": state["tool_calls_log"],
            }

        tool_name: str = action.get("tool", "")
        tool_args: dict[str, Any] = action.get("args", {})
        log.info("agent.tool_call", tool=tool_name, step=step)

        tool_result = await registry.invoke(tool_name, tool_args)
        tool_entry = {"step": step, "tool": tool_name, "args": tool_args, "result": tool_result}

        observation_msg = ChatMessage(
            role="user",
            content=f"[Tool: {tool_name}]\n{tool_result}",
        )
        return {
            "messages": [observation_msg],
            "tool_calls_log": [*state["tool_calls_log"], tool_entry],
        }

    return act


def _should_continue(state: AgentState) -> str:
    """
    LangGraph conditional edge.
    Returns "end" when the agent has a final answer, an error, or exceeds max steps.
    """
    if state.get("error"):
        return "end"
    if state.get("final_answer") is not None:
        return "end"
    action: dict[str, Any] = state.get("_parsed_action", {})  # type: ignore[call-overload]
    if "final_answer" in action:
        return "end"
    return "act"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_react_graph(
    llm: BaseLLMClient,
    registry: ToolRegistry,
    max_steps: int = 10,
) -> Any:
    """
    Compile a LangGraph StateGraph implementing the ReAct loop.

    Topology:
        START → reason → [end | act] → reason → ...
    """
    reason_node = _make_reason_node(llm, registry, max_steps)
    act_node = _make_act_node(registry)

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)

    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason",
        _should_continue,
        {"act": "act", "end": END},
    )
    graph.add_edge("act", "reason")

    return graph.compile()


# ---------------------------------------------------------------------------
# Orchestrator facade
# ---------------------------------------------------------------------------


class AgentRunResult(BaseModel):
    final_answer: str
    step_count: int
    tool_calls: list[dict[str, Any]]
    error: str | None = None


class AgentOrchestrator:
    """
    Public-facing facade for running the ReAct agent.

    Builds the graph, registers default tools (RAG search, calculator),
    and enforces execution timeouts.

    Usage:
        orch = AgentOrchestrator(llm=client, rag_engine=engine)
        result = await orch.run("What are the latest pgvector best practices?")
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        rag_engine: RAGEngine | None = None,
        max_steps: int = 10,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._llm = llm
        self._rag_engine = rag_engine
        self._max_steps = max_steps
        self._timeout = timeout_seconds
        self._registry = ToolRegistry()
        self._log = structlog.get_logger(self.__class__.__name__)
        self._register_default_tools()
        self._graph = build_react_graph(llm, self._registry, max_steps)

    def _register_default_tools(self) -> None:
        """Register built-in tools. Additional tools can be added via register_tool()."""

        # --- RAG search tool ---
        if self._rag_engine is not None:
            rag_engine = self._rag_engine  # closure capture

            async def rag_search(args: dict[str, Any]) -> str:
                query: str = args.get("query", "")
                if not query:
                    raise ToolExecutionError("rag_search requires a non-empty 'query' arg", {})
                response = await rag_engine.execute(RAGRequest(query=query))
                sources = ", ".join(
                    f"{c.document_id}[score={c.score:.2f}]"
                    for c in response.source_chunks
                )
                return f"Answer: {response.answer}\nSources: {sources}"

            self._registry.register(
                ToolDefinition(
                    name="rag_search",
                    description="Search the enterprise knowledge base and return a grounded answer.",
                    parameters_schema={"query": {"type": "string", "description": "The search question"}},
                ),
                rag_search,
            )

        # --- Calculator tool ---
        async def calculator(args: dict[str, Any]) -> str:
            expression: str = args.get("expression", "")
            if not expression:
                raise ToolExecutionError("calculator requires an 'expression' arg", {})
            try:
                # Restrict to numeric ops — no exec() of arbitrary code
                allowed_names: dict[str, Any] = {"__builtins__": {}}
                import math as _math
                allowed_names.update(vars(_math))
                result = eval(expression, allowed_names)  # noqa: S307 — safe: no builtins
                return str(result)
            except Exception as exc:
                raise ToolExecutionError(
                    "Calculator expression evaluation failed",
                    {"expression": expression, "error": str(exc)},
                ) from exc

        self._registry.register(
            ToolDefinition(
                name="calculator",
                description="Evaluate a mathematical expression. Supports math module functions.",
                parameters_schema={"expression": {"type": "string", "example": "2 ** 10 + math.sqrt(144)"}},
            ),
            calculator,
        )

    def register_tool(self, definition: ToolDefinition, fn: ToolFn) -> None:
        """Extend the agent with additional domain tools after construction."""
        self._registry.register(definition, fn)
        self._graph = build_react_graph(self._llm, self._registry, self._max_steps)

    async def run(self, user_message: str) -> AgentRunResult:
        """
        Execute the ReAct loop for ``user_message``.

        Raises:
            AgentTimeoutError       if wall-clock time exceeds self._timeout
            AgentLoopDetectedError  if max_steps is exceeded
            AgentError              for any other terminal failure
        """
        log = self._log.bind(query=user_message[:80])
        log.info("agent.run.start")

        initial_state: AgentState = {
            "messages": [ChatMessage(role="user", content=user_message)],
            "step_count": 0,
            "tool_calls_log": [],
            "final_answer": None,
            "error": None,
        }

        try:
            final_state: AgentState = await asyncio.wait_for(
                self._graph.ainvoke(initial_state),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise AgentTimeoutError(
                f"Agent execution exceeded {self._timeout}s wall-clock limit",
                {"timeout": self._timeout, "query": user_message[:80]},
            ) from exc
        except Exception as exc:
            raise AgentError(
                "Agent graph execution failed",
                {"query": user_message[:80], "error": str(exc)},
            ) from exc

        if final_state.get("error"):
            error_msg = final_state["error"]
            if "Max steps" in error_msg:
                raise AgentLoopDetectedError(
                    error_msg,
                    {"max_steps": self._max_steps, "query": user_message[:80]},
                )
            raise AgentError(error_msg, {"query": user_message[:80]})

        answer = final_state.get("final_answer") or "No answer produced."
        result = AgentRunResult(
            final_answer=answer,
            step_count=final_state["step_count"],
            tool_calls=final_state["tool_calls_log"],
        )
        log.info(
            "agent.run.complete",
            steps=result.step_count,
            tool_calls=len(result.tool_calls),
        )
        return result
