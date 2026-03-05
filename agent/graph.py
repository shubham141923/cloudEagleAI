"""
LangGraph Agent Graph
Assembles the three nodes into a compiled, reusable StateGraph.

Flow
----
  START → intent_node → route_after_intent
                            ├── error → error_node → END
                            └── ok   → tool_node → route_after_tool
                                                     ├── error → error_node → END
                                                     └── ok   → synthesis_node → END
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent.nodes import error_node, intent_node, synthesis_node, tool_node
from agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge helpers
# ---------------------------------------------------------------------------

def _route_after_intent(
    state: AgentState,
) -> Literal["tool_node", "error_node"]:
    if state.get("error"):
        logger.debug("Graph: intent_node → error_node (error=%r)", state["error"])
        return "error_node"
    logger.debug("Graph: intent_node → tool_node")
    return "tool_node"


def _route_after_tool(
    state: AgentState,
) -> Literal["synthesis_node", "error_node"]:
    if state.get("error"):
        logger.debug("Graph: tool_node → error_node (error=%r)", state["error"])
        return "error_node"
    logger.debug("Graph: tool_node → synthesis_node")
    return "synthesis_node"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the country-info agent graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("intent_node", intent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("synthesis_node", synthesis_node)
    graph.add_node("error_node", error_node)

    # Define edges
    graph.add_edge(START, "intent_node")

    graph.add_conditional_edges(
        "intent_node",
        _route_after_intent,
        {"tool_node": "tool_node", "error_node": "error_node"},
    )

    graph.add_conditional_edges(
        "tool_node",
        _route_after_tool,
        {"synthesis_node": "synthesis_node", "error_node": "error_node"},
    )

    graph.add_edge("synthesis_node", END)
    graph.add_edge("error_node", END)

    return graph.compile()


# Singleton compiled graph – imported once at app startup.
country_agent = build_graph()
