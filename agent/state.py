"""
Agent State Definition
Defines the shared state that flows through all LangGraph nodes.
"""

from __future__ import annotations

from typing import Any, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state across all agent nodes.

    Attributes
    ----------
    user_query : str
        The raw question submitted by the user.

    identified_country : Optional[str]
        The country name extracted by the intent node.

    requested_fields : List[str]
        Plain-English topics the LLM identified from the user's question,
        e.g. ["population", "capital city", "official languages"].
        This is LLM-generated — no fixed vocabulary or enum.
        An empty list means the user asked a general overview question.

    raw_country_data : Optional[dict]
        Full JSON payload returned by the REST Countries API.

    answer : Optional[str]
        Final human-readable answer produced by the synthesis node.

    error : Optional[str]
        Non-empty when something goes wrong at any stage.

    steps : List[str]
        Ordered log of processing steps (useful for observability / tracing).
    """

    user_query: str
    identified_country: Optional[str]
    requested_fields: List[str]
    raw_country_data: Optional[dict]
    answer: Optional[str]
    error: Optional[str]
    steps: List[str]
