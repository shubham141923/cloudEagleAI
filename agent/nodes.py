"""
LangGraph Node Implementations
================================
Each function is a LangGraph node that receives the full AgentState,
performs exactly one job, and returns the mutated state.

Pipeline
--------
  START
    └─► intent_node   (LLM: extract country + requested fields)
          ├─ error ──► error_node ──► END
          └─ ok    ──► tool_node  (REST Countries API call)
                          ├─ error ──► error_node ──► END
                          └─ ok    ──► synthesis_node (LLM: compose answer)
                                           └──────────────► END

Design principles
-----------------
* NO static field dictionaries / keyword synonyms.
  The LLM does ALL semantic understanding.
* Gemini is initialised lazily — safe to import without GEMINI_API_KEY.
* Every node is a pure function: (AgentState) -> AgentState.
* Errors are always surfaced via state["error"], never raised past a node.
"""

from __future__ import annotations

import json
import logging
import os
import re

import google.generativeai as genai

from agent.state import AgentState
from agent.tools import CountryAPIError, CountryNotFoundError, fetch_country_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini – lazy singleton (safe to import without API key present)
# ---------------------------------------------------------------------------
_MODEL: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    """Return the shared Gemini model, creating it on first call."""
    global _MODEL
    if _MODEL is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file and restart the server."
            )
        genai.configure(api_key=api_key)
        _MODEL = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("Gemini model initialised.")
    return _MODEL


# ===========================================================================
# Node 1 – Intent Extraction  (pure LLM, zero static vocab)
# ===========================================================================

def intent_node(state: AgentState) -> AgentState:
    """
    Use Gemini to understand the user's question and extract:
      - country   : the country being asked about
      - fields    : free-form list of topics the user wants to know
                    (e.g. ["population", "capital", "currency"])

    No hardcoded vocabulary. The LLM reasons freely about what the user
    is asking for. The output field list is passed as-is to the synthesis
    node, which decides what to fetch from the API response.
    """
    state["steps"].append("intent_node: LLM extracting country and requested topics")
    logger.info("intent_node | query=%r", state["user_query"])

    prompt = f"""You are an intent-extraction assistant for a country information service.

Analyse the user's question and return a JSON object with exactly two keys:

  "country" : string  — The English name of the country being asked about.
                        Use the most widely recognised English name (e.g. "Germany", not "Deutschland").
                        If no specific country is mentioned, set this to null.

  "fields"  : array   — A list of PLAIN ENGLISH topic strings describing what the user
                        wants to know. Use natural phrases like:
                        "population", "capital city", "currency", "official languages",
                        "neighbouring countries", "timezone", "land area", "internet domain",
                        "calling code", "region / continent", etc.
                        If the user asks a general / overview question, return an empty array [].

Rules:
- Return ONLY the raw JSON object. No markdown, no explanation, no extra text.
- The "fields" values must be plain English phrases, not code keys.
- If you cannot identify a country at all, set "country" to null.

User question: {state["user_query"]}

JSON:"""

    try:
        response = _get_model().generate_content(prompt)
        raw = response.text.strip()
        logger.debug("intent_node | LLM output: %s", raw)

        parsed = _parse_json(raw)
        country = (parsed.get("country") or "").strip()
        fields: list[str] = [str(f).strip() for f in (parsed.get("fields") or []) if f]

        if not country:
            state["error"] = (
                "I could not identify a country in your question. "
                "Please mention a specific country name and try again."
            )
            logger.warning("intent_node | no country identified")
            return state

        state["identified_country"] = country
        state["requested_fields"] = fields  # may be empty → synthesis shows overview
        logger.info("intent_node | country=%r  fields=%s", country, fields)

    except Exception as exc:  # noqa: BLE001
        logger.exception("intent_node | error: %s", exc)
        state["error"] = (
            "I had trouble understanding your question. "
            "Could you please rephrase it?"
        )

    return state


# ===========================================================================
# Node 2 – Tool Invocation  (REST Countries API)
# ===========================================================================

def tool_node(state: AgentState) -> AgentState:
    """
    Fetch ALL available country data from the REST Countries API.

    We always fetch the full normalised record here.
    The synthesis node decides which fields to highlight based on user intent.
    Keeping this node dumb keeps the separation of concerns clean.
    """
    if state.get("error"):
        return state

    country = state["identified_country"]
    state["steps"].append(f"tool_node: calling REST Countries API for '{country}'")
    logger.info("tool_node | country=%r", country)

    try:
        data = fetch_country_data(country)
        state["raw_country_data"] = data
        logger.info("tool_node | received fields: %s", list(data.keys()))

    except CountryNotFoundError as exc:
        state["error"] = str(exc)
        logger.warning("tool_node | not found: %s", exc)

    except CountryAPIError as exc:
        state["error"] = f"Could not retrieve data from the Countries API: {exc}"
        logger.error("tool_node | API error: %s", exc)

    except Exception as exc:  # noqa: BLE001
        state["error"] = (
            "An unexpected error occurred while fetching country data. "
            "Please try again later."
        )
        logger.exception("tool_node | unexpected: %s", exc)

    return state


# ===========================================================================
# Node 3 – Answer Synthesis  (LLM composes grounded answer)
# ===========================================================================

def synthesis_node(state: AgentState) -> AgentState:
    """
    Use Gemini to compose a clear, grounded answer from the API data.

    The LLM receives:
      - The original user question
      - The requested fields (from intent_node)
      - The full normalised country data (from tool_node)

    It focuses the answer on the requested fields, but may give a short
    overview if fields is empty. It MUST NOT invent facts not in the data.
    """
    if state.get("error"):
        state["answer"] = state["error"]
        state["steps"].append("synthesis_node: propagating error as answer")
        return state

    state["steps"].append("synthesis_node: LLM composing grounded answer from API data")
    logger.info("synthesis_node | fields requested: %s", state["requested_fields"])

    data = state["raw_country_data"]
    data_json = json.dumps(data, indent=2, ensure_ascii=False)

    fields_hint = (
        f"The user specifically wants to know about: {', '.join(state['requested_fields'])}."
        if state["requested_fields"]
        else "The user asked a general question — give a concise but complete overview."
    )

    prompt = f"""You are a knowledgeable and friendly geography assistant.

{fields_hint}

Answer the user's question using ONLY the structured country data provided below.
- Focus your answer on the topics the user asked about.
- Do NOT invent, guess, or add information that is not in the data.
- If a specific topic the user asked about is genuinely missing from the data, say so explicitly.
- Write in clear, natural prose. Do not output raw JSON.
- Keep the answer concise but complete.

User question: {state["user_query"]}

Country data (all available fields):
{data_json}

Answer:"""

    try:
        response = _get_model().generate_content(prompt)
        state["answer"] = response.text.strip()
        logger.info("synthesis_node | answer: %d chars", len(state["answer"]))

    except Exception as exc:  # noqa: BLE001
        logger.exception("synthesis_node | LLM error: %s", exc)
        # Rule-based fallback — never leave the user without an answer
        state["answer"] = _rule_based_answer(data, state["requested_fields"])

    return state


# ===========================================================================
# Node 4 – Error passthrough
# ===========================================================================

def error_node(state: AgentState) -> AgentState:
    """
    Terminal node for the error path.
    Ensures 'answer' always holds a user-readable message.
    """
    if not state.get("answer"):
        state["answer"] = state.get("error") or "An unknown error occurred."
    state["steps"].append("error_node: error response finalised")
    return state


# ===========================================================================
# Private helpers
# ===========================================================================

def _parse_json(text: str) -> dict:
    """
    Robustly extract the first JSON object from LLM output.
    Handles markdown code fences and leading/trailing prose.
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Find the first complete {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)  # raise naturally if still invalid


def _rule_based_answer(data: dict, fields: list[str]) -> str:
    """
    Fallback answer builder used when the synthesis LLM call fails.
    Formats the API data as readable prose without any LLM.
    """
    name = data.get("common_name", "the country")
    lines = [f"Here is the information I found for {name}:\n"]

    def add(label: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, list):
            if value:
                lines.append(f"- {label}: {', '.join(str(v) for v in value)}")
        elif isinstance(value, int):
            lines.append(f"- {label}: {value:,}")
        else:
            lines.append(f"- {label}: {value}")

    # If user asked for specific fields, try to match them loosely
    all_data = {
        "Capital": data.get("capital"),
        "Population": data.get("population"),
        "Region": data.get("region"),
        "Sub-region": data.get("subregion"),
        "Area (km²)": data.get("area_km2"),
        "Currencies": data.get("currencies"),
        "Languages": data.get("languages"),
        "Timezones": data.get("timezones"),
        "Bordering countries": data.get("borders"),
        "Internet TLD": data.get("tld"),
        "Calling code": data.get("calling_codes"),
    }

    if fields:
        # Show only entries whose label loosely matches a requested field
        fields_lower = [f.lower() for f in fields]
        for label, value in all_data.items():
            if any(word in label.lower() for fl in fields_lower for word in fl.split()):
                add(label, value)
        if len(lines) == 1:
            # Nothing matched — show everything
            for label, value in all_data.items():
                add(label, value)
    else:
        for label, value in all_data.items():
            add(label, value)

    return "\n".join(lines)
