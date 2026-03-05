"""
FastAPI Application — Country Information AI Agent (API-only)
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load .env before importing agent modules (they read env vars at import time)
load_dotenv()

from agent import AgentState, country_agent  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Country Information AI Agent",
    description=(
        "LangGraph-powered agent that answers natural-language questions "
        "about countries using the public REST Countries API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language question about a country.",
        examples=["What is the population of Germany?"],
    )


class QueryResponse(BaseModel):
    answer: str
    country: str | None = None
    fields_queried: list[str] = []
    processing_steps: list[str] = []
    latency_ms: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Ops"])
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def query_country(body: QueryRequest) -> QueryResponse:
    """
    Accept a natural-language question, run it through the LangGraph agent,
    and return a grounded answer with metadata.

    **Example questions:**
    - What is the population of Germany?
    - What currency does Japan use?
    - What is the capital and population of Brazil?
    - Which countries border France?
    """
    logger.info("Received query: %r", body.question)
    t0 = time.perf_counter()

    initial_state: AgentState = {
        "user_query": body.question,
        "identified_country": None,
        "requested_fields": [],
        "raw_country_data": None,
        "answer": None,
        "error": None,
        "steps": [],
    }

    try:
        final_state: AgentState = await country_agent.ainvoke(initial_state)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled agent error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="The agent encountered an unexpected error. Please try again.",
        ) from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Query complete | country=%r fields=%s latency=%.0fms",
        final_state.get("identified_country"),
        final_state.get("requested_fields"),
        latency_ms,
    )

    return QueryResponse(
        answer=final_state.get("answer") or "No answer could be generated.",
        country=final_state.get("identified_country"),
        fields_queried=final_state.get("requested_fields") or [],
        processing_steps=final_state.get("steps") or [],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/examples", tags=["Agent"])
async def get_examples() -> dict:
    """Return sample questions to test the API."""
    return {
        "examples": [
            "What is the population of Germany?",
            "What currency does Japan use?",
            "What is the capital and population of Brazil?",
            "What languages are spoken in Switzerland?",
            "Which countries border France?",
            "What timezone is Australia in?",
            "What is the area of Canada in square kilometers?",
            "Tell me everything about New Zealand.",
        ]
    }
