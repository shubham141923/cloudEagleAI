# GlobeIQ – Country Information AI Agent

> **LangGraph-powered AI agent** that answers natural-language questions about countries using the public [REST Countries API](https://restcountries.com).  
> Pure **API-only** backend — no frontend, fully async, production-ready.

---

## Architecture Overview

```
User Request (POST /query)
         │
         ▼
┌──────────────────────────────────────────────┐
│              FastAPI  (main.py)              │
│         POST /query  →  LangGraph agent      │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│           LangGraph StateGraph               │
│                                              │
│  START → [intent_node]                       │
│                │                             │
│     ┌──────────┴──────────┐                  │
│   error                  ok                  │
│     ▼                     ▼                  │
│ [error_node]          [tool_node]            │
│     │                     │                  │
│    END        ┌───────────┴──────────┐       │
│             error                   ok       │
│               ▼                     ▼        │
│          [error_node]      [synthesis_node]  │
│               │                     │        │
│              END                   END       │
└──────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
  Gemini 2.0 Flash    REST Countries API
  (intent + synthesis)  (data retrieval)
```

### Node Responsibilities

| Node | Role | LLM? |
|------|------|------|
| `intent_node` | Extracts country name and requested data fields from the user's question | ✅ Gemini |
| `tool_node` | Calls the REST Countries API, normalises and validates the response | ❌ |
| `synthesis_node` | Composes a grounded, prose answer from the retrieved data | ✅ Gemini |
| `error_node` | Returns a user-friendly error message if any node fails | ❌ |

---

## Project Structure

```
cloudEagleAI/
├── main.py              # FastAPI application — routes & request/response schemas
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables (copy → .env)
├── .gitignore
├── README.md
└── agent/
    ├── __init__.py      # Exports AgentState & country_agent singleton
    ├── state.py         # AgentState TypedDict definition
    ├── nodes.py         # intent / tool / synthesis / error node logic
    ├── graph.py         # LangGraph StateGraph assembly & compilation
    └── tools.py         # REST Countries API wrapper (with tenacity retries)
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/shubham141923/cloudEagleAI.git
cd cloudEagleAI

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
copy .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 5. Start the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

---

## API Reference

### `POST /query`
Run a natural-language country question through the LangGraph agent.

**Request**
```json
{ "question": "What currency does Japan use?" }
```

**Response**
```json
{
  "answer": "Japan uses the Japanese Yen (JPY, symbol: ¥).",
  "country": "Japan",
  "fields_queried": ["currencies"],
  "processing_steps": [
    "intent_node: extracting country and fields from query",
    "tool_node: fetching data for 'Japan'",
    "synthesis_node: generating final answer from country data"
  ],
  "latency_ms": 1234.56
}
```

---

### `GET /health`
Health check endpoint.

```json
{ "status": "ok", "version": "1.0.0" }
```

---

### `GET /examples`
Returns a list of sample questions to test the API.

```json
{
  "examples": [
    "What is the population of Germany?",
    "What currency does Japan use?",
    "What is the capital and population of Brazil?",
    "What languages are spoken in Switzerland?",
    "Which countries border France?",
    "What timezone is Australia in?",
    "What is the area of Canada in square kilometers?",
    "Tell me everything about New Zealand."
  ]
}
```

---

### Interactive Docs

| URL | Description |
|-----|-------------|
| `GET /docs` | Swagger UI |
| `GET /redoc` | ReDoc UI |
| `GET /openapi.json` | Raw OpenAPI schema |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key |

Copy `.env.example` → `.env` and fill in your values. **Never commit `.env` to version control.**

---

## Design Decisions & Production Considerations

### Why LangGraph?
LangGraph enforces an explicit, inspectable processing pipeline. Each node has a single responsibility, conditional edges handle error routing, and the `AgentState` object provides full observability at every step — critical for debugging production issues.

### Grounding
The `synthesis_node` is strictly instructed to answer *only* from the retrieved API data. This eliminates hallucination by construction — if the data isn't there, the agent says so.

### Error Handling
- **Network retries**: `tenacity` retries transient REST Countries API errors up to 3×.
- **Conditional routing**: If `intent_node` or `tool_node` sets `error`, the graph short-circuits straight to `error_node` — no wasted LLM calls.
- **Graceful degradation**: If the LLM fails at synthesis, a rule-based fallback formats the raw data into a readable answer.
- **Invalid inputs**: Unrecognisable country names return a friendly "not found" message rather than a 500.

### Scalability
- The compiled LangGraph (`country_agent`) is a **module-level singleton** — built once at startup, reused across all requests.
- `ainvoke` is used for **fully async** execution — no threads blocked.
- The REST Countries API is stateless and public — no auth, no DB, no rate-limit issues at moderate load.
- For high scale: add an in-memory TTL cache (e.g. `cachetools`) keyed on country name, and move Gemini calls behind a rate-limit–aware queue.

---

## Known Limitations & Trade-offs

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| REST Countries API has no SLA | Occasional downtime | `tenacity` retries + graceful error messages |
| LLM latency adds ~1–2s per request | Slower than a pure lookup | Acceptable for a Q&A agent; answers can be cached |
| Intent extraction may misidentify ambiguous names (e.g. "Guinea") | Wrong country returned | API picks the best match; response includes matched country name |
| Border/neighbour codes returned as ISO alpha-3 | Not human-readable in raw form | Synthesis LLM converts them in the prose answer |
| No conversation memory | Each question is stateless | Per spec; add `MemorySaver` if multi-turn is needed |
| Gemini API key required | Limits open deployment | Easy to swap for any OpenAI-compatible provider |

---

## License
MIT
