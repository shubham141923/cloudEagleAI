# GlobeIQ – Country Information AI Agent

> **LangGraph-powered AI agent** that answers natural-language questions about countries using the public [REST Countries API](https://restcountries.com).

---

## Live Demo
🔗 **[→ Try it here](https://your-deployment-url.com)**  
📹 **[Video Walkthrough](https://your-video-url.com)**

---

## Architecture Overview

```
User Question
     │
     ▼
┌──────────────────────────────────────────────┐
│              FastAPI  (main.py)              │
│   POST /api/query  →  runs LangGraph agent   │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│           LangGraph StateGraph               │
│                                              │
│  START → [intent_node] ──────────────┐       │
│                │                     │error  │
│                ▼                     ▼       │
│          [tool_node]          [error_node]   │
│                │                     │       │
│                ▼              (answer=error) │
│      [synthesis_node] ───────────────┘       │
│                │                             │
│               END                            │
└──────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
  Gemini 2.0 Flash    REST Countries API
  (intent + synthesis)  (data retrieval)
```

### Node Responsibilities

| Node | Role | LLM? |
|------|------|------|
| `intent_node` | Extracts the country name and requested data fields from the user's question | ✅ Gemini |
| `tool_node` | Calls the REST Countries API, normalises and validates the response | ❌ |
| `synthesis_node` | Composes a grounded, prose answer from the retrieved data | ✅ Gemini |
| `error_node` | Ensures a user-friendly message is always returned if any node fails | ❌ |

---

## Getting Started

### Prerequisites
- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/country-agent.git
cd country-agent/backend

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
copy .env.example .env
# Edit .env and set GEMINI_API_KEY=...

# 5. Start the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

---

## API Reference

### `POST /api/query`
Run a country question through the agent.

**Request**
```json
{ "question": "What currency does Japan use?" }
```

**Response**
```json
{
  "answer": "Japan uses the Japanese Yen (¥).",
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

### `GET /api/health`
Health check. Returns `{"status": "ok"}`.

### `GET /api/examples`
Returns a list of sample questions to seed the UI.

### Interactive docs: `GET /api/docs`

---

## Project Structure

```
country-agent/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt
│   ├── .env.example
│   └── agent/
│       ├── __init__.py
│       ├── state.py         # AgentState TypedDict
│       ├── tools.py         # REST Countries API wrapper
│       ├── nodes.py         # intent / tool / synthesis / error nodes
│       └── graph.py         # LangGraph StateGraph definition
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## Design Decisions & Production Considerations

### Why LangGraph?
LangGraph enforces an explicit, inspectable processing pipeline rather than a single monolithic prompt. Each node has a single responsibility, conditional edges handle error routing, and the state object provides full observability at every step — critical for debugging production issues.

### Grounding
The synthesis node is strictly instructed to answer *only* from the retrieved data. This eliminates hallucination by construction.

### Error Handling
- **Network retries**: `tenacity` retries transient transport errors up to 3×.
- **Graceful degradation**: If the LLM fails at synthesis, a rule-based fallback formats the raw data into a readable answer.
- **Conditional routing**: If intent or tool nodes set `error`, the graph skips straight to `error_node` without calling the LLM again.
- **Invalid inputs**: Unrecognisable country names return a friendly "not found" message rather than an exception.

### Scalability
- The compiled LangGraph (`country_agent`) is a module-level singleton — built once at startup.
- `ainvoke` is used for fully async execution in FastAPI.
- The REST Countries API is stateless and public — no auth, no DB, no rate limit issues at moderate load.
- For high scale: add an in-memory TTL cache (e.g. `cachetools`) per country name, and move the Gemini calls behind a rate-limit–aware queue.

---

## Known Limitations & Trade-offs

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| REST Countries API has no SLA | Occasional downtime | `tenacity` retries + graceful error messages |
| LLM latency adds ~1-2s per request | Slower than a pure lookup | Acceptable for a Q&A agent; could cache answers |
| Intent extraction may misidentify ambiguous country names (e.g. "Guinea") | Wrong country returned | API picks best match; UI shows the matched country name |
| Border/neighbour codes are returned as ISO alpha-3 codes | Not human-readable in raw form | Synthesis LLM converts them in the answer |
| No conversation memory | Each question is stateless | Acceptable per spec; add `MemorySaver` if multi-turn needed |
| Gemini API key required | Limits open deployment | `GEMINI_API_KEY` env var; easy to swap for any OpenAI-compatible provider |

---

## License
MIT
