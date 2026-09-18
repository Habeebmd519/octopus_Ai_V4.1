# Octapus AI V4.1 — Puter + Private Kerala Intelligence

V4.1 is a polished, production-oriented evolution of the V4 backend.

## Start it

From the project root:

```bash
cd v4
python3 gpt.py
```

This now works even when `gpt.py` is launched from inside the `v4/` directory.

Then open:

```text
http://localhost:5000/
```

The Flask server serves the new Octapus AI web app.

You can also run:

```bash
python3 v4/gpt.py
```

from the project root.

## What V4.1 does

- Puter.js is the browser-side LLM layer.
- The default model is `gpt-5.6-luna`.
- Users sign into Puter and their Puter account is used for AI usage under Puter's User-Pays model.
- Your private 18,000+ place dataset remains behind Flask.
- The LLM uses tool calling to retrieve only the information it needs.
- Tools include:
  - `search_places`
  - `get_place`
  - `search_services`
  - `live_search`
  - `travel_info`
  - `search_knowledge`
- The backend keeps fast local indexes/caches for place retrieval.
- The new web UI includes:
  - Puter sign-in
  - conversation history
  - nearby/location mode
  - suggested questions
  - place cards with images
  - responsive mobile layout
  - backend health indicator
  - private-record count
  - persistent browser conversation state

## Important: Groq is optional

Legacy `/api/chat` support remains, but V4.1 defaults:

```env
ENABLE_GROQ=false
```

This prevents the new Puter architecture from requiring a Groq key just to start.

If you still want the legacy Groq route, install the package and set:

```env
ENABLE_GROQ=true
GROQ_API_KEY=...
```

## Python version

Your Mac is using Python 3.9.6. It is end-of-life and your logs correctly warn about this.

Use Python 3.11 or newer if possible:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 v4/gpt.py
```

The LibreSSL warning in your log is another reason to use a modern Python installation.

## Puter model

Puter currently documents GPT-5.6 Luna as:

```js
puter.ai.chat("...", { model: "gpt-5.6-luna" })
```

and supports function calling and model discovery through `puter.ai.listModels()`.

## Architecture

```text
                    ┌───────────────────────┐
                    │  Firestore 18,000+    │
                    │  private Kerala data   │
                    └───────────┬───────────┘
                                │
User ── Web ── Puter LLM ── tool calls ── Flask V4
              │                         ├─ place search
              │                         ├─ OSM
              │                         ├─ Tavily
              │                         ├─ travel
              │                         └─ knowledge
              │
              └─ final answer
```

The browser owns the final LLM call. Flask owns private application data and tools.

## Security note

Do not put Firebase service-account JSON, Tavily keys, Google Maps keys, or any other secret in `frontend/`. Keep them in `.env` / server-side configuration.

## Production

For production, serve Flask behind HTTPS and a reverse proxy. Configure CORS to your actual domain instead of allowing arbitrary origins.

## Files

- `v4/gpt.py` — Flask backend and data/tools
- `v4/intent_detector.py` — deterministic intent routing
- `v4/live_search.py` — Tavily integration
- `v4/response_builder.py` — response compatibility layer
- `frontend/index.html` — polished web application
- `frontend/puter-agent.js` — Puter.js agent/tool loop
- `.env.example` — configuration template
- `docs/API.md` — endpoint reference
