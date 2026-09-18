# Octapus AI V5 Engine

V5 is the deterministic orchestration layer for Octapus AI. It is designed to answer many requests without calling a generative LLM.

## Core pipeline

```text
Input text / future voice transcript
        ↓
Query parser
        ↓
Mode engine
        ↓
Deterministic planner
        ↓
Domain tools
        ↓
Evidence / trust ranking
        ↓
Mode-specific response builder
        ↓
Text + optional voice response contract
```

## LLM policy

`/api/v5/engine` does not call Groq, Puter, or any generative LLM. The response is generated from deterministic rules, retrieved records, structured data and templates.

This is deliberate: simple requests should not consume model credits. A future optional "reasoning escalation" can be added for only the small percentage of requests that cannot be solved by deterministic tools.

## Modes

- `normal`: concise general assistant
- `fun`: playful challenges and return loops
- `quiz`: one-question-at-a-time quiz state and scoring
- `study`: explain → example → practice
- `travel`: logistics-first trip planning
- `explore`: discovery and local recommendations
- `local`: nearby practical help
- `research`: source-first web research with trust ranking
- `story`: interactive branching storytelling

Modes are stateful at the engine contract level. The caller sends `state` back on the next request and receives an updated `state`.

## Web research

V5 can combine Tavily and Google Programmable Search when both are configured. Tavily's answer synthesis is intentionally disabled (`include_answer=false`) so V5 does not pay for a second AI-generated answer when it only needs source material. Deep/raw extraction is reserved for research-style requests.

Google results are scored with a source-trust layer. Government/education domains receive higher trust than ordinary domains, while relevance still matters; a trusted but irrelevant result cannot automatically win.

## Voice

The engine exposes a provider-neutral voice contract:

- STT provider selection
- TTS provider selection
- locale
- interruptibility
- barge-in
- streaming capability

V5 does not force a particular frontend or speech vendor. The later UI can connect a real-time STT/TTS provider to the same engine endpoint.

## API

### `POST /api/v5/engine`

```json
{
  "message": "quiz me on Kerala history",
  "mode": "quiz",
  "history": [],
  "state": {},
  "context": {
    "location": {"lat": 10.0, "lng": 76.3, "text": "Kochi"}
  },
  "voice": true,
  "voiceLocale": "en-IN"
}
```

The response includes:

- `reply`
- `mode`
- `state`
- `plan`
- `evidence`
- `data`
- `aiUsed: false`
- `llmCalls: 0`
- `voice`
- latency metrics

### `GET /api/v5/modes`

Returns all available mode profiles and their capabilities.

### `GET /api/v5/voice/capabilities`

Returns the active STT/TTS contract and supported locales.

## Environment

```env
TAVILY_API_KEY=
GOOGLE_SEARCH_API_KEY=
GOOGLE_SEARCH_CX=
V5_WEB_TIMEOUT=8
TAVILY_SEARCH_DEPTH=basic
V5_STT_PROVIDER=browser
V5_TTS_PROVIDER=browser
V5_VOICE_LOCALE=en-IN
```

Keep Firebase service-account credentials outside the repository/package. For deployment, use `FIREBASE_SERVICE_ACCOUNT_JSON` or the platform's secret manager.
