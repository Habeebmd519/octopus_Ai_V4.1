import re
from typing import Any, Dict, List

MODE_WORDS = {"fun", "quiz", "study", "travel", "explore", "local", "research", "story", "normal"}

LIVE_WORDS = {
    "latest", "current", "today", "now", "tonight", "tomorrow", "this week",
    "open now", "opening", "closing", "price", "fee", "weather", "temperature",
    "news", "available", "availability", "event", "events", "schedule", "status",
}

QUESTION_WORDS = {"what", "where", "when", "why", "who", "which", "how", "can", "is", "are", "do", "does"}


def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def parse_query(message: str, history: List[Dict[str, Any]] | None = None, location: Dict[str, Any] | None = None) -> Dict[str, Any]:
    q = (message or "").strip()
    t = tokens(q)
    live = any(x in q.lower() for x in LIVE_WORDS)
    near = any(x in q.lower() for x in ("near me", "nearby", "around me", "close to me", "nearest"))
    research = any(x in q.lower() for x in ("research", "compare sources", "evidence", "according to", "verify"))
    question = bool(t and (t[0] in QUESTION_WORDS or "?" in q))
    budget = None
    m = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+)|\b([\d,]+)\s*(?:rupees|rs)\b", q, re.I)
    if m:
        budget = int((m.group(1) or m.group(2)).replace(",", ""))
    people = None
    m = re.search(r"\b(\d+)\s*(?:people|persons|person|adults|of us)\b", q, re.I)
    if m:
        people = int(m.group(1))
    days = None
    m = re.search(r"\b(\d+)\s*(?:day|days|night|nights)\b", q, re.I)
    if m:
        days = int(m.group(1))
    return {
        "raw": q,
        "tokens": t,
        "live": live,
        "near": near,
        "research": research,
        "question": question,
        "budget": budget,
        "people": people,
        "days": days,
        "has_location": bool((location or {}).get("lat") and (location or {}).get("lng")),
        "length": len(t),
    }
