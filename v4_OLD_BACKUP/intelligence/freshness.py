from typing import Dict

def classify_freshness(message: str, intent: str = "") -> Dict[str, object]:
    text = (message or "").lower()
    realtime = ("live location", "traffic", "train status", "flight status")
    current = ("today", "tomorrow", "now", "latest", "news", "weather", "rain", "open", "hours", "price", "event", "schedule", "fare")
    semi = ("recommend", "plan", "itinerary", "best time", "trip")
    if any(x in text for x in realtime): return {"freshness": "real_time", "requires_web": True, "max_age_minutes": 5}
    if any(x in text for x in current) or intent.startswith(("weather", "latest", "events", "ticket", "opening")): return {"freshness": "current", "requires_web": True, "max_age_minutes": 60}
    if any(x in text for x in semi): return {"freshness": "semi_dynamic", "requires_web": False, "max_age_minutes": 1440}
    return {"freshness": "static", "requires_web": False, "max_age_minutes": 10080}

