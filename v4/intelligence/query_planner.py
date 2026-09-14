from typing import Any, Dict, List
from .freshness import classify_freshness

def plan(message: str, intent: Dict[str, Any], location: Dict[str, Any]) -> Dict[str, Any]:
    sub, steps = intent["sub_intent"], []
    if intent["requires_location"]: steps.append({"tool": "location", "reason": "nearby or current-location request"})
    if sub.startswith("weather"): steps.append({"tool": "get_weather", "reason": "dedicated weather provider"})
    elif sub.endswith("news") or sub == "latest_news": steps.append({"tool": "search_news", "reason": "fresh multi-source news"})
    elif sub.startswith("events"): steps.append({"tool": "search_events", "reason": "current event discovery"})
    elif sub.startswith("nearby_"): steps.append({"tool": "search_nearby", "reason": "local service discovery"})
    elif sub == "route": steps.append({"tool": "get_route", "reason": "verified routing"})
    else: steps.append({"tool": "search_places", "reason": "private Kerala data"})
    freshness = classify_freshness(message, sub)
    if freshness["requires_web"] and steps[-1]["tool"] not in {"get_weather", "search_news", "search_events"}: steps.append({"tool": "live_search", "reason": "current verification"})
    return {"steps": steps[:4], "freshness": freshness}

