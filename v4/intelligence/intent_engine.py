from typing import Any, Dict
from .entity_extractor import extract_entities

RULES = [
    ("emergency", ("ambulance", "accident", "fire", "snake bite", "police help", "flood emergency", "injured")),
    ("weather_current", ("weather", "rain", "mazha", "temperature", "humidity", "storm")),
    ("latest_news", ("news", "breaking", "latest")),
    ("events_today", ("event", "events", "happening", "entha events")),
    ("location_self", ("where am i", "my current location", "show my location")),
    ("nearby_hospital", ("hospital near", "hospital evide", "hospital nearby")),
    ("nearby_restaurant", ("restaurant near", "food near", "restaurant undo")),
    ("nearby_pharmacy", ("pharmacy near", "medical shop near")),
    ("route", ("distance", "directions", "how far", "route", "ethra distance")),
    ("train_search", ("train", "railway")), ("bus_search", ("bus", "ksrtc")),
    ("hotel_search", ("hotel", "stay", "resort", "hostel", "homestay")),
    ("government_service", ("aadhaar", "ration card", "certificate", "akshaya", "e-district")),
]
NEAR = ("near me", "nearby", "around here", "close to me", "nearest", "from here", "ivide aduth", "ente aduth", "walking distance")

def detect(message: str, history: Any = None) -> Dict[str, Any]:
    text = (message or "").lower()
    sub = "general_search"
    for candidate, words in RULES:
        if any(word in text for word in words): sub = candidate; break
    nearby = any(x in text for x in NEAR)
    if nearby and sub == "general_search": sub = "nearby_tourist_attraction"
    master = "local_discovery" if sub.startswith("nearby_") else sub.split("_")[0]
    location = nearby or sub in {"location_self", "location_refresh"}
    return {"master_intent": master, "sub_intent": sub, "confidence": .94 if sub != "general_search" else .55, "requires_location": location, "requires_web": sub in {"weather_current", "latest_news", "events_today"}, "requires_routing": sub == "route", "requires_current_data": sub in {"weather_current", "latest_news", "events_today"}, "language": "ml" if any("\u0d00" <= c <= "\u0d7f" for c in text) else ("manglish" if any(x in text for x in ("ente", "ivide", "nale", "undo", "evideya")) else "en"), "entities": extract_entities(message), "constraints": {}}

