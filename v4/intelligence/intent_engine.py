"""
Octopus AI V4.2 Intent Engine.

A deterministic routing layer that runs before the LLM. It does not try to
replace the LLM; it gives the LLM a high-quality map of the user's intent,
entities, freshness requirements and the first tools worth trying.

The engine is deliberately dependency-free so it is fast on Render's free tier.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from .entity_extractor import extract_entities
from .freshness import classify_freshness


NEAR_WORDS = (
    "near me", "nearby", "around me", "around here", "close to me",
    "nearest", "from here", "walking distance", "ivide aduth",
    "ente aduth", "evide aduth", "aduth undo", "അടുത്ത്", "എന്റെ അടുത്ത്",
)

INTENT_RULES: Dict[str, Dict[str, Any]] = {
    "emergency": {
        "phrases": ("emergency", "urgent", "accident", "rescue", "help me now",
                    "police help", "ambulance", "fire force", "snake bite",
                    "flood help", "112", "108", "101"),
        "tools": ("search_services", "search_knowledge", "live_search"),
        "freshness": "current",
        "priority": 100,
    },
    "location_self": {
        "phrases": ("where am i", "my location", "current location", "show my location",
                    "ente location", "njān evide", "njan evide"),
        "tools": (),
        "freshness": "real_time",
        "priority": 100,
    },
    "weather_current": {
        "phrases": ("weather", "weather today", "rain", "mazha", "temperature",
                    "humidity", "forecast", "storm", "wind", "will it rain"),
        "tools": ("get_weather",),
        "freshness": "real_time",
        "priority": 95,
    },
    "latest_news": {
        "phrases": ("latest news", "breaking news", "today news", "news today",
                    "latest", "what happened today"),
        "tools": ("search_news",),
        "freshness": "real_time",
        "priority": 92,
    },
    "events_today": {
        "phrases": ("events today", "events this weekend", "what is happening",
                    "happening today", "program today", "event", "events"),
        "tools": ("search_events",),
        "freshness": "current",
        "priority": 88,
    },
    "train_status": {
        "phrases": ("train status", "train running", "live train", "where is my train",
                    "train delay", "train running status"),
        "tools": ("live_search",),
        "freshness": "real_time",
        "priority": 96,
    },
    "train_search": {
        "phrases": ("train", "railway", "railway station", "ernakulam to",
                    "trivandrum to", "kochi to"),
        "tools": ("search_knowledge", "live_search"),
        "freshness": "current",
        "priority": 72,
    },
    "bus_search": {
        "phrases": ("bus", "ksrtc", "bus timing", "bus time", "bus route",
                    "bus stand", "bus station"),
        "tools": ("search_services", "live_search"),
        "freshness": "current",
        "priority": 75,
    },
    "flight_search": {
        "phrases": ("flight", "airport", "flight status", "departure", "arrival"),
        "tools": ("search_services", "live_search"),
        "freshness": "current",
        "priority": 76,
    },
    "route": {
        "phrases": ("distance", "how far", "how long", "route", "directions",
                    "how to go", "how can i reach", "ethra distance", "ethra dooram"),
        "tools": ("travel_info",),
        "freshness": "current",
        "priority": 84,
    },
    "traffic": {
        "phrases": ("traffic", "traffic now", "road block", "road closed",
                    "congestion", "jam", "block undo"),
        "tools": ("live_search", "travel_info"),
        "freshness": "real_time",
        "priority": 90,
    },
    "parking": {
        "phrases": ("parking", "car parking", "bike parking", "parking area"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 70,
    },
    "nearby_hospital": {
        "phrases": ("hospital near", "hospital nearby", "nearest hospital",
                    "clinic near", "doctor near", "medical near", "hospital evide"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 91,
    },
    "nearby_pharmacy": {
        "phrases": ("pharmacy near", "medical shop", "medical store", "medicine near",
                    "pharmacy nearby"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 89,
    },
    "nearby_restaurant": {
        "phrases": ("restaurant near", "restaurant nearby", "food near",
                    "cafe near", "where to eat", "restaurant undo", "food undo"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 83,
    },
    "nearby_hotel": {
        "phrases": ("hotel near", "stay near", "resort near", "homestay near",
                    "accommodation near", "hotel nearby"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 82,
    },
    "nearby_atm": {
        "phrases": ("atm near", "atm nearby", "nearest atm", "cash near",
                    "bank near", "bank nearby"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 80,
    },
    "nearby_fuel": {
        "phrases": ("petrol pump", "fuel station", "petrol near", "diesel near",
                    "ev charging", "charging station"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 80,
    },
    "nearby_emergency_service": {
        "phrases": ("police station near", "fire station near", "police nearby",
                    "fire force near"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 94,
    },
    "food": {
        "phrases": ("food", "eat", "breakfast", "lunch", "dinner", "sadya",
                    "biriyani", "seafood", "local food", "dish"),
        "tools": ("search_places", "search_services"),
        "freshness": "static",
        "priority": 60,
    },
    "stay": {
        "phrases": ("hotel", "resort", "homestay", "hostel", "stay", "room",
                    "accommodation", "where to stay"),
        "tools": ("search_places", "search_services"),
        "freshness": "semi_dynamic",
        "priority": 62,
    },
    "attraction": {
        "phrases": ("tourist place", "tourist attraction", "attraction", "places to visit",
                    "what to see", "sightseeing", "hidden gem", "must visit"),
        "tools": ("search_places",),
        "freshness": "static",
        "priority": 68,
    },
    "beach": {
        "phrases": ("beach", "sea", "coast", "shore", "lighthouse", "sunset beach"),
        "tools": ("search_places",),
        "freshness": "static",
        "priority": 66,
    },
    "waterfall": {
        "phrases": ("waterfall", "waterfalls", "falls", "water fall"),
        "tools": ("search_places",),
        "freshness": "static",
        "priority": 68,
    },
    "hill_station": {
        "phrases": ("hill station", "mountain", "hills", "tea estate", "misty",
                    "cool place", "high range"),
        "tools": ("search_places",),
        "freshness": "static",
        "priority": 65,
    },
    "wildlife": {
        "phrases": ("wildlife", "sanctuary", "national park", "elephant",
                    "tiger", "bird watching", "forest"),
        "tools": ("search_places",),
        "freshness": "static",
        "priority": 66,
    },
    "adventure": {
        "phrases": ("trek", "trekking", "hiking", "camping", "adventure",
                    "offroad", "kayaking", "paragliding", "zipline"),
        "tools": ("search_places", "live_search"),
        "freshness": "semi_dynamic",
        "priority": 66,
    },
    "culture": {
        "phrases": ("culture", "theyyam", "kathakali", "mohiniyattam", "kalaripayattu",
                    "art form", "chenda", "oppana"),
        "tools": ("search_knowledge",),
        "freshness": "static",
        "priority": 65,
    },
    "festival": {
        "phrases": ("festival", "onam", "vishu", "pooram", "thrissur pooram",
                    "pongala", "boat race", "perunnal"),
        "tools": ("search_knowledge", "search_events"),
        "freshness": "current",
        "priority": 78,
    },
    "history": {
        "phrases": ("history", "historic", "travancore", "zamorin", "samoothiri",
                    "pazhassi", "malabar rebellion", "mappila rebellion"),
        "tools": ("search_knowledge",),
        "freshness": "static",
        "priority": 64,
    },
    "writer": {
        "phrases": ("writer", "author", "poet", "novelist", "malayalam writer"),
        "tools": ("search_knowledge",),
        "freshness": "static",
        "priority": 64,
    },
    "book": {
        "phrases": ("book", "novel", "poem", "literature", "short story"),
        "tools": ("search_knowledge",),
        "freshness": "static",
        "priority": 63,
    },
    "government_service": {
        "phrases": ("aadhaar", "pan card", "ration card", "birth certificate",
                    "income certificate", "village office", "akshaya", "edistrict",
                    "property tax", "land tax", "driving licence", "pension"),
        "tools": ("search_knowledge", "live_search"),
        "freshness": "current",
        "priority": 86,
    },
    "education": {
        "phrases": ("course", "college", "psc", "iti", "polytechnic", "scholarship",
                    "keam", "neet", "higher studies", "ktu", "career", "internship"),
        "tools": ("search_knowledge", "live_search"),
        "freshness": "current",
        "priority": 70,
    },
    "trip_plan": {
        "phrases": ("plan a trip", "trip plan", "itinerary", "travel plan",
                    "2 day", "3 day", "one day", "weekend", "schedule"),
        "tools": ("search_places", "travel_info"),
        "freshness": "semi_dynamic",
        "priority": 86,
    },
    "recommendation": {
        "phrases": ("best", "top", "recommend", "recommendation", "suggest",
                    "popular", "trending", "highest rated", "where should i go"),
        "tools": ("search_places",),
        "freshness": "semi_dynamic",
        "priority": 58,
    },
    "best_time": {
        "phrases": ("best time", "when to visit", "which month", "season",
                    "monsoon", "summer", "winter"),
        "tools": ("search_places", "live_search"),
        "freshness": "semi_dynamic",
        "priority": 57,
    },
    "compare": {
        "phrases": ("compare", "versus", "vs", "which is better", "or which",
                    "better than"),
        "tools": ("search_places", "get_place"),
        "freshness": "static",
        "priority": 75,
    },
    "image": {
        "phrases": ("image", "photo", "picture", "pic", "show me a photo"),
        "tools": ("search_places",),
        "freshness": "static",
        "priority": 55,
    },
    "budget_plan": {
        "phrases": ("budget", "cheap", "low cost", "affordable", "under ₹",
                    "under rs", "how much will i spend", "cost for trip"),
        "tools": ("search_places", "travel_info"),
        "freshness": "semi_dynamic",
        "priority": 74,
    },
    "family_trip": {
        "phrases": ("family trip", "with kids", "with children", "parents",
                    "elderly", "baby friendly", "family friendly"),
        "tools": ("search_places",),
        "freshness": "semi_dynamic",
        "priority": 72,
    },
    "honeymoon": {
        "phrases": ("honeymoon", "couple trip", "romantic", "couples", "anniversary"),
        "tools": ("search_places", "search_services"),
        "freshness": "semi_dynamic",
        "priority": 72,
    },
    "sunrise_sunset": {
        "phrases": ("sunrise", "sunset", "sun rise", "sun set", "sunset point"),
        "tools": ("search_places", "live_search"),
        "freshness": "current",
        "priority": 71,
    },
    "shopping": {
        "phrases": ("shopping", "market", "mall", "souvenir", "handicraft"),
        "tools": ("search_places", "search_services"),
        "freshness": "semi_dynamic",
        "priority": 60,
    },
    "accessibility": {
        "phrases": ("wheelchair", "accessible", "disabled friendly", "elder friendly",
                    "step free", "accessibility"),
        "tools": ("search_places", "search_services"),
        "freshness": "semi_dynamic",
        "priority": 69,
    },
    "pet_friendly": {
        "phrases": ("pet friendly", "dog friendly", "cat friendly", "with my dog",
                    "with pet"),
        "tools": ("search_places", "search_services"),
        "freshness": "semi_dynamic",
        "priority": 68,
    },
    "nearby_shopping": {
        "phrases": ("shopping near", "mall near", "supermarket near", "market near", "shopping nearby"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 73,
    },
    "nearby_worship": {
        "phrases": ("temple near", "church near", "mosque near", "worship near", "temple nearby"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 70,
    },
    "nearby_education": {
        "phrases": ("school near", "college near", "university near", "library near"),
        "tools": ("search_nearby",),
        "freshness": "current",
        "priority": 69,
    },
    "general_kerala": {
        "phrases": (),
        "tools": ("search_places", "search_knowledge"),
        "freshness": "static",
        "priority": 1,
    },
}

CATEGORY_HINTS = {
    "nearby_hospital": "osm_health",
    "nearby_pharmacy": "osm_health",
    "nearby_restaurant": "food",
    "nearby_hotel": "osm_stay",
    "nearby_atm": "osm_money",
    "nearby_fuel": "osm_fuel",
    "nearby_emergency_service": "osm_emergency",
    "parking": "osm_transport",
    "bus_search": "osm_transport",
    "flight_search": "osm_transport",
    "shopping": "osm_shopping",
    "culture": "osm_worship",
    "nearby_shopping": "osm_shopping",
    "nearby_worship": "osm_worship",
    "nearby_education": "osm_education",
}


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _language(text: str) -> str:
    if re.search(r"[\u0D00-\u0D7F]", text or ""):
        return "ml"
    manglish = ("ente", "njan", "njān", "evide", "ivide", "undo", "venam", "venam",
                "ethra", "nalla", "poyi", "pokam", "pokan", "aduth", "nale", "inn")
    return "manglish" if any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in manglish) else "en"


def _phrase_score(text: str, phrase: str) -> float:
    p = _normalize(phrase)
    if not p:
        return 0.0
    if p in text:
        # Exact phrase matches are much stronger than single-token matches.
        return min(1.0, 0.45 + 0.10 * len(p.split()))
    tokens = [t for t in re.findall(r"[a-z0-9]+", p) if len(t) > 2]
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if re.search(r"\b" + re.escape(token) + r"\b", text))
    return min(0.4, hits / len(tokens) * 0.4)


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(_phrase_score(text, p) >= 0.45 for p in phrases)


def _compound_signals(text: str) -> List[str]:
    signals = []
    if _contains_any(text, NEAR_WORDS):
        signals.append("nearby")
    if any(x in text for x in ("today", "tomorrow", "now", "latest", "current", "live")):
        signals.append("time_sensitive")
    if any(x in text for x in ("open now", "opening", "closing", "available now")):
        signals.append("open_now")
    if any(x in text for x in ("cheap", "budget", "under ", "low cost", "affordable")):
        signals.append("budget")
    if any(x in text for x in ("family", "kids", "children", "parents", "elder")):
        signals.append("family")
    if any(x in text for x in ("couple", "honeymoon", "romantic")):
        signals.append("romantic")
    if any(x in text for x in ("with dog", "with pet", "pet friendly")):
        signals.append("pet")
    return signals


def _score_intents(text: str) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for intent, spec in INTENT_RULES.items():
        if intent == "general_kerala":
            continue
        best = 0.0
        for phrase in spec["phrases"]:
            best = max(best, _phrase_score(text, phrase))
        if best <= 0:
            continue
        score = best * float(spec.get("priority", 50))
        scored.append((intent, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def detect(message: str, history: Any = None) -> Dict[str, Any]:
    text = _normalize(message)
    scored = _score_intents(text)
    top_intent, top_score = scored[0] if scored else ("general_kerala", 1.0)
    signals = _compound_signals(text)

    # High-precision overrides for phrases that are commonly ambiguous.
    if (
        any(x in text for x in ("distance", "how far", "route", "directions", "how to go"))
        and (" from " in f" {text} " or " to " in f" {text} ")
    ):
        top_intent = "route"
        top_score = 90.0

    writer_names = (
        "vaikom muhammad basheer", "vaikom basheer", "basheer",
        "m t vasudevan nair", "mt vasudevan nair", "o v vijayan",
        "ov vijayan", "sugathakumari", "kumaran asan", "vallathol",
        "madhavikutty", "kamala das", "thakazhi", "s k pottekkatt",
    )
    if any(name in text for name in writer_names):
        top_intent = "writer"
        top_score = 92.0

    # Nearby is a modifier. Prefer the practical service intent when present.
    nearby = "nearby" in signals or _contains_any(text, NEAR_WORDS)
    if nearby:
        nearby_overrides = (
            ("hospital", "nearby_hospital"),
            ("clinic", "nearby_hospital"),
            ("pharmacy", "nearby_pharmacy"),
            ("medical", "nearby_pharmacy"),
            ("restaurant", "nearby_restaurant"),
            ("cafe", "nearby_restaurant"),
            ("hotel", "nearby_hotel"),
            ("resort", "nearby_hotel"),
            ("homestay", "nearby_hotel"),
            ("atm", "nearby_atm"),
            ("bank", "nearby_atm"),
            ("petrol", "nearby_fuel"),
            ("fuel", "nearby_fuel"),
            ("diesel", "nearby_fuel"),
            ("charging", "nearby_fuel"),
            ("police", "nearby_emergency_service"),
            ("fire station", "nearby_emergency_service"),
            ("parking", "parking"),
            ("shopping", "nearby_shopping"),
            ("mall", "nearby_shopping"),
            ("supermarket", "nearby_shopping"),
            ("temple", "nearby_worship"),
            ("church", "nearby_worship"),
            ("mosque", "nearby_worship"),
            ("school", "nearby_education"),
            ("college", "nearby_education"),
            ("university", "nearby_education"),
        )
        for keyword, service_intent in nearby_overrides:
            if keyword in text:
                top_intent = service_intent
                top_score = float(INTENT_RULES.get(service_intent, {}).get("priority", 80))
                break
    if nearby and top_intent == "general_kerala":
        top_intent = "attraction"

    entities = extract_entities(message)
    freshness = classify_freshness(message, top_intent)

    # Current/open-now questions upgrade otherwise static intents.
    if "time_sensitive" in signals or "open_now" in signals:
        freshness = {
            **freshness,
            "freshness": "real_time" if "now" in text or "live" in text else "current",
            "requires_web": True,
            "max_age_minutes": 5 if "now" in text or "live" in text else 60,
        }

    spec = INTENT_RULES.get(top_intent, INTENT_RULES["general_kerala"])
    tools = list(spec.get("tools", ()))

    if nearby and "search_nearby" not in tools:
        tools.insert(0, "search_nearby")
    if "time_sensitive" in signals and "live_search" not in tools and top_intent not in {"weather_current", "nearby_hospital", "nearby_pharmacy"}:
        tools.append("live_search")

    # Avoid duplicate tools while preserving order.
    tools = list(dict.fromkeys(tools))[:5]

    confidence = min(0.99, 0.45 + top_score / 140.0)
    if top_intent == "general_kerala":
        confidence = 0.52

    master = top_intent.split("_")[0]
    if top_intent.startswith("nearby_"):
        master = "local_discovery"
    if top_intent in {"writer", "book", "history", "culture", "festival",
                      "government_service", "education", "food"}:
        master = top_intent

    return {
        "master_intent": master,
        "sub_intent": top_intent,
        "confidence": round(confidence, 3),
        "requires_location": nearby or top_intent == "location_self",
        "requires_web": bool(freshness.get("requires_web")) or bool("live_search" in tools),
        "requires_routing": "travel_info" in tools,
        "requires_current_data": freshness.get("freshness") in {"current", "real_time"},
        "language": _language(text),
        "entities": entities,
        "signals": signals,
        "candidate_intents": [
            {"intent": name, "score": round(score, 2)}
            for name, score in scored[:5]
        ],
        "recommended_tools": tools,
        "category_hint": CATEGORY_HINTS.get(top_intent),
        "freshness": freshness,
        "constraints": entities.get("constraints", []),
        "trigger_reasons": {
            "intent": top_intent,
            "matchedSignals": signals,
            "category": CATEGORY_HINTS.get(top_intent),
            "confidence": round(confidence, 3),
        },
    }


def intent_catalog() -> Dict[str, Any]:
    return {
        name: {
            "tools": list(spec.get("tools", ())),
            "freshness": spec.get("freshness"),
            "priority": spec.get("priority"),
            "phrases": list(spec.get("phrases", ())),
        }
        for name, spec in INTENT_RULES.items()
    }
