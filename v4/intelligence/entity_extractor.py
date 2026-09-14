"""
Fast entity/constraint extraction for Octopus AI.

This is intentionally deterministic and forgiving of English, common
Malayalam-English transliteration and travel shorthand.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


ALIASES = {
    "cochin": "Kochi", "kochi": "Kochi", "ernakulam": "Kochi",
    "trivandrum": "Thiruvananthapuram", "tvm": "Thiruvananthapuram",
    "calicut": "Kozhikode", "alleppey": "Alappuzha", "trichur": "Thrissur",
    "quilon": "Kollam", "palghat": "Palakkad", "cannanore": "Kannur",
    "kasargod": "Kasaragod", "waynad": "Wayanad", "munar": "Munnar",
    "wagamon": "Vagamon", "thekady": "Thekkady", "athirapilly": "Athirappilly",
}

DISTRICTS = [
    "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha",
    "Kottayam", "Idukki", "Ernakulam", "Thrissur", "Palakkad",
    "Malappuram", "Kozhikode", "Wayanad", "Kannur", "Kasaragod",
]

KNOWN_PLACES = [
    "Munnar", "Kochi", "Wayanad", "Varkala", "Thekkady", "Vagamon",
    "Bekal", "Athirappilly", "Guruvayur", "Alappuzha", "Kumarakom",
    "Kovalam", "Ponmudi", "Fort Kochi", "Kappad", "Kozhikode",
    "Kannur", "Kollam", "Kottayam", "Thekkady", "Gavi", "Vagamon",
]

TRANSPORT = ("train", "bus", "flight", "car", "bike", "scooter", "walking",
             "ferry", "metro", "taxi", "auto", "cab")
DIETARY = ("vegetarian", "veg", "vegan", "halal", "seafood", "non veg", "non-veg")
PREFERENCES = (
    "family", "couple", "solo", "quiet", "budget", "luxury", "adventure",
    "relaxation", "romantic", "honeymoon", "pet friendly", "wheelchair",
    "accessible", "kids", "elder friendly", "photography", "sunrise", "sunset",
)
MEALS = ("breakfast", "lunch", "dinner", "brunch")
ACTIVITIES = (
    "trekking", "trek", "hiking", "camping", "kayaking", "boating",
    "paragliding", "zipline", "wildlife", "photography", "shopping",
)


def _norm(text: str) -> str:
    text = (text or "").lower()
    text = text.replace(",", " ").replace("₹", " rs ")
    return re.sub(r"\s+", " ", text).strip()


def _unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(x for x in items if x))


def _number(text: str, patterns: List[str]) -> Optional[float]:
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            try:
                value = float(m.group(1))
                multiplier = m.groupdict().get("multiplier")
                if multiplier:
                    value *= 1000
                return value
            except (TypeError, ValueError):
                pass
    return None


def extract_entities(message: str) -> Dict[str, Any]:
    original = message or ""
    lower = _norm(original)

    places: List[str] = []
    for alias, canonical in ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", lower):
            places.append(canonical)

    for name in KNOWN_PLACES + DISTRICTS:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", lower):
            places.append(name)

    places = _unique(places)

    districts = [x for x in DISTRICTS if x.lower() in {p.lower() for p in places}]
    cities = [x for x in places if x not in districts]

    budget = _number(
        lower,
        [
            r"(?:under|below|within|max(?:imum)?|budget(?:\s+of)?|less than)\s+(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s*(?P<multiplier>k)?",
            r"(?:rs\.?\s*|₹\s*)(\d+(?:\.\d+)?)\s*(?P<multiplier>k)?",
        ],
    )

    days = None
    m = re.search(r"\b(\d{1,2})\s*(?:day|days|night|nights)\b", lower)
    if m:
        days = int(m.group(1))

    people = None
    m = re.search(r"\b(?:for\s+)?(\d{1,3})\s*(?:people|persons|adults|pax)\b", lower)
    if m:
        people = int(m.group(1))
    elif re.search(r"\b(?:for\s+)?(\d{1,3})\s*(?:of us|of me)\b", lower):
        people = int(re.search(r"\b(?:for\s+)?(\d{1,3})\s*(?:of us|of me)\b", lower).group(1))

    children = None
    m = re.search(r"\b(\d{1,2})\s*(?:kids?|children)\b", lower)
    if m:
        children = int(m.group(1))

    distance = _number(lower, [r"(?:within|under|less than)\s+(\d+(?:\.\d+)?)\s*km\b"])
    rating = _number(lower, [r"(?:rating|rated)\s*(?:of\s*)?(\d(?:\.\d+)?)\s*(?:star|stars)?"])

    transport = next((x for x in TRANSPORT if x in lower), None)
    dietary = [x for x in DIETARY if x in lower]
    preferences = [x for x in PREFERENCES if x in lower]
    activities = [x for x in ACTIVITIES if x in lower]
    meals = [x for x in MEALS if x in lower]

    dates = []
    for phrase in (
        "today", "tomorrow", "tonight", "this weekend", "next weekend",
        "next week", "this month", "next month", "onam", "vishu",
    ):
        if phrase in lower:
            dates.append(phrase)

    times = []
    for pattern in (r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b", r"\b(\d{1,2}\s*(?:am|pm))\b"):
        times.extend(m.group(1) for m in re.finditer(pattern, lower, re.I))

    origin = None
    destination = None
    m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+(?:today|tomorrow|now|by|via)\b|$)", lower)
    if m:
        origin, destination = m.group(1).strip(), m.group(2).strip()

    constraints = []
    if budget is not None:
        constraints.append({"type": "budget_max", "value": budget, "currency": "INR"})
    if distance is not None:
        constraints.append({"type": "distance_max_km", "value": distance})
    if rating is not None:
        constraints.append({"type": "rating_min", "value": rating})
    if people is not None:
        constraints.append({"type": "people", "value": people})
    if children is not None:
        constraints.append({"type": "children", "value": children})
    for p in preferences:
        constraints.append({"type": "preference", "value": p})
    for a in activities:
        constraints.append({"type": "activity", "value": a})

    return {
        "places": places,
        "districts": districts,
        "cities": cities,
        "origin": origin,
        "destination": destination,
        "dates": _unique(dates),
        "times": _unique(times),
        "duration_days": max(1, min(days, 30)) if days else None,
        "people": max(1, min(100, people)) if people else None,
        "children": max(0, min(30, children)) if children is not None else None,
        "budget": budget,
        "currency": "INR",
        "distance_km": distance,
        "rating_min": rating,
        "transport": transport,
        "preferences": _unique(preferences),
        "dietary": _unique(dietary),
        "activities": _unique(activities),
        "meals": _unique(meals),
        "constraints": constraints,
    }
