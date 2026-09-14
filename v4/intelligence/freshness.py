"""Freshness policy for Octopus AI answers."""
from typing import Dict


REAL_TIME = (
    "live", "right now", "now", "traffic now", "train status",
    "flight status", "open now", "available now", "currently",
)
CURRENT = (
    "today", "tomorrow", "latest", "news", "weather", "rain", "open",
    "opening", "closing", "hours", "price", "ticket", "fare", "event",
    "events", "schedule", "status", "timing", "this weekend",
)
SEMI_DYNAMIC = (
    "recommend", "recommendation", "best", "plan", "itinerary",
    "trip", "budget", "hotel", "restaurant", "stay", "route",
)


def classify_freshness(message: str, intent: str = "") -> Dict[str, object]:
    text = (message or "").lower()

    if intent in {"weather_current", "train_status", "traffic"} or any(x in text for x in REAL_TIME):
        return {"freshness": "real_time", "requires_web": True, "max_age_minutes": 5}

    if intent in {
        "latest_news", "events_today", "government_service", "education",
        "festival", "flight_search", "bus_search", "nearby_hospital",
        "nearby_pharmacy", "nearby_restaurant", "nearby_hotel", "nearby_atm",
        "nearby_fuel", "nearby_emergency_service",
    } or any(x in text for x in CURRENT):
        return {"freshness": "current", "requires_web": True, "max_age_minutes": 60}

    if intent in {
        "recommendation", "trip_plan", "budget_plan", "best_time",
        "food", "stay", "family_trip", "honeymoon", "shopping",
        "accessibility", "pet_friendly", "adventure",
    } or any(x in text for x in SEMI_DYNAMIC):
        return {"freshness": "semi_dynamic", "requires_web": False, "max_age_minutes": 1440}

    return {"freshness": "static", "requires_web": False, "max_age_minutes": 10080}
