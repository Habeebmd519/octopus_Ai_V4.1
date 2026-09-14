import re
from typing import Any, Dict, List

ALIASES = {
    "cochin": "Kochi", "kochi": "Kochi", "trivandrum": "Thiruvananthapuram",
    "calicut": "Kozhikode", "alleppey": "Alappuzha", "trichur": "Thrissur",
    "quilon": "Kollam", "palghat": "Palakkad", "cannanore": "Kannur",
}
DISTRICTS = ["Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha", "Kottayam", "Idukki", "Ernakulam", "Thrissur", "Palakkad", "Malappuram", "Kozhikode", "Wayanad", "Kannur", "Kasaragod"]
KNOWN_PLACES = ["Munnar", "Kochi", "Wayanad", "Varkala", "Thekkady", "Vagamon", "Bekal", "Athirappilly", "Guruvayur", "Alappuzha"]

def extract_entities(message: str) -> Dict[str, Any]:
    text = message or ""
    lower = text.lower()
    places: List[str] = []
    for alias, canonical in ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", lower): places.append(canonical)
    for name in KNOWN_PLACES + DISTRICTS:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", lower): places.append(name)
    budget = re.search(r"(?:₹|rs\.?|rupees?\s*)?(\d+(?:\.\d+)?)(k)?\b", lower)
    amount = float(budget.group(1)) * (1000 if budget and budget.group(2) else 1) if budget else None
    days = re.search(r"\b(\d+)\s*(?:day|days|nights?)\b", lower)
    people = re.search(r"\b(?:for\s+)?(\d+)\s*(?:people|persons|adults?)\b", lower)
    children = re.search(r"\b(\d+)\s*(?:kids?|children)\b", lower)
    distance = re.search(r"(?:within|under|less than)\s*(\d+(?:\.\d+)?)\s*km\b", lower)
    transport = next((x for x in ("train", "bus", "flight", "car", "bike", "walking", "ferry", "metro") if x in lower), None)
    dietary = [x for x in ("vegetarian", "vegan", "halal", "seafood") if x in lower]
    preferences = [x for x in ("family", "couple", "solo", "quiet", "budget", "luxury", "adventure", "relaxation") if x in lower]
    return {"places": list(dict.fromkeys(places)), "districts": [x for x in DISTRICTS if x in places], "cities": [x for x in places if x not in DISTRICTS], "dates": [x for x in ("today", "tomorrow", "this weekend", "next saturday") if x in lower], "times": [], "duration_days": int(days.group(1)) if days else None, "people": int(people.group(1)) if people else None, "children": int(children.group(1)) if children else None, "budget": amount, "currency": "INR", "distance_km": float(distance.group(1)) if distance else None, "transport": transport, "preferences": preferences, "dietary": dietary, "constraints": []}

