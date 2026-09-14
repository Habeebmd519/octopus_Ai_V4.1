def should_use_live_search(message: str) -> bool:
    msg = (message or "").lower().strip()

    live_search_triggers = [
        "latest",
        "current",
        "today",
        "now",
        "open now",
        "opening time",
        "closing time",
        "timing",
        "time now",
        "entry fee",
        "ticket price",
        "price",
        "weather",
        "temperature",
        "rain",
        "train",
        "bus",
        "flight",
        "news",
        "nearby",
        "near me",
        "hospital nearby",
        "restaurant nearby",
        "hotel nearby",
        "petrol pump",
        "atm nearby",
        "pharmacy nearby",
        "is it open",
        "available now"
    ]

    return any(trigger in msg for trigger in live_search_triggers)


def build_live_search_query(message: str, location: dict = None, intent: str = "") -> str:
    from datetime import datetime
    msg = (message or "").strip()
    location = location or {}

    kerala_words = [
        "kerala",
        "kochi",
        "ernakulam",
        "munnar",
        "wayanad",
        "calicut",
        "kozhikode",
        "malappuram",
        "trivandrum",
        "thiruvananthapuram",
        "kollam",
        "alappuzha",
        "idukki",
        "thrissur",
        "kannur",
        "kasaragod",
        "palakkad",
        "pathanamthitta",
        "kottayam"
    ]

    if any(word in msg.lower() for word in kerala_words):
        return msg
    # Kerala is useful default context only for local-intelligence wording.
    local_context = any(word in msg.lower() for word in ("near", "nearby", "district", "kerala", "munnar", "wayanad", "kochi", "events"))
    city = location.get("city") or location.get("district") or ""
    if city and local_context:
        suffix = f" {city} Kerala"
        if any(x in msg.lower() for x in ("today", "tomorrow", "weekend", "latest", "events")):
            suffix += " " + datetime.now().strftime("%B %Y")
        return msg + suffix
    return msg
