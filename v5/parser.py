
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------
# V5 QUERY PARSER
# English + Malayalam
# Deterministic: no LLM required
# ---------------------------------------------------------

MODE_WORDS = {
    "fun",
    "quiz",
    "study",
    "travel",
    "explore",
    "local",
    "research",
    "story",
    "normal",
}

LIVE_WORDS = {
    "latest",
    "current",
    "today",
    "now",
    "tonight",
    "tomorrow",
    "this week",
    "open now",
    "opening",
    "closing",
    "price",
    "fee",
    "weather",
    "temperature",
    "news",
    "available",
    "availability",
    "event",
    "events",
    "schedule",
    "status",
}

QUESTION_WORDS = {
    "what",
    "where",
    "when",
    "why",
    "who",
    "which",
    "how",
    "can",
    "is",
    "are",
    "do",
    "does",
    "did",
}

# Common Kerala destinations / places.
# This is intentionally deterministic and can be expanded later.
KERALA_PLACES = {
    "munnar": "Munnar",
    "മുന്നാർ": "Munnar",
    "മുന്നാറിൽ": "Munnar",
    "kochi": "Kochi",
    "കൊച്ചി": "Kochi",
    "കൊച്ചിയിൽ": "Kochi",
    "fort kochi": "Fort Kochi",
    "ഫോർട്ട് കൊച്ചി": "Fort Kochi",
    "wayanad": "Wayanad",
    "വയനാട്": "Wayanad",
    "വയനാട്ടിൽ": "Wayanad",
    "alleppey": "Alappuzha",
    "alappuzha": "Alappuzha",
    "ആലപ്പുഴ": "Alappuzha",
    "ആലപ്പുഴയിൽ": "Alappuzha",
    "kumarakom": "Kumarakom",
    "കുമരകം": "Kumarakom",
    "thekkady": "Thekkady",
    "തേക്കടി": "Thekkady",
    "kovalam": "Kovalam",
    "കോവളം": "Kovalam",
    "varkala": "Varkala",
    "വർക്കല": "Varkala",
    "varkala beach": "Varkala",
    "bekal": "Bekal",
    "ബേക്കൽ": "Bekal",
    "bekal fort": "Bekal Fort",
    "ബേക്കൽ കോട്ട": "Bekal Fort",
    "thrissur": "Thrissur",
    "തൃശ്ശൂർ": "Thrissur",
    "തൃശൂർ": "Thrissur",
    "kannur": "Kannur",
    "കണ്ണൂർ": "Kannur",
    "kasaragod": "Kasaragod",
    "കാസർഗോഡ്": "Kasaragod",
    "kollam": "Kollam",
    "കൊല്ലം": "Kollam",
    "kovalam": "Kovalam",
    "തിരുവനന്തപുരം": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "trivandrum": "Thiruvananthapuram",
    "palakkad": "Palakkad",
    "പാലക്കാട്": "Palakkad",
    "kottayam": "Kottayam",
    "കോട്ടയം": "Kottayam",
    "idukki": "Idukki",
    "ഇടുക്കി": "Idukki",
}


GREETING_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "hey octapus",
    "hello octapus",
    "namaskaram",
    "good morning",
    "good afternoon",
    "good evening",
    "നമസ്കാരം",
    "ഹായ്",
    "ഹലോ",
)


IDENTITY_PATTERNS = (
    "what is your name",
    "whats your name",
    "your name",
    "who are you",
    "നിന്റെ പേര്",
    "നിങ്ങളുടെ പേര്",
    "ആരാണ് നീ",
    "നീ ആരാണ്",
)


MODEL_PATTERNS = (
    "which model",
    "what model",
    "model are you using",
    "which ai model",
    "what ai are you using",
    "ഏത് മോഡൽ",
    "ഏത് മോഡലാണ്",
    "ഏത് ai മോഡൽ",
)


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving Malayalam Unicode."""
    return re.sub(r"\s+", " ", (text or "").strip())


def tokens(text: str) -> List[str]:
    """
    Tokenize English/numeric text while also retaining Malayalam words.

    The old parser only recognized [a-z0-9], which caused Malayalam
    words such as 'മുന്നാറിൽ' to disappear completely.
    """
    value = normalize_text(text).lower()

    # Malayalam Unicode block + English + numbers.
    return re.findall(
        r"[a-zA-Z0-9]+|[\u0D00-\u0D7F]+",
        value,
    )


def detect_language(text: str) -> str:
    """
    Return:
      ml   = Malayalam dominant
      en   = English dominant
      mixed = both
    """
    value = text or ""

    malayalam_count = len(re.findall(r"[\u0D00-\u0D7F]", value))
    english_count = len(re.findall(r"[A-Za-z]", value))

    if malayalam_count and english_count:
        return "mixed"

    if malayalam_count:
        return "ml"

    return "en"


def _contains_any(text: str, patterns) -> bool:
    value = normalize_text(text).lower()
    return any(pattern.lower() in value for pattern in patterns)


def detect_greeting(text: str) -> bool:
    value = normalize_text(text).lower()

    if value in GREETING_PATTERNS:
        return True

    return any(value.startswith(pattern) for pattern in GREETING_PATTERNS)


def detect_identity(text: str) -> bool:
    return _contains_any(text, IDENTITY_PATTERNS)


def detect_model_question(text: str) -> bool:
    return _contains_any(text, MODEL_PATTERNS)


def extract_budget(text: str) -> Optional[int]:
    value = normalize_text(text)

    match = re.search(
        r"(?:₹|rs\.?|inr)\s*([\d,]+)"
        r"|\b([\d,]+)\s*(?:rupees|rs)\b",
        value,
        re.I,
    )

    if not match:
        return None

    raw = match.group(1) or match.group(2)
    return int(raw.replace(",", ""))


def extract_people(text: str) -> Optional[int]:
    value = normalize_text(text)

    # English
    match = re.search(
        r"\b(\d+)\s*(?:people|persons|person|adults|of us)\b",
        value,
        re.I,
    )

    if match:
        return int(match.group(1))

    # Malayalam:
    # 4 പേർ
    # 4 ആളുകൾ
    match = re.search(
        r"(\d+)\s*(?:പേർ|ആളുകൾ|ആളുകള്‍)",
        value,
    )

    if match:
        return int(match.group(1))

    return None


def extract_days(text: str) -> Optional[int]:
    """
    Supports:
      2 days
      2 day
      2 nights
      3 ദിവസം
      3 ദിവസത്തേക്ക്
      3 ദിവസത്തെ
      2 രാത്രികൾ
    """
    value = normalize_text(text).lower()

    # English
    match = re.search(
        r"\b(\d+)\s*(?:day|days|night|nights)\b",
        value,
        re.I,
    )

    if match:
        return int(match.group(1))

    # Malayalam
    match = re.search(
        r"(\d+)\s*"
        r"(?:ദിവസം|ദിവസങ്ങൾ|ദിവസങ്ങള്‍|ദിവസത്തേക്ക്|ദിവസത്തെ|"
        r"രാത്രി|രാത്രികൾ|രാത്രികള്‍)",
        value,
    )

    if match:
        return int(match.group(1))

    return None


def _clean_place_token(value: str) -> str:
    """
    Remove common punctuation and Malayalam case markers.
    """
    value = value.strip(" .,!?;:'\"()[]{}")

    return value.lower()


def extract_places(text: str) -> List[str]:
    """
    Detect known Kerala destinations.

    Returns canonical English names, e.g.
      ['Munnar']
    """
    value = normalize_text(text).lower()
    found = []

    # Longest first so 'fort kochi' wins over 'kochi'.
    candidates = sorted(
        KERALA_PLACES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for phrase, canonical in candidates:
        if phrase.lower() in value and canonical not in found:
            found.append(canonical)

    return found


def extract_origin_destination(text: str):
    """
    Detect simple travel patterns:

      Kochi to Munnar
      from Kochi to Munnar
      കൊച്ചിയിൽ നിന്ന് മുന്നാറിലേക്ക്
      കൊച്ചിയിൽ നിന്നും മുന്നാറിലേക്ക്

    Returns:
      origin, destination
    """
    value = normalize_text(text).lower()

    places = extract_places(value)

    # English:
    # from Kochi to Munnar
    english_match = re.search(
        r"(?:from\s+)(.+?)(?:\s+to\s+)(.+?)(?:$|[?.!,])",
        value,
        re.I,
    )

    if english_match:
        origin_text = english_match.group(1).strip()
        destination_text = english_match.group(2).strip()

        origin_places = extract_places(origin_text)
        destination_places = extract_places(destination_text)

        origin = origin_places[0] if origin_places else None
        destination = destination_places[0] if destination_places else None

        if origin or destination:
            return origin, destination

    # Simpler English:
    # Kochi to Munnar
    english_match = re.search(
        r"(.+?)\s+to\s+(.+?)(?:$|[?.!,])",
        value,
        re.I,
    )

    if english_match:
        origin_places = extract_places(english_match.group(1))
        destination_places = extract_places(english_match.group(2))

        origin = origin_places[0] if origin_places else None
        destination = destination_places[0] if destination_places else None

        if origin or destination:
            return origin, destination

    # Malayalam:
    # കൊച്ചിയിൽ നിന്ന് മുന്നാറിലേക്ക്
    # കൊച്ചിയിൽ നിന്നും മുന്നാറിലേക്ക്
    malayalam_match = re.search(
        r"(.+?)(?:യിൽ|യില്‍)?\s*"
        r"(?:നിന്ന്|നിന്നും)\s+"
        r"(.+?)(?:ലേക്ക്|യിലേക്ക്)"
        r"(?:$|[?.!,])",
        value,
    )

    if malayalam_match:
        origin_places = extract_places(malayalam_match.group(1))
        destination_places = extract_places(malayalam_match.group(2))

        origin = origin_places[0] if origin_places else None
        destination = destination_places[0] if destination_places else None

        if origin or destination:
            return origin, destination

    # If only one known place exists, treat it as destination.
    if len(places) == 1:
        return None, places[0]

    # If multiple places exist and no explicit route wording,
    # use the last place as destination.
    if len(places) >= 2:
        return places[0], places[-1]

    return None, None


def detect_question(text: str, token_list: List[str]) -> bool:
    value = normalize_text(text).lower()

    if "?" in value:
        return True

    if token_list and token_list[0] in QUESTION_WORDS:
        return True

    # Malayalam question patterns
    malayalam_question_patterns = (
        "എന്താണ്",
        "എന്താണ്",
        "എന്തൊക്കെയാണ്",
        "എവിടെയാണ്",
        "എവിടെ",
        "എപ്പോൾ",
        "എന്തുകൊണ്ട്",
        "എങ്ങനെ",
        "ആരാണ്",
        "എത്ര",
        "നല്ല സ്ഥലങ്ങൾ",
        "നല്ല സ്ഥലങ്ങള്‍",
        "പോകാൻ",
        "പോകാന്‍",
        "കാണാൻ",
        "കാണാന്‍",
    )

    return any(pattern in value for pattern in malayalam_question_patterns)


def detect_near(text: str) -> bool:
    value = normalize_text(text).lower()

    english = (
        "near me",
        "nearby",
        "around me",
        "close to me",
        "nearest",
        "nearby places",
    )

    malayalam = (
        "എന്റെ അടുത്ത്",
        "എന്റെ അടുത്തുള്ള",
        "അടുത്തുള്ള",
        "സമീപം",
        "അടുത്ത്",
    )

    return any(x in value for x in english + malayalam)


def detect_live(text: str) -> bool:
    value = normalize_text(text).lower()

    if any(x in value for x in LIVE_WORDS):
        return True

    malayalam_live = (
        "ഇപ്പോൾ",
        "ഇപ്പോള്‍",
        "ഇന്നത്തെ",
        "ഇന്ന്",
        "നാളെ",
        "കാലാവസ്ഥ",
        "വില",
        "തുറന്നിട്ടുണ്ടോ",
        "തുറന്നിട്ടുണ്ടോ",
        "ലഭ്യമാണോ",
        "പരിപാടി",
        "വാർത്ത",
        "വാര്‍ത്ത",
    )

    return any(x in value for x in malayalam_live)


def detect_research(text: str) -> bool:
    value = normalize_text(text).lower()

    english = (
        "research",
        "compare sources",
        "evidence",
        "according to",
        "verify",
        "fact check",
        "fact-check",
    )

    malayalam = (
        "തെളിവ്",
        "ഉറവിടങ്ങൾ",
        "ഉറവിടങ്ങള്‍",
        "പരിശോധിക്കുക",
        "സ്ഥിരീകരിക്കുക",
        "താരതമ്യം",
    )

    return any(x in value for x in english + malayalam)


def detect_intent(text: str) -> str:
    """
    Basic deterministic conversational intent.
    """
    if detect_greeting(text):
        return "greeting"

    if detect_identity(text):
        return "identity"

    if detect_model_question(text):
        return "model_info"

    value = normalize_text(text).lower()

    if any(
        x in value
        for x in (
            "quiz",
            "quiz me",
            "test me",
            "ക്വിസ്",
            "എന്നെ പരീക്ഷിക്കൂ",
            "ചോദ്യങ്ങൾ ചോദിക്കൂ",
            "ചോദ്യങ്ങള്‍ ചോദിക്കൂ",
        )
    ):
        return "quiz"

    if any(
        x in value
        for x in (
            "plan a trip",
            "travel",
            "trip",
            "itinerary",
            "യാത്ര",
            "യാത്രാ പദ്ധതി",
            "ട്രിപ്പ്",
            "പോകാൻ",
            "പോകാന്‍",
        )
    ):
        return "travel"

    if detect_research(text):
        return "research"

    return "general"


def parse_query(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    location: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    q = normalize_text(message)
    q_lower = q.lower()

    t = tokens(q)

    language = detect_language(q)

    live = detect_live(q)
    near = detect_near(q)
    research = detect_research(q)
    question = detect_question(q, t)

    budget = extract_budget(q)
    people = extract_people(q)
    days = extract_days(q)

    places = extract_places(q)
    origin, destination = extract_origin_destination(q)

    # If explicit origin/destination wasn't found but there is
    # exactly one recognized place, it is the destination.
    if not destination and len(places) == 1:
        destination = places[0]

    # Browser/device location is separate from a place mentioned
    # in the user's message.
    browser_location = bool(
        (location or {}).get("lat") is not None
        and (location or {}).get("lng") is not None
    )

    intent = detect_intent(q)

    return {
        "raw": q,
        "tokens": t,

        # Language
        "language": language,

        # Intent
        "intent": intent,

        # Query properties
        "live": live,
        "near": near,
        "research": research,
        "question": question,

        # Conversation shortcuts
        "greeting": detect_greeting(q),
        "identity": detect_identity(q),
        "model_info": detect_model_question(q),

        # Travel / planning entities
        "days": days,
        "people": people,
        "budget": budget,
        "places": places,
        "origin": origin,
        "destination": destination,

        # Location from browser/device
        "has_location": browser_location,

        # Useful metadata
        "length": len(t),
    }