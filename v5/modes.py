from dataclasses import dataclass, field
from typing import Dict, List, Set

@dataclass(frozen=True)
class ModeProfile:
    name: str
    description: str
    tone: str
    max_words: int
    tools: Set[str] = field(default_factory=set)
    followup_style: str = "helpful"
    response_sections: List[str] = field(default_factory=list)
    addictive_loop: bool = False

MODE_PROFILES: Dict[str, ModeProfile] = {
    "normal": ModeProfile("normal", "General Octapus assistant", "warm, practical, natural", 180,
        {"places", "knowledge", "services", "live", "travel"}, response_sections=["answer", "next_step"]),
    "fun": ModeProfile("fun", "Playful entertainment and discovery", "playful, energetic, witty, warm", 150,
        {"places", "knowledge", "services", "live"}, followup_style="challenge_or_tease", response_sections=["hook", "content", "next_play"], addictive_loop=True),
    "quiz": ModeProfile("quiz", "Interactive learning game", "encouraging, game-like, concise", 110,
        {"knowledge", "live"}, followup_style="one_question_at_a_time", response_sections=["feedback", "score", "question"], addictive_loop=True),
    "study": ModeProfile("study", "Teaching and practice", "clear, patient, Socratic", 220,
        {"knowledge", "live", "places"}, response_sections=["explain", "example", "practice"]),
    "travel": ModeProfile("travel", "Trip planning and travel operations", "decisive, practical, time-aware", 260,
        {"places", "services", "travel", "live", "knowledge"}, response_sections=["plan", "logistics", "tips"]),
    "explore": ModeProfile("explore", "Discovery and local exploration", "curious, vivid, concise", 190,
        {"places", "services", "knowledge", "live", "travel"}, response_sections=["discoveries", "why", "next"]),
    "local": ModeProfile("local", "Nearby practical assistant", "direct, useful, location-aware", 150,
        {"services", "places", "live", "travel"}, response_sections=["nearby", "practical"]),
    "research": ModeProfile("research", "Evidence-first research", "neutral, precise, source-conscious", 320,
        {"knowledge", "live", "places", "services", "travel"}, response_sections=["finding", "evidence", "caveat"]),
    "story": ModeProfile("story", "Interactive storytelling", "cinematic, imaginative, conversational", 300,
        {"knowledge", "places"}, followup_style="continue_story", response_sections=["scene", "choice"], addictive_loop=True),
}

ALIASES = {
    "default": "normal", "chat": "normal", "assistant": "normal",
    "game": "fun", "play": "fun", "funny": "fun",
    "quiz me": "quiz", "learning": "study", "learn": "study",
    "trip": "travel", "holiday": "travel", "tour": "travel",
    "discover": "explore", "nearby": "local", "researching": "research",
    "storytelling": "story", "stories": "story",
}

def normalize_mode(value: str) -> str:
    key = (value or "normal").strip().lower().replace("_", "-")
    if key in MODE_PROFILES:
        return key
    return ALIASES.get(key, "normal")


def detect_mode(message: str, requested_mode: str = "", current_mode: str = "normal") -> str:
    if requested_mode:
        return normalize_mode(requested_mode)
    q = (message or "").lower()
    explicit = [
        ("quiz", "quiz"), ("fun mode", "fun"), ("joke", "fun"),
        ("game", "fun"), ("study mode", "study"), ("teach me", "study"),
        ("trip plan", "travel"), ("travel mode", "travel"),
        ("explore mode", "explore"), ("near me", "local"),
        ("research mode", "research"), ("research this", "research"),
        ("tell me a story", "story"), ("story mode", "story"),
    ]
    for needle, mode in explicit:
        if needle in q:
            return mode
    return normalize_mode(current_mode)
