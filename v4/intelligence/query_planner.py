"""
Octopus AI V4.2 research planner.

Converts the intent engine's signals into a short, ordered tool plan.
The plan is advisory: Puter remains the final agent and can skip/reorder tools
when the user's request makes that appropriate.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .freshness import classify_freshness


TOOL_PRIORITY = {
    "get_weather": 100,
    "search_nearby": 96,
    "travel_info": 94,
    "search_news": 92,
    "search_events": 90,
    "search_knowledge": 84,
    "get_place": 82,
    "search_places": 80,
    "live_search": 70,
}


def _dedupe_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for step in steps:
        tool = step.get("tool")
        if not tool or tool in seen:
            continue
        seen.add(tool)
        out.append(step)
    return out


def plan(message: str, intent: Dict[str, Any], location: Dict[str, Any]) -> Dict[str, Any]:
    freshness = intent.get("freshness") or classify_freshness(message, intent.get("sub_intent", ""))
    tools = list(intent.get("recommended_tools") or [])
    steps: List[Dict[str, Any]] = []

    if intent.get("requires_location") and intent.get("sub_intent") != "location_self":
        steps.append({"tool": "search_nearby", "reason": "The request depends on the user's location."})

    for tool in tools:
        reason = "Recommended by the intent engine."
        if tool == "live_search":
            reason = "The request contains information that can change."
        elif tool == "search_places":
            reason = "Use Octopus AI's private Kerala place database first."
        elif tool == "search_knowledge":
            reason = "Use the structured Kerala knowledge collection."
        elif tool == "travel_info":
            reason = "A route, distance or travel-time calculation is requested."
        elif tool == "get_weather":
            reason = "Weather must come from a current provider."
        elif tool == "search_nearby":
            reason = "Nearby services require current coordinates."
        steps.append({"tool": tool, "reason": reason})

    # Freshness is a safety requirement, not merely a keyword hint.
    if freshness.get("requires_web") and "live_search" not in {s["tool"] for s in steps}:
        steps.append({"tool": "live_search", "reason": "Freshness policy requires current verification."})

    steps = _dedupe_steps(steps)
    steps.sort(key=lambda x: TOOL_PRIORITY.get(x["tool"], 50), reverse=True)

    # Keep the first tool round compact. The LLM can request additional tools.
    steps = steps[:5]

    return {
        "steps": steps,
        "freshness": freshness,
        "locationAvailable": bool(location.get("available")),
        "confidence": intent.get("confidence", 0.5),
        "triggerReasons": intent.get("trigger_reasons", {}),
    }
