import random
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .evidence import rank_sources
from .modes import MODE_PROFILES, detect_mode, normalize_mode
from .parser import parse_query
from .voice import VoiceEngine

Tool = Callable[..., Dict[str, Any]]

class OctapusV5Engine:
    """Deterministic orchestration layer. LLMs are optional, never required."""

    def __init__(self, tools: Dict[str, Tool], web_research=None):
        self.tools = tools
        self.web = web_research
        self.voice = VoiceEngine()
        self.rng = random.Random()

    def _call(self, name: str, **kwargs) -> Dict[str, Any]:
        fn = self.tools.get(name)
        if not fn:
            return {"ok": False, "error": "tool_unavailable", "tool": name}
        started = time.perf_counter()
        try:
            result = fn(kwargs)
            if not isinstance(result, dict):
                result = {"ok": True, "data": result}
            result.setdefault("ok", True)
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return result
        except Exception as exc:
            return {"ok": False, "error": str(exc), "tool": name, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    def plan(self, message: str, mode: str, parsed: Dict[str, Any]) -> List[str]:
        profile = MODE_PROFILES[mode]
        q = parsed["raw"].lower()
        plan: List[str] = []
        if mode == "quiz":
            plan.append("knowledge")
        elif mode == "story":
            plan.append("knowledge" if any(x in q for x in ("kerala", "history", "place", "real")) else "none")
        elif parsed["near"] or mode == "local":
            plan.append("services")
            if "place" in q or "visit" in q:
                plan.append("places")
        elif any(x in q for x in ("restaurant", "hotel", "place", "waterfall", "beach", "visit", "where to")):
            plan.append("places")
        elif any(x in q for x in ("history", "culture", "festival", "food", "writer", "book", "government", "emergency", "education")):
            plan.append("knowledge")
        if any(x in q for x in ("how far", "distance", "route", "drive", "travel time", "get there", "from ", " to ")):
            plan.append("travel")
        if parsed["live"] or parsed["research"] or mode == "research":
            plan.append("live")
        # A general grounded query gets one cheap local retrieval pass instead of an LLM.
        if not plan and mode in ("normal", "study", "explore", "travel") and parsed["length"] >= 2:
            plan.append("places")
        # Only tools allowed by the mode profile.
        return [x for x in dict.fromkeys(plan) if x == "none" or x in profile.tools]

    def retrieve(self, message: str, mode: str, parsed: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        plan = self.plan(message, mode, parsed)
        evidence: List[Dict[str, Any]] = []
        data: Dict[str, Any] = {}
        latencies = []
        location = context.get("location") or {}
        for step in plan:
            if step == "none":
                continue
            if step == "places":
                r = self._call("search_places", query=message, limit=6)
                data["places"] = r
            elif step == "knowledge":
                category = context.get("knowledge_category", "general_kerala")
                r = self._call("search_knowledge", query=message, category=category, limit=5)
                data["knowledge"] = r
            elif step == "services":
                r = self._call("search_services", query=message, category=context.get("service_category", "food"), limit=8,
                               lat=location.get("lat", 0), lng=location.get("lng", 0))
                data["services"] = r
            elif step == "travel":
                origin = context.get("origin") or location.get("text") or "current location"
                destination = parsed.get("destination") or context.get("destination") or context.get("current_place_name") or message
                r = self._call("travel_info", origin=origin, destination=destination)
                data["travel"] = r
            elif step == "live":
                if self.web:
                    r = self.web.research(message, limit=6, deep=(mode == "research" or parsed["research"]))
                else:
                    r = self._call("live_search", query=message, limit=5)
                data["live"] = r
            for value in data.values():
                if isinstance(value, dict) and value.get("latency_ms") is not None:
                    latencies.append(value["latency_ms"])
        data["plan"] = plan
        data["latency_ms"] = round(sum(latencies), 1)
        return data

    def _first_items(self, result: Dict[str, Any], keys=("places", "results", "knowledge")) -> List[Dict[str, Any]]:
        for key in keys:
            val = result.get(key) if isinstance(result, dict) else None
            if isinstance(val, list):
                return val
        return []

    def _quiz(self, message: str, data: Dict[str, Any], state: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        score = int(state.get("score", 0) or 0)
        qnum = int(state.get("question", 0) or 0)
        last_answer = str(state.get("expected", "")).strip().lower()
        user_answer = message.strip().lower()
        feedback = ""
        if last_answer and user_answer:
            if last_answer in user_answer or user_answer in last_answer:
                score += 1; feedback = "✅ Correct!"
            else:
                feedback = f"❌ Not quite. The answer was **{state.get('expected')}**."
        items = self._first_items(data.get("knowledge", {}), ("results", "knowledge"))
        # Use structured knowledge when possible; otherwise a reliable fallback question.
        if items:
            item = items[0]
            topic = item.get("title") or item.get("name") or "Kerala"
            answer = item.get("name") or item.get("title") or topic
            question = f"Quick challenge: what can you tell me about **{topic}**?"
        else:
            question = "Quick challenge: Which state is known as God's Own Country?"
            answer = "Kerala"
        qnum += 1
        next_state = {"score": score, "question": qnum, "expected": answer, "mode": "quiz"}
        text = f"{feedback + ' ' if feedback else ''}🏆 **Score: {score}**  •  Question {qnum}\n\n{question}\n\nReply with your answer — or say **hint**."
        return text, next_state

    def _fun(self, message: str, data: Dict[str, Any], state: Dict[str, Any]) -> str:
        places = self._first_items(data.get("places", {}))
        topic = (places[0].get("name") if places else None) or "Kerala"
        prompts = [
            f"🎮 **Fun mode:** okay, let's make this interesting. Your mission: discover one surprising thing about **{topic}** and tell me if you'd actually try it.",
            f"😈 **Challenge accepted.** I found **{topic}**. You get one choice: **explore it**, **quiz yourself on it**, or **let me turn it into a mini adventure**.",
            f"🔥 **Boredom detected.** Today's target is **{topic}**. Give me 30 seconds and I'll turn it into a tiny game. Pick: **A) challenge  B) riddle  C) adventure**."
        ]
        return self.rng.choice(prompts)

    def build_response(self, message: str, mode: str, parsed: Dict[str, Any], data: Dict[str, Any], context: Dict[str, Any], state: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        if mode == "quiz":
            return self._quiz(message, data, state)
        if mode == "fun":
            return self._fun(message, data, state), state
        if mode == "story":
            place = self._first_items(data.get("places", {}))
            name = place[0].get("name") if place else context.get("current_place_name", "Kerala")
            return f"🌴 **Story mode**\n\nYou arrive in **{name}** just before sunset. The road ahead splits in two: one path leads toward a quiet view, the other toward lights, music, and something that definitely wasn't on the map.\n\n**Choose:** `quiet` or `mystery`.", state
        if mode == "research":
            live = data.get("live", {})
            results = live.get("results", []) if isinstance(live, dict) else []
            if not results:
                return "I couldn't collect enough independent sources to give you a research-grade answer yet.", state
            top = results[:4]
            lines = [f"**Research brief**\n\nI found {len(results)} relevant sources. Confidence: **{live.get('consensus', {}).get('confidence', 'low')}**."]
            for i, s in enumerate(top, 1):
                lines.append(f"{i}. **{s.get('title','Source')}** — {s.get('content','').strip()[:240]}\n   Source: {s.get('url','')}")
            lines.append("\nI kept the synthesis source-first; where sources disagree, Octapus should show the disagreement rather than silently choosing one.")
            return "\n\n".join(lines), state
        if mode == "travel":
            places = self._first_items(data.get("places", {}))
            travel = data.get("travel", {}).get("travelInfo") if isinstance(data.get("travel"), dict) else None
            lines = ["🧭 **Travel plan**"]
            if places:
                lines.append("\n**Good matches:** " + ", ".join((p.get("name") or "place") for p in places[:4]))
            if travel:
                lines.append(f"\n**Route:** {travel.get('distanceText') or travel.get('distance') or 'distance unavailable'} • {travel.get('durationText') or travel.get('duration') or 'time unavailable'}")
            if parsed.get("days"):
                lines.append(f"\n**Duration:** {parsed['days']} days")
            lines.append("\nI can turn this into a day-by-day route once the destination and starting point are clear.")
            return "\n".join(lines), state
        if mode in ("explore", "local"):
            items = self._first_items(data.get("services", {})) or self._first_items(data.get("places", {}))
            if items:
                lines = ["🧭 **What I found**"]
                for p in items[:5]:
                    lines.append(f"• **{p.get('name') or p.get('title') or 'Place'}**" + (f" — {p.get('description','')[:140]}" if p.get('description') else ""))
                return "\n".join(lines), state
        # Normal / study
        q = message.strip().lower()
        if q in {"hi", "hello", "hey", "hey octapus", "namaskaram", "good morning", "good evening"}:
            return "Namaskaram 🙏 I’m ready. Give me a place, question, plan, challenge, or switch to a mode like **fun**, **quiz**, **study**, **travel**, or **research**.", state
        places = self._first_items(data.get("places", {}))
        knowledge = self._first_items(data.get("knowledge", {}), ("results", "knowledge"))
        live = data.get("live", {})
        if mode == "study" and knowledge:
            k = knowledge[0]
            return f"📚 **Learn it simply**\n\n**{k.get('title') or k.get('name') or 'Topic'}**\n\n{k.get('description') or k.get('content') or 'I found the topic, but the stored explanation is limited.'}\n\n**Practice:** Tell me the idea back in one sentence and I’ll check it.", state
        if places:
            lines = [f"I found **{len(places)}** strong match(es):"]
            for p in places[:5]:
                lines.append(f"• **{p.get('name') or 'Place'}**" + (f" — {p.get('description','')[:150]}" if p.get('description') else ""))
            return "\n".join(lines), state
        if knowledge:
            k = knowledge[0]
            return f"**{k.get('title') or k.get('name') or 'Answer'}**\n\n{k.get('description') or k.get('content') or 'I found a relevant knowledge record but it contains limited detail.'}", state
        if isinstance(live, dict) and live.get("results"):
            s = live["results"][0]
            return f"Here’s the strongest current source I found:\n\n**{s.get('title','Source')}**\n{s.get('content','')[:600]}\n\nSource: {s.get('url','')}", state
        return "I don't have enough grounded information to answer that confidently yet. Try giving me a place, topic, time, or goal and I'll route it to the right capability.", state

    def run(self, message: str, *, mode: str = "", history=None, context=None, state=None, voice: bool = False, voice_locale: str = "") -> Dict[str, Any]:
        started = time.perf_counter()
        context = context or {}
        state = dict(state or {})
        current_mode = state.get("mode", context.get("mode", "normal"))
        selected_mode = detect_mode(message, mode, current_mode)
        parsed = parse_query(message, history or [], context.get("location"))
        data = self.retrieve(message, selected_mode, parsed, context)
        reply, new_state = self.build_response(message, selected_mode, parsed, data, context, state)
        result = {
            "version": "5.0",
            "mode": selected_mode,
            "reply": reply,
            "state": new_state,
            "parser": parsed,
            "plan": data.get("plan", []),
            "evidence": data.get("live", {}).get("results", []) if isinstance(data.get("live"), dict) else [],
            "data": data,
            "aiUsed": False,
            "llmCalls": 0,
            "voice": self.voice.prepare(reply, voice_locale) if voice else None,
            "metrics": {"latency_ms": round((time.perf_counter() - started) * 1000, 1), "tool_latency_ms": data.get("latency_ms", 0)},
        }
        return result
