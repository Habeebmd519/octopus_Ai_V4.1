from v5.engine import OctapusV5Engine


def make_engine():
    return OctapusV5Engine({
        "search_places": lambda a: {"ok": True, "places": [{"name": "Munnar", "description": "Tea hills"}]},
        "search_knowledge": lambda a: {"ok": True, "results": [{"title": "Kerala", "description": "A state of India"}]},
        "search_services": lambda a: {"ok": True, "places": [{"name": "Cafe", "description": "Nearby cafe"}]},
        "travel_info": lambda a: {"ok": True, "travelInfo": {"distanceText": "130 km", "durationText": "4 hr"}},
        "live_search": lambda a: {"ok": True, "results": [{"title": "Source", "url": "https://kerala.gov.in", "content": "Official"}]},
    })


def test_no_llm_normal():
    r = make_engine().run("Tell me about Munnar", mode="normal")
    assert r["aiUsed"] is False
    assert r["llmCalls"] == 0
    assert "Munnar" in r["reply"]


def test_quiz_state():
    r = make_engine().run("quiz me on Kerala", mode="quiz", state={})
    assert r["mode"] == "quiz"
    assert r["state"]["question"] == 1
    assert r["state"]["expected"]


def test_mode_specific_fun():
    r = make_engine().run("I am bored", mode="fun")
    assert r["mode"] == "fun"
    assert any(x in r["reply"].lower() for x in ("challenge", "fun", "boredom"))


def test_research_uses_no_llm():
    r = make_engine().run("research Kerala tourism", mode="research")
    assert r["llmCalls"] == 0
