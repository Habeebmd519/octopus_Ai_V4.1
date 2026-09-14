def build_v4_response(
    answer,
    places=None,
    live_answer=None,
    live_results=None,
    intent=None,
    source="local_data",
    intelligence=None,
):
    return {
        "version": "v4.2",
        "intent": intent,
        "source": source,
        "answer": answer,
        "places": places or [],
        "liveAnswer": live_answer,
        "liveResults": live_results or [],
        "intelligence": intelligence or {},
    }


# Backward-compatible name used by the legacy V3 routes.
build_v3_response = build_v4_response
