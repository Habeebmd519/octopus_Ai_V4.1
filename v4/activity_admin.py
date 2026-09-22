"""
Octopus AI — Real Activity + Admin Dashboard API
=================================================

This module is designed to plug into the EXISTING Flask + Firebase/Firestore
backend. It does NOT create fake/demo data.

It provides:

    POST /api/activity/event
    POST /api/activity/batch
    POST /api/activity/location

    GET  /api/admin/dashboard?range=1d
    GET  /api/admin/dashboard?range=7d
    GET  /api/admin/dashboard?range=30d
    GET  /api/admin/dashboard?range=90d

Firestore collections:

    octopus_activity
    octopus_locations

Integration in v4/gpt.py
------------------------

After your existing Flask `app` and Firebase Admin `db` are initialized:

    from v4.activity_admin import register_activity_admin
    register_activity_admin(app, db)

IMPORTANT:
- Do NOT initialize Firebase a second time here.
- Do NOT put serviceAccountKey.json in the frontend.
- This module stores anonymous browser visitor IDs, not names/emails.
- Location is stored only when the browser explicitly grants it.
- The dashboard never creates fake numbers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from flask import jsonify, request


ACTIVITY_COLLECTION = "octopus_activity"
LOCATION_COLLECTION = "octopus_locations"

ALLOWED_RANGES = {
    "1d": 1,
    "7d": 7,
    "30d": 30,
    "90d": 90,
}


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_datetime(value: Any) -> datetime:
    """Convert Firestore/Python/string timestamps to timezone-aware UTC."""
    if value is None:
        return _utc_now()

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    # Firestore Timestamp objects normally expose to_datetime().
    try:
        dt = value.to_datetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

    return _utc_now()


def _text(value: Any, limit: int = 2000) -> str:
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _range_days(value: str) -> int:
    return ALLOWED_RANGES.get(value, 7)


def _masked_visitor(visitor_id: str) -> str:
    """
    Keep the dashboard useful without exposing the complete anonymous
    browser identifier.
    """
    visitor_id = _text(visitor_id, 120)

    if not visitor_id:
        return "anonymous"

    if len(visitor_id) <= 12:
        return visitor_id

    return visitor_id[:8] + "•••" + visitor_id[-4:]


def _relative_time(dt: datetime) -> str:
    now = _utc_now()
    seconds = max(0, int((now - dt).total_seconds()))

    if seconds < 60:
        return "just now"

    if seconds < 3600:
        return f"{seconds // 60} min ago"

    if seconds < 86400:
        return f"{seconds // 3600} hr ago"

    return f"{seconds // 86400} d ago"


def _area_from_location(location: Any) -> str:
    """
    Only return an area if the application actually supplied one.

    We intentionally DO NOT reverse-geocode coordinates here. The browser
    currently sends lat/lng, and converting that into a district without a
    real reverse-geocoding step would be misleading.
    """
    if not isinstance(location, dict):
        return "Private / unknown"

    area = _text(location.get("area"), 120)
    return area or "Private / unknown"


def _clean_location(location: Any) -> Optional[Dict[str, Any]]:
    """
    Keep only the location fields needed by the backend.

    Location is accepted only when status == granted.
    """
    if not isinstance(location, dict):
        return None

    if location.get("status") != "granted":
        return None

    result: Dict[str, Any] = {
        "status": "granted",
    }

    # Coordinates are optional because browser permission can be granted
    # but a coordinate may still be unavailable.
    for key in ("lat", "lng", "accuracy"):
        if location.get(key) is not None:
            result[key] = location.get(key)

    area = _text(location.get("area"), 120)
    if area:
        result["area"] = area

    return result


def _normalize_event(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Whitelist fields from the browser.

    Never blindly store the entire request body.
    """
    location = _clean_location(body.get("location"))

    return {
        "event": _text(body.get("event"), 80) or "unknown",
        "visitor_id": _text(body.get("visitor_id"), 160),
        "session_id": _text(body.get("session_id"), 160),
        "chat_id": _text(body.get("chat_id"), 160),

        # The browser timestamp is kept as metadata, but the server timestamp
        # is authoritative for dashboard time/range queries.
        "client_timestamp": _text(body.get("timestamp"), 80),

        "timestamp": _utc_now(),

        "language": _text(body.get("language"), 30),
        "mode": _text(body.get("mode"), 40),

        "question": _text(body.get("question"), 3000),
        "question_length": max(0, _safe_int(body.get("question_length"))),

        "source": _text(body.get("source"), 40),
        "error": _text(body.get("error"), 1000),

        "location_consent": _text(body.get("location_consent"), 30),
        "location": location,

        "viewport": (
            body.get("viewport")
            if isinstance(body.get("viewport"), dict)
            else None
        ),
    }


def _validate_visitor(body: Dict[str, Any]):
    visitor_id = _text(body.get("visitor_id"), 160)

    if not visitor_id:
        return None, (
            jsonify({
                "ok": False,
                "error": "visitor_id is required",
            }),
            400,
        )

    return visitor_id, None


# ---------------------------------------------------------------------------
# Firestore read helper
# ---------------------------------------------------------------------------

def _read_events(db, start: datetime) -> List[Dict[str, Any]]:
    """
    Read only events inside the requested period.

    This uses a single timestamp range, so Firestore does not need a custom
    composite index for this query.
    """
    query = (
        db.collection(ACTIVITY_COLLECTION)
        .where("timestamp", ">=", start)
    )

    events: List[Dict[str, Any]] = []

    for snap in query.stream():
        try:
            data = snap.to_dict() or {}
            data["_dt"] = _to_utc_datetime(data.get("timestamp"))
            events.append(data)
        except Exception:
            # One malformed document must not break the whole dashboard.
            continue

    return events


# ---------------------------------------------------------------------------
# Dashboard aggregation
# ---------------------------------------------------------------------------

def _build_dashboard(events: List[Dict[str, Any]], days: int) -> Dict[str, Any]:
    now = _utc_now()
    active_cutoff = now - timedelta(minutes=5)

    users: Set[str] = {
        _text(e.get("visitor_id"), 160)
        for e in events
        if _text(e.get("visitor_id"), 160)
    }

    questions = [
        e for e in events
        if e.get("event") == "chat_message"
        and _text(e.get("question"), 3000)
    ]

    chats = {
        _text(e.get("chat_id"), 160) or _text(e.get("session_id"), 160)
        for e in events
        if _text(e.get("chat_id"), 160)
        or _text(e.get("session_id"), 160)
    }

    active_users = {
        _text(e.get("visitor_id"), 160)
        for e in events
        if (
            _text(e.get("visitor_id"), 160)
            and e.get("_dt", now) >= active_cutoff
            and e.get("event") in {
                "active",
                "heartbeat",
                "page_visible",
                "chat_message",
                "chat_response",
                "voice_started",
            }
        )
    }

    voice_events = [
        e for e in events
        if e.get("event") == "voice_started"
    ]

    voice_users = {
        _text(e.get("visitor_id"), 160)
        for e in voice_events
        if _text(e.get("visitor_id"), 160)
    }

    location_users = {
        _text(e.get("visitor_id"), 160)
        for e in events
        if (
            _text(e.get("visitor_id"), 160)
            and (
                e.get("location_consent") == "granted"
                or isinstance(e.get("location"), dict)
            )
        )
    }

    puter_responses = [
        e for e in events
        if e.get("event") == "chat_response"
        and _text(e.get("source"), 40).lower() == "puter"
    ]

    fallback_events = [
        e for e in events
        if (
            e.get("event") == "ai_fallback"
            or (
                e.get("event") == "chat_response"
                and _text(e.get("source"), 40).lower() == "v5"
            )
        )
    ]

    # -----------------------------------------------------------------------
    # New vs returning users
    # -----------------------------------------------------------------------

    visitor_dates: Dict[str, Set[str]] = defaultdict(set)

    for event in events:
        visitor_id = _text(event.get("visitor_id"), 160)

        if visitor_id:
            visitor_dates[visitor_id].add(
                event["_dt"].date().isoformat()
            )

    new_users = sum(
        1 for dates in visitor_dates.values()
        if len(dates) == 1
    )

    returning_users = sum(
        1 for dates in visitor_dates.values()
        if len(dates) > 1
    )

    # -----------------------------------------------------------------------
    # Language / mode
    # -----------------------------------------------------------------------

    language_counts = Counter(
        _text(e.get("language"), 30) or "unknown"
        for e in questions
    )

    mode_counts = Counter(
        _text(e.get("mode"), 40) or "normal"
        for e in questions
    )

    # -----------------------------------------------------------------------
    # Areas
    # -----------------------------------------------------------------------

    area_counts = Counter()

    for event in events:
        location = event.get("location")

        if isinstance(location, dict):
            area = _text(location.get("area"), 120)

            if area:
                area_counts[area] += 1

    # -----------------------------------------------------------------------
    # Daily chart
    # -----------------------------------------------------------------------

    daily_users: Dict[str, Set[str]] = defaultdict(set)
    daily_questions = Counter()

    for event in events:
        day = event["_dt"].date().isoformat()

        visitor_id = _text(event.get("visitor_id"), 160)

        if visitor_id:
            daily_users[day].add(visitor_id)

        if event.get("event") == "chat_message":
            daily_questions[day] += 1

    daily: List[Dict[str, Any]] = []

    for offset in range(days - 1, -1, -1):
        day = (now - timedelta(days=offset)).date().isoformat()

        daily.append({
            "date": day,
            "users": len(daily_users.get(day, set())),
            "questions": int(daily_questions.get(day, 0)),
        })

    # -----------------------------------------------------------------------
    # Recent questions
    # -----------------------------------------------------------------------

    question_rows: List[Dict[str, Any]] = []

    for event in sorted(
        questions,
        key=lambda item: item["_dt"],
        reverse=True,
    )[:100]:

        question_rows.append({
            "question": _text(event.get("question"), 3000),
            "language": _text(event.get("language"), 30),
            "mode": _text(event.get("mode"), 40),
            "visitor_id": _masked_visitor(
                _text(event.get("visitor_id"), 160)
            ),
            "time": _relative_time(event["_dt"]),
        })

    # -----------------------------------------------------------------------
    # Live activity
    # -----------------------------------------------------------------------

    activity_rows: List[Dict[str, Any]] = []

    for event in sorted(
        events,
        key=lambda item: item["_dt"],
        reverse=True,
    )[:100]:

        location = event.get("location")

        activity_rows.append({
            "time": _relative_time(event["_dt"]),
            "event": _text(event.get("event"), 80),
            "visitor_id": _masked_visitor(
                _text(event.get("visitor_id"), 160)
            ),
            "language": _text(event.get("language"), 30),
            "mode": _text(event.get("mode"), 40),
            "area": _area_from_location(location),
        })

    # -----------------------------------------------------------------------
    # User summary table
    # -----------------------------------------------------------------------

    by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for event in events:
        visitor_id = _text(event.get("visitor_id"), 160)

        if visitor_id:
            by_user[visitor_id].append(event)

    user_rows: List[Dict[str, Any]] = []

    sorted_users = sorted(
        by_user.items(),
        key=lambda item: max(
            event["_dt"] for event in item[1]
        ),
        reverse=True,
    )

    for visitor_id, user_events in sorted_users[:200]:
        user_events = sorted(
            user_events,
            key=lambda item: item["_dt"],
        )

        user_questions = [
            event
            for event in user_events
            if event.get("event") == "chat_message"
        ]

        latest = user_events[-1]
        latest_location = latest.get("location")

        last_question = (
            user_questions[-1]
            if user_questions
            else {}
        )

        user_rows.append({
            "visitor_id": _masked_visitor(visitor_id),

            "first_seen": user_events[0]["_dt"].isoformat(),
            "last_seen": user_events[-1]["_dt"].isoformat(),

            "questions": len(user_questions),

            "language": _text(
                last_question.get("language")
                or latest.get("language"),
                30,
            ),

            "mode": _text(
                last_question.get("mode")
                or latest.get("mode"),
                40,
            ),

            "area": _area_from_location(latest_location),
        })

    return {
        "ok": True,
        "demo": False,
        "range": f"{days}d",
        "generated_at": now.isoformat(),

        "summary": {
            "total_users": len(users),
            "active_now": len(active_users),
            "questions": len(questions),
            "chats": len(chats),

            "voice_starts": len(voice_events),
            "voice_users": len(voice_users),

            "puter_responses": len(puter_responses),
            "v5_fallback": len(fallback_events),

            "location_shared": len(location_users),

            "new_users": new_users,
            "returning_users": returning_users,
        },

        "daily": daily,

        "languages": [
            {
                "name": name,
                "count": count,
            }
            for name, count in language_counts.most_common()
        ],

        "modes": [
            {
                "name": name,
                "count": count,
            }
            for name, count in mode_counts.most_common()
        ],

        "areas": [
            {
                "name": name,
                "count": count,
            }
            for name, count in area_counts.most_common(20)
        ],

        "questions": question_rows,
        "activity": activity_rows,
        "users": user_rows,
    }


# ---------------------------------------------------------------------------
# Flask registration
# ---------------------------------------------------------------------------

def register_activity_admin(app, db):
    """
    Register all activity + admin routes on the EXISTING Flask application.

    Call exactly once from v4/gpt.py after Firebase `db` exists.
    """

    if db is None:
        raise RuntimeError(
            "Firestore db is None. Initialize Firebase before "
            "calling register_activity_admin(app, db)."
        )

    # -----------------------------------------------------------------------
    # Client activity ingestion
    # -----------------------------------------------------------------------

    @app.route("/api/activity/event", methods=["POST"])
    def activity_event():
        body = request.get_json(silent=True)

        if not isinstance(body, dict):
            return jsonify({
                "ok": False,
                "error": "JSON object required",
            }), 400

        visitor_id, error = _validate_visitor(body)

        if error:
            return error

        event = _normalize_event(body)
        event["visitor_id"] = visitor_id

        # Server timestamp is authoritative.
        event["timestamp"] = _utc_now()

        db.collection(ACTIVITY_COLLECTION).add(event)

        return jsonify({
            "ok": True,
            "stored": True,
        })


    @app.route("/api/activity/batch", methods=["POST"])
    def activity_batch():
        body = request.get_json(silent=True)

        if not isinstance(body, dict):
            return jsonify({
                "ok": False,
                "error": "JSON object required",
            }), 400

        events = body.get("events")

        if not isinstance(events, list):
            return jsonify({
                "ok": False,
                "error": "events must be a list",
            }), 400

        # Prevent an accidental huge request from writing unlimited data.
        events = events[-100:]

        batch = db.batch()
        collection = db.collection(ACTIVITY_COLLECTION)

        saved = 0

        for raw in events:
            if not isinstance(raw, dict):
                continue

            visitor_id, error = _validate_visitor(raw)

            if error:
                continue

            event = _normalize_event(raw)
            event["visitor_id"] = visitor_id
            event["timestamp"] = _utc_now()

            ref = collection.document()

            batch.set(ref, event)
            saved += 1

        if saved:
            batch.commit()

        return jsonify({
            "ok": True,
            "stored": True,
            "saved": saved,
        })


    @app.route("/api/activity/location", methods=["POST"])
    def activity_location():
        body = request.get_json(silent=True)

        if not isinstance(body, dict):
            return jsonify({
                "ok": False,
                "error": "JSON object required",
            }), 400

        visitor_id = _text(body.get("visitor_id"), 160)
        session_id = _text(body.get("session_id"), 160)
        location = body.get("location")

        if not visitor_id:
            return jsonify({
                "ok": False,
                "error": "visitor_id is required",
            }), 400

        if not isinstance(location, dict):
            return jsonify({
                "ok": False,
                "error": "location object is required",
            }), 400

        if location.get("status") != "granted":
            return jsonify({
                "ok": False,
                "error": "location was not explicitly granted",
            }), 400

        safe_location = _clean_location(location)

        if not safe_location:
            return jsonify({
                "ok": False,
                "error": "invalid granted location",
            }), 400

        db.collection(LOCATION_COLLECTION).add({
            "visitor_id": visitor_id,
            "session_id": session_id,
            "location": safe_location,
            "timestamp": _utc_now(),
        })

        return jsonify({
            "ok": True,
            "stored": True,
            "area": safe_location.get("area"),
        })


    # -----------------------------------------------------------------------
    # Admin dashboard
    # -----------------------------------------------------------------------

    @app.route("/api/admin/dashboard", methods=["GET"])
    def admin_dashboard():
        range_name = _text(
            request.args.get("range", "7d"),
            10,
        )

        days = _range_days(range_name)

        start = _utc_now() - timedelta(days=days)

        events = _read_events(db, start)

        result = _build_dashboard(events, days)

        return jsonify(result)


    # -----------------------------------------------------------------------
    # Simple backend test endpoint
    # -----------------------------------------------------------------------

    @app.route("/api/admin/dashboard/health", methods=["GET"])
    def admin_dashboard_health():
        """
        Use this in the browser first:

            http://127.0.0.1:5002/api/admin/dashboard/health

        A successful response proves the module is registered.
        """
        return jsonify({
            "ok": True,
            "service": "octopus-admin-dashboard",
            "firestore": True,
            "activity_collection": ACTIVITY_COLLECTION,
            "location_collection": LOCATION_COLLECTION,
        })


# Backwards-compatible alias in case you prefer this name.
register_activity_routes = register_activity_admin
