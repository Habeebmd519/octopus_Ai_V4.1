# Octopus AI activity contract

The upgraded frontend keeps recent history as chat sessions and sends privacy-aware activity events.

Recommended backend endpoints:

POST /api/activity/event
- receives one event
- fields include visitor_id, session_id/chat_id, timestamp, language, mode, visibility, user_agent, viewport
- location is included only after browser permission
- chat_message may include question (currently limited to 1000 chars)

POST /api/activity/batch
- receives {"events":[...]}

POST /api/activity/location
- receives granted location
- optionally reverse-geocodes to a broad district/area and returns {"ok":true,"area":"Malappuram"}

Frontend events:
page_open, page_visible, page_hidden, active, heartbeat, chat_created, chat_open,
chat_message, chat_response, chat_error, ai_fallback, language_changed, mode_changed,
location_granted, location_denied, puter_connected, voice_started.

Privacy:
- visitor_id is a random anonymous browser ID stored locally.
- no automatic location request.
- coordinates are sent only after explicit browser permission.
- dashboard should aggregate by broad area/district rather than exposing individual coordinates.
- language and mode are included with AI requests.
