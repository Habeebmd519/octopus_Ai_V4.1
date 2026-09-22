# Octopus AI Frontend Upgrade

Replace `frontend/index.html` with this file and keep the existing `frontend/puter-agent.js`.

Upgrades:
1. Recent is now a list of full chat sessions. Each chat can contain many questions/answers.
2. First question becomes the chat title.
3. Each chat remembers language and mode.
4. Language: Malayalam, English, Malayalam + English.
5. Mode picker: Normal, Fun, Quiz, Study, Travel, Explore, Local, Research, Story.
6. Anonymous visitor ID for active-user analytics.
7. Activity heartbeat about every 30 seconds while the page is visible.
8. Explicit location permission flow.
9. Language/mode/location/session metadata is sent to the backend.
10. Puter stays primary and V5 stays fallback.
11. Puter request no longer sends `temperature`.
12. Existing Malayalam voice and slow running-octopus thinking animation remain.
13. Activity events are queued locally if the backend analytics endpoints are not ready yet.

See ACTIVITY_CONTRACT.md before building the dashboard/backend analytics.
