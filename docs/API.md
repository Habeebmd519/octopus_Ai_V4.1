# V4 Agent API

## POST /api/agent/context

Request:
```json
{
  "message": "best romantic places in Munnar",
  "history": [],
  "currentPlaceId": null,
  "lastMatchedPlaceIds": [],
  "userLat": 0,
  "userLng": 0,
  "userLocationText": ""
}
```

Returns the system prompt, tool schemas, routing information and a small initial context.

## POST /api/agent/tool

Request:
```json
{
  "name": "search_places",
  "arguments": {"query": "romantic places in Munnar", "limit": 6}
}
```

## GET /api/agent/tools
Returns the current tool definitions.

## GET /api/agent/health
Returns V4/Puter configuration and cache status.
