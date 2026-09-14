import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import requests

def get_weather(latitude: Optional[float] = None, longitude: Optional[float] = None, location: Optional[str] = None, forecast_days: int = 3) -> Dict[str, Any]:
    endpoint, key = os.getenv("WEATHER_API_URL", ""), os.getenv("WEATHER_API_KEY", "")
    if not endpoint or not key:
        return {"ok": False, "error_code": "weather_unavailable", "message": "Current weather information is unavailable because no weather provider is configured.", "retryable": False, "location": location, "current": None, "forecast": [], "source": None, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(endpoint, params={"key": key, "q": location or f"{latitude},{longitude}", "days": max(1, min(int(forecast_days), 7))}, timeout=8)
        response.raise_for_status(); data = response.json(); current = data.get("current", {})
        return {"ok": True, "location": data.get("location", {}).get("name", location), "current": {"temperature_c": current.get("temp_c"), "feels_like_c": current.get("feelslike_c"), "condition": current.get("condition", {}).get("text"), "rain_probability": None, "humidity": current.get("humidity"), "wind_kmh": current.get("wind_kph")}, "forecast": data.get("forecast", {}).get("forecastday", []), "source": "configured_weather_provider", "retrieved_at": datetime.now(timezone.utc).isoformat()}
    except requests.RequestException:
        return {"ok": False, "error_code": "weather_unavailable", "message": "Current weather information is unavailable.", "retryable": True, "location": location, "current": None, "forecast": [], "source": None, "retrieved_at": datetime.now(timezone.utc).isoformat()}
