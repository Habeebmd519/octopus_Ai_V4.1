from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class LocationContext:
    available: bool = False
    permissionState: str = "unknown"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracyMeters: Optional[float] = None
    timestamp: Optional[str] = None
    formattedAddress: str = ""
    area: str = ""
    city: str = ""
    district: str = ""
    state: str = ""
    country: str = ""
    source: str = "unknown"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def valid_coordinates(latitude: Any, longitude: Any) -> bool:
    try:
        lat, lng = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return False
    return -90 <= lat <= 90 and -180 <= lng <= 180 and not (lat == 0 and lng == 0)


def parse_location(payload: Dict[str, Any]) -> LocationContext:
    raw = payload.get("location") if isinstance(payload.get("location"), dict) else payload
    lat = raw.get("latitude", raw.get("lat", raw.get("userLat")))
    lng = raw.get("longitude", raw.get("lng", raw.get("userLng")))
    valid = valid_coordinates(lat, lng)
    return LocationContext(
        available=bool(raw.get("available", valid)) and valid,
        permissionState=str(raw.get("permissionState", "unknown")),
        latitude=float(lat) if valid else None,
        longitude=float(lng) if valid else None,
        accuracyMeters=_number_or_none(raw.get("accuracyMeters", raw.get("accuracy"))),
        timestamp=raw.get("timestamp"), formattedAddress=str(raw.get("formattedAddress", raw.get("userLocationText", ""))),
        area=str(raw.get("area", "")), city=str(raw.get("city", "")),
        district=str(raw.get("district", "")), state=str(raw.get("state", "")),
        country=str(raw.get("country", "")), source=str(raw.get("source", "unknown")),
    )


def _number_or_none(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (ValueError, TypeError):
        return None

