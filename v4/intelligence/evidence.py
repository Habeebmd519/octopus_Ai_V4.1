from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List
from urllib.parse import urlsplit, urlunsplit


OFFICIAL_SUFFIXES = (".gov.in", ".nic.in", ".kerala.gov.in", ".ac.in")
HIGH_AUTHORITY_DOMAINS = {
    "irctc.co.in", "indianrail.gov.in", "kerala.gov.in",
    "keralatourism.org", "imd.gov.in", "mohfw.gov.in",
}


def clean_url(url: str) -> str:
    p = urlsplit(url or "")
    return urlunsplit((p.scheme, p.netloc, p.path, "", "")) if p.scheme in {"http", "https"} else ""


def source_authority(url: str) -> float:
    host = (urlsplit(url or "").netloc or "").lower().split(":")[0]
    if host in HIGH_AUTHORITY_DOMAINS:
        return 1.0
    if host.endswith(OFFICIAL_SUFFIXES):
        return 0.92
    if host.endswith(".org"):
        return 0.68
    if host.endswith(".edu") or host.endswith(".ac.in"):
        return 0.85
    if host:
        return 0.55
    return 0.25


def normalize_web_results(results: Iterable[Dict[str, Any]], source_type: str = "web") -> List[Dict[str, Any]]:
    out, seen = [], set()
    for item in results or []:
        url = clean_url(str(item.get("url", "")))
        key = url or str(item.get("title", "")).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        published = item.get("published_at") or item.get("publishedAt")
        out.append({
            "source_type": source_type,
            "title": str(item.get("title", "")).strip(),
            "url": url,
            "publisher": urlsplit(url).netloc,
            "published_at": published,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "claim": str(item.get("content", item.get("snippet", ""))).strip()[:1000],
            "content": str(item.get("content", "")).strip()[:4000],
            "relevance": float(item.get("score", 0) or 0),
            "freshness": 0.85 if published else 0.55,
            "authority": source_authority(url),
        })
    out.sort(key=lambda x: (x["authority"], x["relevance"], x["freshness"]), reverse=True)
    return out
