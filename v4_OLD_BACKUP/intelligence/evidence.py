from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List
from urllib.parse import urlsplit, urlunsplit

def clean_url(url: str) -> str:
    p = urlsplit(url or "")
    return urlunsplit((p.scheme, p.netloc, p.path, "", "")) if p.scheme in {"http", "https"} else ""

def normalize_web_results(results: Iterable[Dict[str, Any]], source_type: str = "web") -> List[Dict[str, Any]]:
    out, seen = [], set()
    for item in results or []:
        url = clean_url(str(item.get("url", "")))
        key = url or str(item.get("title", "")).strip().lower()
        if not key or key in seen: continue
        seen.add(key)
        out.append({"source_type": source_type, "title": str(item.get("title", "")).strip(), "url": url, "publisher": urlsplit(url).netloc, "published_at": item.get("published_at"), "retrieved_at": datetime.now(timezone.utc).isoformat(), "claim": str(item.get("content", item.get("snippet", ""))).strip()[:1000], "content": str(item.get("content", "")).strip()[:4000], "relevance": float(item.get("score", 0) or 0), "freshness": 0.7, "authority": 0.5})
    return out

