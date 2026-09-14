"""Tavily live search with a small process-local cache."""
import os
import time
import threading
from typing import Any, Dict

from tavily import TavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_CACHE_TTL = max(5, int(os.getenv("TAVILY_CACHE_TTL_SECONDS", "45")))
TAVILY_CACHE_MAX = max(20, int(os.getenv("TAVILY_CACHE_MAX", "256")))

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
_cache: Dict[str, Any] = {}
_lock = threading.RLock()


def _cache_key(query: str, max_results: int) -> str:
    return f"{query.strip().lower()}::{max_results}"


def tavily_live_search(query: str, max_results: int = 3):
    """
    Live web search for current questions. Results are cached briefly to avoid
    paying for duplicate tool calls during one multi-turn agent interaction.
    """
    if not tavily_client:
        return {
            "ok": False,
            "source": "tavily",
            "error": "TAVILY_API_KEY missing",
            "answer": None,
            "results": []
        }

    query = (query or "").strip()
    max_results = max(1, min(int(max_results or 3), 8))
    key = _cache_key(query, max_results)
    now = time.time()

    with _lock:
        item = _cache.get(key)
        if item and now - item[0] < TAVILY_CACHE_TTL:
            return item[1]

    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
                "published_at": item.get("published_at"),
            })

        result = {
            "ok": True,
            "source": "tavily",
            "error": None,
            "answer": response.get("answer"),
            "results": results
        }
        with _lock:
            _cache[key] = (now, result)
            if len(_cache) > TAVILY_CACHE_MAX:
                oldest = sorted(_cache.items(), key=lambda pair: pair[1][0])[:max(1, TAVILY_CACHE_MAX // 8)]
                for old_key, _ in oldest:
                    _cache.pop(old_key, None)
        return result

    except Exception as e:
        return {
            "ok": False,
            "source": "tavily",
            "error": str(e),
            "answer": None,
            "results": []
        }
