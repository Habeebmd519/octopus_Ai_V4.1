import os
import requests
from typing import Any, Dict, List
from .evidence import normalize_source, rank_sources, consensus

class WebResearch:
    def __init__(self, tavily_client=None):
        self.tavily = tavily_client
        self.google_key = os.getenv("GOOGLE_SEARCH_API_KEY", "")
        self.google_cx = os.getenv("GOOGLE_SEARCH_CX", "")
        self.timeout = float(os.getenv("V5_WEB_TIMEOUT", "8"))

    def google(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not (self.google_key and self.google_cx):
            return []
        try:
            r = requests.get("https://www.googleapis.com/customsearch/v1", params={
                "key": self.google_key, "cx": self.google_cx, "q": query, "num": min(10, limit),
            }, timeout=self.timeout)
            r.raise_for_status()
            return [normalize_source({
                "title": x.get("title"), "url": x.get("link"), "content": x.get("snippet"), "score": 0.65,
            }, "google") for x in r.json().get("items", [])]
        except Exception:
            return []

    def tavily_search(self, query: str, limit: int = 5, deep: bool = False) -> List[Dict[str, Any]]:
        if not self.tavily:
            return []
        try:
            response = self.tavily.search(
                query=query,
                search_depth="advanced" if deep else "basic",
                max_results=min(10, limit),
                include_answer=False,
                include_raw_content="markdown" if deep else False,
                include_usage=True,
            )
            return [normalize_source(x, "tavily") for x in response.get("results", [])]
        except Exception:
            return []

    def research(self, query: str, limit: int = 6, deep: bool = False) -> Dict[str, Any]:
        # Run both indexes when configured. Deduplicate by canonical URL.
        tavily = self.tavily_search(query, limit=limit, deep=deep)
        google = self.google(query, limit=limit)
        merged = {}
        for x in tavily + google:
            url = x.get("url") or (x.get("title", "") + x.get("content", ""))
            merged[url] = x
        ranked = rank_sources(query, merged.values(), limit=limit)
        return {
            "ok": bool(ranked), "query": query, "results": ranked,
            "consensus": consensus(ranked),
            "providers": {"tavily": bool(tavily), "google": bool(google)},
        }
