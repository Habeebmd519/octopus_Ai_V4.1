import os
from tavily import TavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


def tavily_live_search(query: str, max_results: int = 3, deep: bool = False, include_raw_content: bool = False):
    """
    Tavily live search for Octopus AI V3.
    Use only for latest/current/nearby/timing questions.
    """

    if not tavily_client:
        return {
            "ok": False,
            "source": "tavily",
            "error": "TAVILY_API_KEY missing",
            "answer": None,
            "results": []
        }

    try:
        response = tavily_client.search(
            query=query,
            search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "advanced" if deep else "basic"),
            max_results=max_results,
            include_answer=False,
            include_raw_content=("markdown" if include_raw_content else False)
        )

        results = []

        for item in response.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
                "raw_content": item.get("raw_content")
            })

        return {
            "ok": True,
            "source": "tavily",
            "error": None,
            "answer": None,
            "results": results
        }

    except Exception as e:
        return {
            "ok": False,
            "source": "tavily",
            "error": str(e),
            "answer": None,
            "results": []
        }