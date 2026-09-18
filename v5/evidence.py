import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List

TRUST_TIERS = {
    1: {"gov.in", "nic.in", "kerala.gov.in", "ac.in", "edu.in"},
    2: {"who.int", "un.org", "unesco.org", "wikipedia.org"},
    3: {"reuters.com", "bbc.com", "thehindu.com", "indianexpress.com", "ndtv.com"},
}


def domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url or "", re.I)
    return (m.group(1).lower() if m else "").split(":")[0]


def trust_score(url: str, title: str = "") -> float:
    d = domain(url)
    if not d:
        return 0.20
    for tier, domains in TRUST_TIERS.items():
        if any(d == x or d.endswith("." + x) for x in domains):
            return {1: 1.00, 2: 0.92, 3: 0.82}[tier]
    if d.endswith(".gov") or ".gov." in d:
        return 0.98
    return 0.55


def lexical_similarity(query: str, text: str) -> float:
    q = Counter(re.findall(r"[a-z0-9]+", (query or "").lower()))
    t = Counter(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if not q or not t:
        return 0.0
    overlap = sum(min(q[k], t[k]) for k in q.keys() & t.keys())
    return overlap / max(1, sum(q.values()))


def normalize_source(item: Dict[str, Any], source: str) -> Dict[str, Any]:
    url = item.get("url") or ""
    title = item.get("title") or ""
    content = item.get("content") or item.get("snippet") or ""
    relevance = float(item.get("score") or 0.0)
    return {
        "source": source,
        "url": url,
        "title": title,
        "content": content,
        "relevance": max(0.0, min(1.0, relevance)),
        "trust": trust_score(url, title),
    }


def rank_sources(query: str, sources: Iterable[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
    out = []
    for s in sources:
        x = dict(s)
        lex = lexical_similarity(query, f"{x.get('title','')} {x.get('content','')}")
        x["lexical"] = lex
        # Evidence score intentionally values trust but never lets a trusted irrelevant page win.
        x["evidence_score"] = round(0.50 * float(x.get("relevance", 0)) + 0.25 * lex + 0.25 * float(x.get("trust", 0.5)), 4)
        out.append(x)
    out.sort(key=lambda x: x["evidence_score"], reverse=True)
    return out[:limit]


def consensus(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [s for s in sources if s.get("content")]
    high = [s for s in usable if float(s.get("trust", 0)) >= 0.8]
    return {
        "sourceCount": len(usable),
        "highTrustCount": len(high),
        "confidence": "high" if len(high) >= 2 else ("medium" if len(usable) >= 2 else "low"),
    }
