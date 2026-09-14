"""
Octopus AI V4.2 performance primitives.

No external dependencies. These helpers keep the Flask worker fast under
repeated Puter tool calls by providing thread-safe TTL/LRU caches, lightweight
metrics, and normalized cache keys.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple


def stable_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


class TTLCache:
    """Small thread-safe TTL + LRU cache."""

    def __init__(self, maxsize: int = 512, ttl: float = 60.0):
        self.maxsize = max(1, int(maxsize))
        self.ttl = max(0.1, float(ttl))
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return default
            expires, value = item
            if expires <= now:
                self._data.pop(key, None)
                return default
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> Any:
        expires = time.monotonic() + (self.ttl if ttl is None else max(0.1, float(ttl)))
        with self._lock:
            self._data[key] = (expires, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
        return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> Dict[str, int]:
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (expires, _) in self._data.items() if expires <= now]
            for k in expired:
                self._data.pop(k, None)
            return {"size": len(self._data), "maxsize": self.maxsize}


class Metrics:
    """Process-local metrics; intentionally lightweight and privacy-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests = 0
        self.errors = 0
        self.tool_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.timings: Dict[str, Dict[str, float]] = {}
        self.intent_counts: Dict[str, int] = {}

    def request(self, ok: bool = True) -> None:
        with self._lock:
            self.requests += 1
            if not ok:
                self.errors += 1

    def tool(self) -> None:
        with self._lock:
            self.tool_calls += 1

    def cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def intent(self, name: str) -> None:
        with self._lock:
            self.intent_counts[name] = self.intent_counts.get(name, 0) + 1

    @contextmanager
    def timer(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            with self._lock:
                bucket = self.timings.setdefault(name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
                bucket["count"] += 1
                bucket["total_ms"] += elapsed
                bucket["max_ms"] = max(bucket["max_ms"], elapsed)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            timings = {}
            for name, item in self.timings.items():
                count = item["count"]
                timings[name] = {
                    "count": count,
                    "avgMs": round(item["total_ms"] / count, 2) if count else 0,
                    "maxMs": round(item["max_ms"], 2),
                }
            return {
                "uptimeSeconds": round(max(0, time.time() - self.started_at), 1),
                "requests": self.requests,
                "errors": self.errors,
                "toolCalls": self.tool_calls,
                "cacheHits": self.cache_hits,
                "cacheMisses": self.cache_misses,
                "cacheHitRate": round(self.cache_hits / max(1, self.cache_hits + self.cache_misses), 4),
                "topIntents": sorted(
                    [{"intent": k, "count": v} for k, v in self.intent_counts.items()],
                    key=lambda x: x["count"],
                    reverse=True,
                )[:20],
                "timings": timings,
            }


def cached_call(
    cache: TTLCache,
    metrics: Metrics,
    key: str,
    loader: Callable[[], Any],
    ttl: Optional[float] = None,
) -> Any:
    value = cache.get(key, None)
    if value is not None:
        metrics.cache(True)
        return value
    metrics.cache(False)
    value = loader()
    return cache.set(key, value, ttl=ttl)
