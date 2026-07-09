"""SmartKB 查询缓存。

默认使用进程内 TTL 缓存；配置 Redis 后自动切换为 Redis 缓存。
"""

from __future__ import annotations

import hashlib
import json
import time
from threading import RLock
from typing import Any

from config import CACHE_BACKEND, COLLECTION_NAME, QUERY_CACHE_TTL_SECONDS, REDIS_URL


class InMemoryTTLCacheStore:
    """轻量内存 TTL 缓存，用于本地开发和离线测试。"""

    def __init__(self, reason: str = ""):
        self.reason = reason
        self.rows: dict[str, tuple[float, str]] = {}
        self.lock = RLock()

    def get_json(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            self._purge_expired()
            row = self.rows.get(key)
            if not row:
                return None
            expires_at, payload = row
            if expires_at < time.time():
                self.rows.pop(key, None)
                return None
            return json.loads(payload)

    def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int = QUERY_CACHE_TTL_SECONDS,
    ) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self.lock:
            self.rows[key] = (time.time() + ttl_seconds, payload)

    def delete_prefix(self, prefix: str) -> int:
        with self.lock:
            keys = [key for key in self.rows if key.startswith(prefix)]
            for key in keys:
                self.rows.pop(key, None)
            return len(keys)

    def stats(self) -> dict[str, Any]:
        with self.lock:
            self._purge_expired()
            return {
                "backend": "memory",
                "keys": len(self.rows),
                "fallback_reason": self.reason,
            }

    def _purge_expired(self) -> None:
        now = time.time()
        for key, (expires_at, _) in list(self.rows.items()):
            if expires_at < now:
                self.rows.pop(key, None)


class RedisTTLCacheStore:
    """Redis TTL 缓存。"""

    def __init__(self, redis_url: str = REDIS_URL, prefix: str = "smartkb:"):
        if not redis_url:
            raise RuntimeError("REDIS_URL 未配置")
        try:
            import redis
        except Exception as exc:
            raise RuntimeError(f"redis package unavailable: {exc}") from exc

        self.prefix = prefix
        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.8,
        )
        self.client.ping()

    def get_json(self, key: str) -> dict[str, Any] | None:
        payload = self.client.get(self._key(key))
        return json.loads(payload) if payload else None

    def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int = QUERY_CACHE_TTL_SECONDS,
    ) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        self.client.setex(self._key(key), ttl_seconds, payload)

    def delete_prefix(self, prefix: str) -> int:
        keys = list(self.client.scan_iter(self._key(prefix) + "*", count=200))
        if keys:
            self.client.delete(*keys)
        return len(keys)

    def stats(self) -> dict[str, Any]:
        keys = list(self.client.scan_iter(self._key("*"), count=200))
        return {"backend": "redis", "keys": len(keys)}

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"


def make_query_cache_key(query: str, top_k: int) -> str:
    payload = json.dumps(
        {
            "collection": COLLECTION_NAME,
            "query": " ".join(query.strip().split()),
            "top_k": int(top_k),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"query:{digest}"


_cache_store: InMemoryTTLCacheStore | RedisTTLCacheStore | None = None


def get_cache_store() -> InMemoryTTLCacheStore | RedisTTLCacheStore:
    global _cache_store
    if _cache_store is not None:
        return _cache_store

    if CACHE_BACKEND == "memory" or (CACHE_BACKEND == "auto" and not REDIS_URL):
        _cache_store = InMemoryTTLCacheStore()
        return _cache_store

    try:
        _cache_store = RedisTTLCacheStore()
    except Exception as exc:
        _cache_store = InMemoryTTLCacheStore(str(exc)[:200])
    return _cache_store
