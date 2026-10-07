"""
Agent Memory (v2.1, post-audit)

Fixes vs v2.0:
  • Expired entries are evicted on read AND on a periodic sweep (no unbounded growth)
  • ConversationMemory is Redis-backed when REDIS_URL is set — required because the
    container runs 2 uvicorn workers; in-process history was randomly lost.
  • When Redis is absent, a loud warning is logged on multi-worker start.
"""
from __future__ import annotations
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.memory")


class MemoryStore:
    def __init__(self, default_ttl: int = 300, sweep_every: int = 200):
        self._store: Dict[str, Dict] = {}
        self._default_ttl = default_ttl
        self._ops = 0
        self._sweep_every = sweep_every
        self._redis = None
        self._connect()

    def _connect(self):
        url = os.getenv("REDIS_URL", "")
        if not url:
            if int(os.getenv("WORKERS", "1")) > 1:
                logger.warning("WORKERS>1 but REDIS_URL unset — cache and chat history will NOT be shared across workers")
            return
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2)
            logger.info("Memory backend: Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}); using in-process memory")

    def _sweep(self):
        self._ops += 1
        if self._ops % self._sweep_every:
            return
        now = time.time()
        for k in [k for k, v in self._store.items() if v["expires"] != -1 and v["expires"] < now]:
            self._store.pop(k, None)

    async def get(self, key: str) -> Optional[Any]:
        self._sweep()
        if self._redis:
            try:
                v = await self._redis.get(f"c360:{key}")
                return json.loads(v) if v else None
            except Exception:
                pass
        e = self._store.get(key)
        if not e:
            return None
        if e["expires"] != -1 and time.time() >= e["expires"]:
            self._store.pop(key, None)
            return None
        return e["value"]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self._default_ttl
        if self._redis:
            try:
                await self._redis.setex(f"c360:{key}", ttl, json.dumps(value, default=str))
                return
            except Exception:
                pass
        self._store[key] = {"value": value, "expires": time.time() + ttl if ttl > 0 else -1}

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        if self._redis:
            try:
                await self._redis.delete(f"c360:{key}")
            except Exception:
                pass

    async def delete_prefix(self, prefix: str) -> int:
        n = 0
        for k in [k for k in self._store if k.startswith(prefix)]:
            self._store.pop(k, None); n += 1
        if self._redis:
            try:
                async for k in self._redis.scan_iter(f"c360:{prefix}*"):
                    await self._redis.delete(k); n += 1
            except Exception:
                pass
        return n

    async def save_agent_result(self, customer_id: str, agent_name: str, result: Dict) -> None:
        ctx = await self.get(f"ctx:{customer_id}") or {}
        ctx[agent_name] = {"result": result, "timestamp": time.time()}
        await self.set(f"ctx:{customer_id}", ctx, ttl=3600)

    async def clear_customer(self, customer_id: str) -> int:
        return await self.delete_prefix(f"pipeline:{customer_id}:") + await self.delete_prefix(f"ctx:{customer_id}")


class ConversationMemory:
    """Sliding-window chat history. Redis-backed when available so all workers share it."""

    def __init__(self, store: MemoryStore, max_turns: int = 20, ttl: int = 3600):
        self._store = store
        self._local: Dict[str, List] = defaultdict(list)
        self._max = max_turns
        self._ttl = ttl

    async def add(self, customer_id: str, role: str, content: str) -> None:
        h = await self.get(customer_id, last_n=self._max)
        h.append({"role": role, "content": content, "ts": time.time()})
        h = h[-self._max:]
        if self._store._redis:
            await self._store.set(f"chat:{customer_id}", h, ttl=self._ttl)
        else:
            self._local[customer_id] = h

    async def get(self, customer_id: str, last_n: int = 10) -> List[Dict]:
        if self._store._redis:
            return (await self._store.get(f"chat:{customer_id}") or [])[-last_n:]
        return list(self._local[customer_id][-last_n:])

    async def clear(self, customer_id: str) -> None:
        self._local.pop(customer_id, None)
        await self._store.delete(f"chat:{customer_id}")


memory = MemoryStore(default_ttl=300)
conversation_memory = ConversationMemory(memory, max_turns=20)
