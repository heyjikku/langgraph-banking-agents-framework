"""
Deterministic utilities — replaces `random` everywhere in the agent layer.

Why: agents run concurrently via asyncio.gather. Global `random.seed()` calls
from one agent contaminate another's draws, and a fraud decision that flips
between Allow and Flag on retry is unacceptable in a bank. Every derived
value must be a pure function of its inputs.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any


def stable_hash(*parts: Any) -> int:
    """64-bit integer hash of the JSON-serialised parts. Same input → same output, across processes."""
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return int(hashlib.blake2b(payload.encode(), digest_size=8).hexdigest(), 16)


def unit(*parts: Any) -> float:
    """Deterministic float in [0, 1)."""
    return (stable_hash(*parts) % 10_000_000) / 10_000_000


def between(lo: float, hi: float, *parts: Any) -> float:
    """Deterministic float in [lo, hi)."""
    return lo + (hi - lo) * unit(*parts)


def pick(options: list, *parts: Any):
    """Deterministic choice from a list."""
    if not options:
        return None
    return options[stable_hash(*parts) % len(options)]


def data_fingerprint(customer_data: dict) -> str:
    """Short hash of the customer payload — used as cache key component so stale data never serves."""
    return hashlib.blake2b(
        json.dumps(customer_data, sort_keys=True, default=str).encode(), digest_size=6
    ).hexdigest()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
