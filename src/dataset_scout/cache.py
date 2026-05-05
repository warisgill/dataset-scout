"""SQLite-backed cache for expensive operations.

Wraps LLM calls (decompose, strategy assessment, embeddings) and other
high-cost intermediate results so re-runs are nearly free. Single
SQLite file at `<ctx.cache_dir>/cache.db`, WAL mode for safe concurrent
readers.

Design notes (after the rubber-duck pass):

- **Eviction is age-based, not LRU.** Tracking `accessed_at` on every
  `get()` would push writes onto the read path and contend the writer
  lock. Instead, when the cache exceeds its byte cap, we evict expired
  entries first, then oldest-by-`created_at` until under cap. Read
  paths never write.
- **Per-namespace TTL defaults, with optional per-call override.**
  Most call sites should pass nothing and inherit the namespace
  default; specialty paths can override.
- **Stable keys are the caller's responsibility.** This module hashes
  whatever bytes the caller hands it; cache-key construction
  (e.g. `intent.stable_hash() + assessor_version`) lives at the call
  site so the namespace boundaries stay clean.
- **Values are bytes.** Callers that store JSON should use
  `cache.set_json(...)` and `cache.get_json(...)` helpers.

Network-free at import time. No external dependencies beyond `sqlite3`
from the standard library.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# ─── namespace defaults ─────────────────────────────────────────────

# TTLs in seconds. None means "no expiry". Per-call `ttl=` overrides.
_NAMESPACE_TTL_DEFAULTS: dict[str, float | None] = {
    "decompose": 7 * 24 * 3600.0,  # 7 days — prompt + brief stable
    "strategy": 7 * 24 * 3600.0,  # 7 days
    "coverage": 7 * 24 * 3600.0,  # 7 days
    "embedding": 30 * 24 * 3600.0,  # 30 days — embedding-model stable
    "hf_meta": 7 * 24 * 3600.0,  # 7 days
    "hf_sample": None,  # tied to (id, revision) — never expire
}

# Sentinel for "use the namespace default" in set(... ttl=...). Distinct
# from None (which means "never expire"), so callers can override either way.
_USE_NAMESPACE_DEFAULT: Any = object()

# Default cap. Overridable via env var. The cache is best-effort: a
# small overshoot between evictions is fine.
DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# Global guard for cross-thread serialisation of the writer connection.
# Per-process; sqlite WAL handles cross-process concurrency on its own.
_WRITE_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _stable_key(parts: object) -> str:
    """Hash arbitrary JSON-serialisable parts into a stable cache key.

    Convenience for call sites: `cache.set("ns", _stable_key((intent_hash, "v2")), ...)`.
    Keys are SHA-256 hex; collision-resistant and fixed-width.
    """
    canonical = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def namespace_default_ttl(namespace: str) -> float | None:
    """Public lookup so call sites can introspect / unit-test policy."""
    return _NAMESPACE_TTL_DEFAULTS.get(namespace)


# ─── cache class ────────────────────────────────────────────────────


class Cache:
    """SQLite-backed KV cache.

    Construct via `Cache.open(cache_dir)` for the standard usage; pass
    a `:memory:` path in tests for an isolated in-process cache.

    The cache is process-shared via the file. Multiple processes can
    read concurrently (WAL); writes are serialised by SQLite's
    file-level lock.
    """

    def __init__(self, db_path: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.db_path = db_path
        self.max_bytes = max_bytes
        # Connection lifecycle: one connection per Cache instance,
        # used for both reads and writes. SQLite is thread-safe enough
        # in DEFAULT mode for our usage; we additionally hold
        # `_WRITE_LOCK` around mutations.
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10.0)
        self._init_schema()

    @classmethod
    def open(cls, cache_dir: Path, *, max_bytes: int | None = None) -> Cache:
        """Open or create the cache at the canonical path."""
        env_max = os.environ.get("DATASET_SCOUT_CACHE_MAX_BYTES")
        if max_bytes is None and env_max:
            try:
                max_bytes = int(env_max)
            except ValueError:
                max_bytes = None
        return cls(cache_dir / "cache.db", max_bytes=max_bytes or DEFAULT_MAX_BYTES)

    def _init_schema(self) -> None:
        with _WRITE_LOCK:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    namespace   TEXT NOT NULL,
                    key         TEXT NOT NULL,
                    value       BLOB NOT NULL,
                    created_at  REAL NOT NULL,
                    expires_at  REAL,
                    size_bytes  INTEGER NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_expires ON entries(expires_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at)"
            )
            self._conn.commit()

    # ─── core ops ───────────────────────────────────────────────

    def get(self, namespace: str, key: str) -> bytes | None:
        """Return the cached value, or None if absent / expired.

        Read-only — does not mutate `accessed_at` (we don't track it;
        eviction is age-based not LRU).
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT value, expires_at FROM entries WHERE namespace=? AND key=?",
            (namespace, key),
        )
        row = cur.fetchone()
        if row is None:
            return None
        value, expires_at = row
        if expires_at is not None and expires_at <= _now():
            return None
        return bytes(value)

    def set(
        self,
        namespace: str,
        key: str,
        value: bytes,
        *,
        ttl: float | None = _USE_NAMESPACE_DEFAULT,
    ) -> None:
        """Insert / replace a cache entry.

        `ttl` semantics:
            - omitted (the default): use the namespace default.
            - explicit float: that many seconds until expiry.
            - explicit None: never expire.
        """
        if ttl is _USE_NAMESPACE_DEFAULT:
            ttl_resolved: float | None = _NAMESPACE_TTL_DEFAULTS.get(namespace)
        else:
            ttl_resolved = ttl
        now = _now()
        expires_at = (now + ttl_resolved) if ttl_resolved is not None else None
        size = len(value)
        with _WRITE_LOCK:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO entries
                    (namespace, key, value, created_at, expires_at, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (namespace, key, value, now, expires_at, size),
            )
            self._conn.commit()
        # Eviction outside the write lock to keep that critical
        # section short. evict() takes the lock itself.
        self._maybe_evict()

    def get_json(self, namespace: str, key: str) -> Any | None:
        raw = self.get(namespace, key)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))

    def set_json(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        ttl: float | None = _USE_NAMESPACE_DEFAULT,
    ) -> None:
        encoded = json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")
        self.set(namespace, key, encoded, ttl=ttl)

    def delete(self, namespace: str, key: str) -> None:
        with _WRITE_LOCK:
            self._conn.execute(
                "DELETE FROM entries WHERE namespace=? AND key=?",
                (namespace, key),
            )
            self._conn.commit()

    # ─── maintenance ────────────────────────────────────────────

    def info(self) -> dict[str, Any]:
        """Summary stats: total bytes, entry count, per-namespace counts."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM entries")
        count, total_bytes = cur.fetchone()
        cur.execute(
            "SELECT namespace, COUNT(*), COALESCE(SUM(size_bytes), 0) "
            "FROM entries GROUP BY namespace ORDER BY namespace"
        )
        per_ns = [
            {"namespace": ns, "entries": int(c), "bytes": int(b)}
            for ns, c, b in cur.fetchall()
        ]
        return {
            "db_path": str(self.db_path),
            "max_bytes": self.max_bytes,
            "total_bytes": int(total_bytes or 0),
            "total_entries": int(count or 0),
            "by_namespace": per_ns,
        }

    def prune(self) -> int:
        """Remove expired entries. Returns count removed."""
        with _WRITE_LOCK:
            cur = self._conn.execute(
                "DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (_now(),),
            )
            removed = cur.rowcount
            self._conn.commit()
        return int(removed or 0)

    def clear(self, namespace: str | None = None) -> int:
        """Remove all entries (optionally only within one namespace)."""
        with _WRITE_LOCK:
            if namespace is None:
                cur = self._conn.execute("DELETE FROM entries")
            else:
                cur = self._conn.execute(
                    "DELETE FROM entries WHERE namespace=?", (namespace,)
                )
            removed = cur.rowcount
            self._conn.commit()
        return int(removed or 0)

    def _maybe_evict(self) -> None:
        """If the cache exceeds `max_bytes`, evict expired then oldest.

        Best-effort: a small overshoot between calls is acceptable.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM entries")
        total = int(cur.fetchone()[0] or 0)
        if total <= self.max_bytes:
            return
        # First pass: drop expired entries.
        self.prune()
        cur.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM entries")
        total = int(cur.fetchone()[0] or 0)
        if total <= self.max_bytes:
            return
        # Second pass: drop oldest-by-created_at until under cap.
        with _WRITE_LOCK:
            cur = self._conn.execute(
                "SELECT namespace, key, size_bytes FROM entries ORDER BY created_at ASC"
            )
            rows = cur.fetchall()
            for ns, k, sz in rows:
                self._conn.execute(
                    "DELETE FROM entries WHERE namespace=? AND key=?",
                    (ns, k),
                )
                total -= int(sz)
                if total <= self.max_bytes:
                    break
            self._conn.commit()

    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ─── module-level convenience for call sites ────────────────────────


@contextmanager
def open_cache(cache_dir: Path, *, max_bytes: int | None = None) -> Iterator[Cache]:
    """Context-manager wrapper. Most call sites prefer `with open_cache(...)`."""
    cache = Cache.open(cache_dir, max_bytes=max_bytes)
    try:
        yield cache
    finally:
        cache.close()


__all__ = [
    "DEFAULT_MAX_BYTES",
    "Cache",
    "_stable_key",
    "namespace_default_ttl",
    "open_cache",
]
