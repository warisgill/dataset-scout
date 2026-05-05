"""Unit tests for `dataset_scout.cache`."""

from __future__ import annotations

import time

import pytest

from dataset_scout.cache import (
    DEFAULT_MAX_BYTES,
    Cache,
    _stable_key,
    namespace_default_ttl,
    open_cache,
)

pytestmark = pytest.mark.unit


def _fresh_cache(tmp_path, max_bytes: int = DEFAULT_MAX_BYTES) -> Cache:
    return Cache(tmp_path / "cache.db", max_bytes=max_bytes)


def test_round_trip_bytes(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("decompose", "k1", b"hello", ttl=None)
        assert c.get("decompose", "k1") == b"hello"
    finally:
        c.close()


def test_round_trip_json(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set_json("strategy", "k1", {"x": 1, "y": [2, 3]}, ttl=None)
        assert c.get_json("strategy", "k1") == {"x": 1, "y": [2, 3]}
    finally:
        c.close()


def test_miss_returns_none(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        assert c.get("decompose", "absent") is None
        assert c.get_json("strategy", "absent") is None
    finally:
        c.close()


def test_namespaces_isolated(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("decompose", "shared_key", b"a", ttl=None)
        c.set("strategy", "shared_key", b"b", ttl=None)
        assert c.get("decompose", "shared_key") == b"a"
        assert c.get("strategy", "shared_key") == b"b"
    finally:
        c.close()


def test_explicit_ttl_expires(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("decompose", "k", b"v", ttl=0.05)
        assert c.get("decompose", "k") == b"v"
        time.sleep(0.1)
        assert c.get("decompose", "k") is None
    finally:
        c.close()


def test_namespace_default_ttl_used(tmp_path, monkeypatch):
    """Omitting ttl uses the namespace default."""
    # decompose default is 7 days; can't wait for that, but we can
    # monkeypatch _NAMESPACE_TTL_DEFAULTS to a tiny value and verify.
    from dataset_scout import cache as cache_mod

    monkeypatch.setitem(cache_mod._NAMESPACE_TTL_DEFAULTS, "decompose", 0.05)
    c = _fresh_cache(tmp_path)
    try:
        c.set("decompose", "k", b"v")  # no ttl passed
        assert c.get("decompose", "k") == b"v"
        time.sleep(0.1)
        assert c.get("decompose", "k") is None
    finally:
        c.close()


def test_explicit_ttl_none_never_expires(tmp_path, monkeypatch):
    """Explicit ttl=None overrides the namespace default to no-expiry."""
    from dataset_scout import cache as cache_mod

    # Even with a tiny default, explicit None must override.
    monkeypatch.setitem(cache_mod._NAMESPACE_TTL_DEFAULTS, "decompose", 0.05)
    c = _fresh_cache(tmp_path)
    try:
        c.set("decompose", "k", b"v", ttl=None)
        time.sleep(0.1)
        assert c.get("decompose", "k") == b"v"
    finally:
        c.close()


def test_delete(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("ns", "k", b"v", ttl=None)
        c.delete("ns", "k")
        assert c.get("ns", "k") is None
    finally:
        c.close()


def test_clear_namespace(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("a", "1", b"x", ttl=None)
        c.set("a", "2", b"x", ttl=None)
        c.set("b", "1", b"y", ttl=None)
        removed = c.clear("a")
        assert removed == 2
        assert c.get("a", "1") is None
        assert c.get("b", "1") == b"y"
    finally:
        c.close()


def test_clear_all(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("a", "1", b"x", ttl=None)
        c.set("b", "1", b"y", ttl=None)
        removed = c.clear()
        assert removed == 2
        assert c.info()["total_entries"] == 0
    finally:
        c.close()


def test_prune_only_expired(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("ns", "live", b"x", ttl=None)
        c.set("ns", "dead", b"y", ttl=0.05)
        time.sleep(0.1)
        removed = c.prune()
        assert removed == 1
        assert c.get("ns", "live") == b"x"
    finally:
        c.close()


def test_info(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("decompose", "k1", b"abc", ttl=None)
        c.set("decompose", "k2", b"abcd", ttl=None)
        c.set("strategy", "k1", b"x", ttl=None)
        info = c.info()
        assert info["total_entries"] == 3
        assert info["total_bytes"] == 8  # 3+4+1
        ns_map = {n["namespace"]: n for n in info["by_namespace"]}
        assert ns_map["decompose"]["entries"] == 2
        assert ns_map["decompose"]["bytes"] == 7
        assert ns_map["strategy"]["entries"] == 1
    finally:
        c.close()


def test_eviction_age_based(tmp_path):
    """When over byte cap, oldest entries are evicted first."""
    # Use a tiny cap so we exercise eviction in ms.
    c = Cache(tmp_path / "cache.db", max_bytes=10)
    try:
        c.set("ns", "old", b"123456", ttl=None)  # 6 bytes
        time.sleep(0.01)
        c.set("ns", "mid", b"123", ttl=None)  # 3 bytes; total 9 — under cap
        time.sleep(0.01)
        c.set("ns", "new", b"12345", ttl=None)  # 5 bytes; total 14 — over cap
        # _maybe_evict ran on the last set. "old" is oldest, should go first.
        assert c.get("ns", "old") is None
        assert c.get("ns", "mid") == b"123"
        assert c.get("ns", "new") == b"12345"
    finally:
        c.close()


def test_overwrite_replaces(tmp_path):
    c = _fresh_cache(tmp_path)
    try:
        c.set("ns", "k", b"v1", ttl=None)
        c.set("ns", "k", b"v2", ttl=None)
        assert c.get("ns", "k") == b"v2"
        assert c.info()["total_entries"] == 1
    finally:
        c.close()


def test_open_cache_context_manager(tmp_path):
    with open_cache(tmp_path) as c:
        c.set("ns", "k", b"v", ttl=None)
        assert c.get("ns", "k") == b"v"


def test_stable_key_deterministic():
    a = _stable_key(("intent_hash_xyz", "v2"))
    b = _stable_key(("intent_hash_xyz", "v2"))
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_stable_key_different_inputs():
    assert _stable_key(("a",)) != _stable_key(("b",))


def test_namespace_default_ttl_lookup():
    assert namespace_default_ttl("decompose") is not None
    assert namespace_default_ttl("hf_sample") is None  # never expire
    assert namespace_default_ttl("nonexistent") is None


def test_persistence_across_open(tmp_path):
    """A second Cache opening the same file sees prior entries."""
    c1 = _fresh_cache(tmp_path)
    c1.set("ns", "k", b"v", ttl=None)
    c1.close()
    c2 = _fresh_cache(tmp_path)
    try:
        assert c2.get("ns", "k") == b"v"
    finally:
        c2.close()
