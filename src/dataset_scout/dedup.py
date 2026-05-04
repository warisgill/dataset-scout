"""MinHash-based dedup + leakage-aware splitter (M4b).

Group rows by near-duplicate similarity using MinHash + LSH, then
assign each group to exactly one train/val/test split. This is the
upgrade that flips `recipe.lock.yaml` `audit_readiness` from
`preview` to `ready`: hash-mod splits could put near-duplicate rows
in different splits, leaking eval signal into training. With this
splitter, every group of similar rows lives on one side of the line.

Tunables:
- `num_perm=128` — standard MinHash signature size; trades accuracy
  for memory.
- `threshold=0.8` — Jaccard-similarity threshold for considering two
  rows "near-duplicates." Conservative; tune lower if your domain has
  high paraphrase rates.
- `shingle_size=5` — character n-gram size for tokenisation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataset_scout.core import NormalizedRecord


# Module-level constants for tunable behaviour. Bump
# DEDUP_VERSION when changing them so the lockfile reflects which
# parameters were used.
DEDUP_VERSION = "1"
DEFAULT_NUM_PERM = 128
DEFAULT_LSH_THRESHOLD = 0.8
DEFAULT_SHINGLE_SIZE = 5


def _shingles(text: str, n: int = DEFAULT_SHINGLE_SIZE) -> Iterable[bytes]:
    """Lowercase character n-grams. Empty / very short text yields its
    whole content as one shingle so MinHash isn't completely empty."""
    text = text.lower()
    if len(text) < n:
        if text:
            yield text.encode("utf-8")
        return
    for i in range(len(text) - n + 1):
        yield text[i : i + n].encode("utf-8")


def _row_signature(
    record: NormalizedRecord,
    leakage_keys: list[str],
    *,
    num_perm: int,
    shingle_size: int,
) -> Any:
    """Build a MinHash signature for one record using the configured
    leakage_keys columns. Always includes record.text since the
    detector's primary surface is the text field."""
    from datasketch import MinHash  # type: ignore[import-untyped]

    parts: list[str] = []
    if record.text:
        parts.append(record.text)
    for key in leakage_keys:
        if key == "text":
            continue  # already added
        v = record.extras.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif v is not None:
            parts.append(str(v))
    text = " | ".join(parts)
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text, n=shingle_size):
        m.update(shingle)
    return m


def _build_clusters(
    records: list[NormalizedRecord],
    leakage_keys: list[str],
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    threshold: float = DEFAULT_LSH_THRESHOLD,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> list[int]:
    """Return a parallel list of cluster IDs (one per record).

    Records with no near-duplicates form singleton clusters with their
    own cluster id (their index).
    """
    from datasketch import MinHashLSH

    if not records:
        return []

    sigs = [
        _row_signature(r, leakage_keys, num_perm=num_perm, shingle_size=shingle_size)
        for r in records
    ]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, sig in enumerate(sigs):
        lsh.insert(str(i), sig)

    # Union-find for transitive grouping. Two near-duplicate pairs
    # (A~B and B~C) should land in one cluster even if A and C aren't
    # individually similar enough.
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, sig in enumerate(sigs):
        for j_str in lsh.query(sig):
            j = int(j_str)
            if i != j:
                union(i, j)

    return [find(i) for i in range(len(records))]


def leakage_aware_split(
    records: list[NormalizedRecord],
    proportions: dict[str, float],
    *,
    seed: int,
    leakage_keys: list[str],
    num_perm: int = DEFAULT_NUM_PERM,
    threshold: float = DEFAULT_LSH_THRESHOLD,
) -> tuple[dict[str, list[NormalizedRecord]], dict[str, Any]]:
    """Split records into train/val/test groups, never separating
    near-duplicates across splits.

    Returns (splits, stats). `stats` carries:
      - method: "minhash_lsh"
      - num_perm, threshold, shingle_size: the parameters used
      - clusters_total: count of distinct clusters
      - clusters_singleton: clusters with exactly one record
      - clusters_largest: size of the biggest cluster
    """
    if not records:
        return ({"train": [], "val": [], "test": []}, _empty_stats())

    cluster_ids = _build_clusters(
        records,
        leakage_keys,
        num_perm=num_perm,
        threshold=threshold,
    )

    # Group by cluster.
    cluster_to_indices: dict[int, list[int]] = {}
    for i, cid in enumerate(cluster_ids):
        cluster_to_indices.setdefault(cid, []).append(i)

    # Cumulative thresholds for split assignment.
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for name in ("train", "val", "test"):
        running += proportions[name]
        cumulative.append((name, running))

    splits: dict[str, list[NormalizedRecord]] = {"train": [], "val": [], "test": []}

    # For each cluster, derive a stable bucket from (seed, cluster id).
    for cid, indices in cluster_to_indices.items():
        h = hashlib.sha256()
        h.update(str(seed).encode("utf-8"))
        h.update(b":")
        h.update(str(cid).encode("utf-8"))
        fraction = int(h.hexdigest()[:16], 16) / 16**16
        bucket = cumulative[-1][0]
        for name, threshold_value in cumulative:
            if fraction < threshold_value:
                bucket = name
                break
        for idx in indices:
            splits[bucket].append(records[idx])

    sizes = [len(v) for v in cluster_to_indices.values()]
    stats = {
        "method": "minhash_lsh",
        "num_perm": num_perm,
        "threshold": threshold,
        "shingle_size": DEFAULT_SHINGLE_SIZE,
        "dedup_version": DEDUP_VERSION,
        "clusters_total": len(cluster_to_indices),
        "clusters_singleton": sum(1 for s in sizes if s == 1),
        "clusters_largest": max(sizes) if sizes else 0,
        "rows_in_dup_clusters": sum(s for s in sizes if s > 1),
    }
    return splits, stats


def _empty_stats() -> dict[str, Any]:
    return {
        "method": "minhash_lsh",
        "num_perm": DEFAULT_NUM_PERM,
        "threshold": DEFAULT_LSH_THRESHOLD,
        "shingle_size": DEFAULT_SHINGLE_SIZE,
        "dedup_version": DEDUP_VERSION,
        "clusters_total": 0,
        "clusters_singleton": 0,
        "clusters_largest": 0,
        "rows_in_dup_clusters": 0,
    }
