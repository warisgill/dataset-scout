"""Kaggle Datasets source plugin.

Reads from Kaggle's public REST API via `httpx` (no `kaggle` Python
package dependency — that package has import-time auth side-effects
we'd rather avoid).

Auth: HTTP Basic with `(KAGGLE_USERNAME, KAGGLE_KEY)`. Either:
  - `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars (preferred), or
  - `~/.kaggle/kaggle.json` containing `{"username": ..., "key": ...}`.

Scope (v1, by design):
  - `search` (per-direction queries, mirroring HuggingFaceSource)
  - `fetch_metadata`
  - `card_url`, `terms_check`
  - `stream_sample` / `stream_rows` raise `SourceUnsupportedError`.

Why discovery-only? Kaggle datasets are arbitrary archives — CSV, ZIP,
parquet, sometimes Jupyter notebooks. Generic streaming would either
silently produce nonsense or require per-dataset adapters. The source
honestly reports "I do not stream" so the strategy assessor degrades
to metadata-only assessment for Kaggle candidates and `curate` blocks
materialisation by surfacing a clear failure category.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    DecompositionDirection,
    Intent,
)
from dataset_scout.errors import SourceUnsupportedError
from dataset_scout.licenses import guess_spdx
from dataset_scout.sources.base import Budget, Obligation

if TYPE_CHECKING:
    from dataset_scout.context import ScoutContext


_KAGGLE_API = "https://www.kaggle.com/api/v1"
_KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/{ref}"
_DEFAULT_LIMIT = 20  # per query
_REQUEST_TIMEOUT = 15.0


# ─── credentials ────────────────────────────────────────────────────


def kaggle_credentials(ctx: ScoutContext) -> tuple[str, str] | None:
    """Resolve Kaggle credentials from ctx.api_keys or ~/.kaggle/kaggle.json.

    Returns `(username, key)` or None when no creds are configured.
    The factory uses None to mean "Kaggle disabled — quietly skip."
    """
    username = ctx.api_keys.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    key = ctx.api_keys.get("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    # Fall back to ~/.kaggle/kaggle.json
    home = Path.home() / ".kaggle" / "kaggle.json"
    if home.is_file():
        try:
            data = json.loads(home.read_text(encoding="utf-8"))
            u = data.get("username")
            k = data.get("key")
            if isinstance(u, str) and isinstance(k, str):
                return u, k
        except (OSError, json.JSONDecodeError):
            return None
    return None


# ─── translation ────────────────────────────────────────────────────


def _coerce_kaggle_dt(value: Any) -> datetime | None:
    """Kaggle returns ISO-8601 strings; tolerate already-parsed datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            # Kaggle uses 'Z'-suffixed UTC.
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _ref_from_payload(payload: dict[str, Any]) -> str | None:
    """The dataset's canonical 'owner/slug' identifier.

    Kaggle returns this in different shapes across endpoints:
    - list endpoint: top-level `ref`
    - view endpoint: top-level `ref`
    - some legacy responses: build from `ownerName` + `urlSlug`
    """
    ref = payload.get("ref")
    if isinstance(ref, str) and "/" in ref:
        return ref
    owner = payload.get("ownerName") or payload.get("ownerUser")
    slug = payload.get("urlSlug") or payload.get("datasetSlug")
    if isinstance(owner, str) and isinstance(slug, str):
        return f"{owner}/{slug}"
    return None


def _coerce_tags(value: Any) -> list[str]:
    """Kaggle returns tags as `[{"name": ..., "ref": ...}, ...]` or None."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for tag in value:
        if isinstance(tag, dict):
            name = tag.get("name") or tag.get("ref")
            if isinstance(name, str):
                out.append(name)
        elif isinstance(tag, str):
            out.append(tag)
    return out


def _build_metadata(payload: dict[str, Any]) -> CandidateMetadata:
    """Translate a Kaggle dataset payload into the source-agnostic envelope."""
    ref = _ref_from_payload(payload) or "unknown/unknown"
    description = (
        payload.get("description")
        or payload.get("subtitle")
        or payload.get("title")
        or None
    )
    license_raw_obj = payload.get("licenseName") or payload.get("license") or None
    if isinstance(license_raw_obj, dict):
        license_raw = license_raw_obj.get("name") or license_raw_obj.get("nameNullable")
    else:
        license_raw = license_raw_obj if isinstance(license_raw_obj, str) else None

    bytes_value = payload.get("totalBytes")
    if not isinstance(bytes_value, int):
        bytes_value = None

    downloads = payload.get("downloadCount")
    if not isinstance(downloads, int):
        downloads = None

    likes = payload.get("voteCount")
    if not isinstance(likes, int):
        likes = None

    tags = _coerce_tags(payload.get("tags"))

    extras: dict[str, Any] = {}
    creator = payload.get("creatorName") or payload.get("ownerName")
    if creator:
        extras["creator"] = creator
    usability = payload.get("usabilityRating")
    if isinstance(usability, (int, float)):
        extras["usability_rating"] = float(usability)

    # Kaggle exposes a field set distinct from HF, but we still record
    # whichever yaml-card-style fields the payload happened to declare
    # so probes treat presence/absence consistently.
    card_fields_present = frozenset(
        k
        for k in (
            "license",
            "licenseName",
            "tags",
            "description",
            "subtitle",
            "title",
        )
        if payload.get(k)
    )

    return CandidateMetadata(
        description=description if isinstance(description, str) else None,
        card_url=_KAGGLE_DATASET_URL.format(ref=ref),
        homepage_url=None,
        license_raw=license_raw,
        license_spdx=guess_spdx(license_raw),
        languages_declared=[],  # Kaggle has no per-dataset language declaration
        uploaded_at=_coerce_kaggle_dt(payload.get("creationDate")),
        last_modified=_coerce_kaggle_dt(payload.get("lastUpdated")),
        content_date_range=None,
        rows=None,
        bytes=bytes_value,
        downloads=downloads,
        likes=likes,
        columns=[],
        label_column_guess=None,
        text_column_guess=None,
        card_fields_present=card_fields_present,
        requires_auth=bool(payload.get("isPrivate")),
        gated=False,  # Kaggle's gating model differs; treat as ungated for now
        tags=tags,
        task_categories=[],
        extras=extras,
    )


# ─── query construction ────────────────────────────────────────────


def _build_search_query(intent: Intent) -> str:
    """One Kaggle query string from the original Intent (mirrors HF logic)."""
    if intent.threat_families:
        return " ".join(f.replace("_", " ") for f in intent.threat_families)
    return intent.raw_brief.strip()


def _direction_queries(direction: DecompositionDirection) -> list[str]:
    """Per-keyword queries — mirrors HF source's behaviour for parity.

    Capped at 3 keywords per direction. Falls back to the direction
    name when the LLM returns no keywords.
    """
    if direction.keywords:
        return list(direction.keywords[:3])
    return [direction.name.replace("_", " ")]


# ─── source ─────────────────────────────────────────────────────────


class KaggleSource:
    """Concrete `Source` plugin for Kaggle Datasets (discovery-only).

    Streaming is intentionally not implemented; see module docstring
    for rationale. `stream_sample` and `stream_rows` raise
    `SourceUnsupportedError` so callers can branch explicitly rather
    than treating "no rows" as ambiguous.
    """

    name: str = "kaggle"

    def __init__(
        self,
        *,
        username: str,
        key: str,
        limit: int = _DEFAULT_LIMIT,
        client: httpx.Client | None = None,
    ) -> None:
        self._auth = httpx.BasicAuth(username, key)
        self._limit = limit
        # Tests inject a respx-mounted Client.
        self._client = client or httpx.Client(timeout=_REQUEST_TIMEOUT)
        self._owns_client = client is None

    def search(
        self,
        intent: Intent,
        directions: list[DecompositionDirection],
        *,
        budget: Budget,
    ) -> Iterator[Candidate]:
        """Yield candidates round-robin across the original brief and each direction.

        Same round-robin shape as HuggingFaceSource so a high-recall
        first direction can't starve later directions.
        """
        per_query_results: list[list[Candidate]] = []

        original_query = _build_search_query(intent)
        if original_query:
            per_query_results.append(list(self._search_one(original_query, surfaced_by=[])))

        for direction in directions:
            for query in _direction_queries(direction):
                if query:
                    per_query_results.append(
                        list(self._search_one(query, surfaced_by=[direction.name]))
                    )

        idx = 0
        any_left = True
        while any_left:
            any_left = False
            for results in per_query_results:
                if idx < len(results):
                    yield results[idx]
                    any_left = True
            idx += 1

    def _search_one(self, query: str, *, surfaced_by: list[str]) -> Iterator[Candidate]:
        url = f"{_KAGGLE_API}/datasets/list"
        params: dict[str, str] = {"search": query, "page": "1"}
        try:
            response = self._client.get(url, params=params, auth=self._auth)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            # Network / auth / parse failures should not crash the run;
            # emit zero candidates from this query.
            return
        if not isinstance(payload, list):
            return
        # Kaggle returns up to 20 per page; cap further per-call.
        for item in payload[: self._limit]:
            if not isinstance(item, dict):
                continue
            ref = _ref_from_payload(item)
            if ref is None:
                continue
            yield Candidate(
                source=self.name,
                id=ref,
                revision=None,  # Kaggle datasets don't have HF-style sha pins
                metadata=_build_metadata(item),
                streamable=False,  # explicit: this source does not stream
                surfaced_by=list(surfaced_by),
            )

    def fetch_metadata(self, candidate: Candidate) -> dict[str, Any]:
        """Fetch the full payload for one dataset (`/datasets/view/<owner>/<slug>`)."""
        url = f"{_KAGGLE_API}/datasets/view/{candidate.id}"
        response = self._client.get(url, auth=self._auth)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def stream_sample(
        self,
        candidate: Candidate,
        n: int,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        """Discovery-only source. Always raises SourceUnsupportedError."""
        raise SourceUnsupportedError(
            f"KaggleSource does not stream rows; '{candidate.id}' is "
            "discovery-only. Use the dataset card for sample inspection."
        )
        yield  # pragma: no cover - keeps mypy seeing this as a generator

    def stream_rows(
        self,
        candidate: Candidate,
        *,
        config: str | None = None,
        split: str = "train",
        take: int | None = None,
        seed: int = 42,
    ) -> Iterator[dict[str, Any]]:
        """Discovery-only source. Always raises SourceUnsupportedError."""
        raise SourceUnsupportedError(
            f"KaggleSource does not stream rows; '{candidate.id}' is "
            "discovery-only. Materialise it manually via the Kaggle CLI."
        )
        yield  # pragma: no cover

    def card_url(self, candidate: Candidate) -> str:
        return _KAGGLE_DATASET_URL.format(ref=candidate.id)

    def terms_check(self, intent: Intent) -> list[Obligation]:
        return [
            Obligation(
                source=self.name,
                summary=(
                    "Kaggle datasets carry per-dataset terms; verify the "
                    "license and any competition/host restrictions before redistributing."
                ),
                url="https://www.kaggle.com/terms",
            )
        ]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


__all__ = ["KaggleSource", "kaggle_credentials"]
