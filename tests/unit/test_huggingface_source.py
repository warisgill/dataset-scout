"""Tests for the HuggingFaceSource translation layer.

These cover only the pure DatasetInfo -> CandidateMetadata translation.
Networked behavior is exercised by recorded HTTP tests under tests/recorded/
once cassettes land.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from dataset_scout.core import CandidateMetadata
from dataset_scout.licenses import guess_spdx
from dataset_scout.sources.huggingface import (
    _build_metadata,
    _build_search_query,
    _card_data_to_dict,
    _coerce_dt,
    _coerce_languages,
)

pytestmark = pytest.mark.unit


# ─── Fixture: a stand-in for huggingface_hub's DatasetInfo ───────────


@dataclass
class FakeCardData:
    """Minimal stand-in for huggingface_hub.DatasetCardData.

    Real DatasetCardData supports `.get(key, default)` and `.to_dict()`.
    """

    _data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


@dataclass
class FakeDatasetInfo:
    """Stand-in for huggingface_hub.hf_api.DatasetInfo."""

    id: str
    sha: str | None = None
    card_data: FakeCardData | None = None
    description: str | None = None
    tags: list[str] | None = None
    downloads: int | None = None
    likes: int | None = None
    gated: bool | str = False
    private: bool = False
    created_at: datetime | None = None
    last_modified: datetime | None = None
    main_size: int | dict[str, Any] | None = None
    citation: str | None = None
    paperswithcode_id: str | None = None


# ─── _coerce_languages ──────────────────────────────────────────────


def test_coerce_languages_handles_string_list_and_none():
    assert _coerce_languages(None) == []
    assert _coerce_languages("en") == ["en"]
    assert _coerce_languages(["en", "ja", "zh"]) == ["en", "ja", "zh"]
    # Numeric / odd entries are stringified or dropped, never crash.
    assert _coerce_languages([1, 2]) == ["1", "2"]
    assert _coerce_languages({"unexpected": "shape"}) == []


# ─── _coerce_dt ─────────────────────────────────────────────────────


def test_coerce_dt_passes_through_datetime():
    dt = datetime(2026, 1, 1, tzinfo=UTC)
    assert _coerce_dt(dt) is dt


def test_coerce_dt_parses_iso_strings():
    out = _coerce_dt("2026-01-02T03:04:05Z")
    assert isinstance(out, datetime)
    assert out.year == 2026


def test_coerce_dt_returns_none_on_garbage():
    assert _coerce_dt("not a date") is None
    assert _coerce_dt(None) is None


# ─── _card_data_to_dict ─────────────────────────────────────────────


def test_card_data_to_dict_handles_none():
    assert _card_data_to_dict(None) == {}


def test_card_data_to_dict_handles_dict_passthrough():
    d = {"license": "mit"}
    assert _card_data_to_dict(d) == d


def test_card_data_to_dict_uses_to_dict_method():
    out = _card_data_to_dict(FakeCardData({"license": "apache-2.0"}))
    assert out == {"license": "apache-2.0"}


# ─── _build_metadata ────────────────────────────────────────────────


def test_build_metadata_minimal_info():
    info = FakeDatasetInfo(id="org/x")
    meta = _build_metadata(info)
    assert isinstance(meta, CandidateMetadata)
    assert meta.card_url == "https://huggingface.co/datasets/org/x"
    assert meta.license_raw is None
    assert meta.license_spdx is None
    assert meta.languages_declared == []
    assert meta.gated is False
    assert meta.requires_auth is False
    assert meta.card_fields_present == frozenset()


def test_build_metadata_populates_license_and_languages_from_card():
    card = FakeCardData(
        {
            "license": "mit",
            "language": ["en", "ja"],
            "task_categories": ["text-classification"],
        }
    )
    info = FakeDatasetInfo(id="org/x", card_data=card)
    meta = _build_metadata(info)
    assert meta.license_raw == "mit"
    assert meta.license_spdx == "MIT"
    assert meta.languages_declared == ["en", "ja"]
    assert meta.task_categories == ["text-classification"]
    assert {"license", "language", "task_categories"} <= meta.card_fields_present


def test_build_metadata_handles_list_of_licenses():
    card = FakeCardData({"license": ["mit", "apache-2.0"]})
    info = FakeDatasetInfo(id="org/x", card_data=card)
    meta = _build_metadata(info)
    assert meta.license_raw == "mit"
    assert meta.license_spdx == "MIT"
    assert meta.extras.get("additional_licenses") == ["apache-2.0"]


def test_build_metadata_passes_through_dates_and_counts():
    info = FakeDatasetInfo(
        id="org/x",
        downloads=1234,
        likes=42,
        created_at=datetime(2025, 6, 1, tzinfo=UTC),
        last_modified=datetime(2026, 1, 2, tzinfo=UTC),
    )
    meta = _build_metadata(info)
    assert meta.downloads == 1234
    assert meta.likes == 42
    assert meta.uploaded_at == datetime(2025, 6, 1, tzinfo=UTC)
    assert meta.last_modified == datetime(2026, 1, 2, tzinfo=UTC)


def test_build_metadata_marks_gated_and_private():
    info = FakeDatasetInfo(id="org/x", gated="manual", private=True)
    meta = _build_metadata(info)
    assert meta.gated is True
    assert meta.requires_auth is True


def test_build_metadata_handles_main_size_int_or_dict():
    info_int = FakeDatasetInfo(id="org/x", main_size=4096)
    info_dict = FakeDatasetInfo(id="org/x", main_size={"size_in_bytes": 8192})
    assert _build_metadata(info_int).bytes == 4096
    assert _build_metadata(info_dict).bytes == 8192


def test_build_metadata_unknown_license_yields_none_spdx():
    card = FakeCardData({"license": "see-license-file"})
    info = FakeDatasetInfo(id="org/x", card_data=card)
    meta = _build_metadata(info)
    assert meta.license_raw == "see-license-file"
    assert meta.license_spdx is None


# ─── _build_search_query ────────────────────────────────────────────


def test_build_search_query_uses_threat_families_when_set():
    from dataset_scout import Intent

    intent = Intent(
        raw_brief="find labeled prompt injection corpora in english",
        threat_families=["prompt_injection", "jailbreak"],
    )
    q = _build_search_query(intent)
    # Long free-text NL queries get zero HF hits; families are the
    # high-precision signal so we use them verbatim.
    assert q == "prompt injection jailbreak"


def test_build_search_query_falls_back_to_brief_when_no_families():
    from dataset_scout import Intent

    intent = Intent(raw_brief="dataset for classifying tool-call outputs")
    q = _build_search_query(intent)
    assert q == "dataset for classifying tool-call outputs"


# ─── guess_spdx (sanity) ────────────────────────────────────────────


def test_guess_spdx_canonicalizes_common_inputs():
    assert guess_spdx("MIT") == "MIT"
    assert guess_spdx("Apache 2.0") == "Apache-2.0"
    assert guess_spdx("apache-2") == "Apache-2.0"
    assert guess_spdx("CC-BY-4.0") == "CC-BY-4.0"
    assert guess_spdx("openrail++") == "OpenRAIL-M"


def test_guess_spdx_returns_none_for_unknowns():
    assert guess_spdx(None) is None
    assert guess_spdx("") is None
    assert guess_spdx("other") is None
    assert guess_spdx("see license file") is None
    assert guess_spdx("definitely not a real license") is None
