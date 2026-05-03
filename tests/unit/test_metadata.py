"""Tests for the CandidateMetadata envelope."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataset_scout import Candidate, CandidateMetadata, ColumnInfo

pytestmark = pytest.mark.unit


def test_metadata_minimal_defaults():
    m = CandidateMetadata()
    assert m.license_raw is None
    assert m.license_spdx is None
    assert m.languages_declared == []
    assert m.card_fields_present == frozenset()
    assert m.gated is False
    assert m.requires_auth is False


def test_metadata_round_trip_json():
    m = CandidateMetadata(
        license_raw="mit",
        license_spdx="MIT",
        languages_declared=["en", "ja"],
        card_fields_present=frozenset({"license", "language"}),
        columns=[ColumnInfo(name="text", dtype="string")],
        downloads=100,
    )
    rehydrated = CandidateMetadata.model_validate_json(m.model_dump_json())
    assert rehydrated == m


def test_metadata_card_fields_coerces_from_list():
    m = CandidateMetadata(card_fields_present=["license", "language"])  # type: ignore[arg-type]
    assert m.card_fields_present == frozenset({"license", "language"})


def test_metadata_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CandidateMetadata(unknown_field=True)  # type: ignore[call-arg]


def test_candidate_default_metadata_is_envelope():
    c = Candidate(source="huggingface", id="org/x", revision="r1")
    assert isinstance(c.metadata, CandidateMetadata)
    assert c.requires_auth is False


def test_candidate_requires_auth_reflects_metadata():
    c = Candidate(
        source="huggingface",
        id="org/x",
        revision="r1",
        metadata=CandidateMetadata(gated=True),
    )
    assert c.requires_auth is True

    c2 = Candidate(
        source="huggingface",
        id="org/x",
        revision="r1",
        metadata=CandidateMetadata(requires_auth=True),
    )
    assert c2.requires_auth is True
