"""Tests for decomposition.yaml persistence + reuse (recommendation F)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dataset_scout.core import DecompositionDirection
from dataset_scout.decomposition_io import (
    DECOMPOSITION_FILE_VERSION,
    load_decomposition,
    write_decomposition,
)

pytestmark = pytest.mark.unit


def _direction(name: str = "x") -> DecompositionDirection:
    return DecompositionDirection(
        name=name,
        rationale="why " + name,
        keywords=[name + "_kw"],
    )


def test_write_returns_none_on_empty_input(tmp_path: Path):
    assert write_decomposition([], tmp_path) is None


def test_write_to_explicit_yaml_path(tmp_path: Path):
    """User passes ``--out path/file.yaml`` — write there directly,
    don't create a folder named ``file.yaml`` and nest the file inside."""
    target_path = tmp_path / "nested" / "my_decomposition.yaml"
    target = write_decomposition([_direction("a")], target_path)
    assert target == target_path
    assert target_path.is_file()
    # The parent (which we autocreated) must NOT have a directory of the same name inside it.
    assert not (target_path.parent / "my_decomposition.yaml" / "decomposition.yaml").exists()


def test_write_to_explicit_yml_path(tmp_path: Path):
    target_path = tmp_path / "decomp.yml"
    target = write_decomposition([_direction("a")], target_path)
    assert target == target_path
    assert target_path.is_file()


def test_write_to_directory_uses_default_filename(tmp_path: Path):
    """Recon passes a directory; backwards compatibility."""
    target = write_decomposition([_direction("a")], tmp_path / "outdir")
    assert target is not None
    assert target.name == "decomposition.yaml"
    assert target.parent.name == "outdir"


def test_write_persists_yaml(tmp_path: Path):
    target = write_decomposition([_direction("a"), _direction("b")], tmp_path)
    assert target is not None
    assert target.exists()
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload["decomposition_version"] == DECOMPOSITION_FILE_VERSION
    assert [d["name"] for d in payload["directions"]] == ["a", "b"]


def test_round_trip(tmp_path: Path):
    originals = [_direction("a"), _direction("b"), _direction("c")]
    target = write_decomposition(originals, tmp_path)
    assert target is not None
    loaded = load_decomposition(target)
    assert [d.name for d in loaded] == ["a", "b", "c"]
    assert loaded[0].keywords == ["a_kw"]


def test_load_accepts_bare_list_form(tmp_path: Path):
    """Hand-written decomposition.yaml might be a bare list."""
    target = tmp_path / "raw.yaml"
    target.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "x",
                    "rationale": "r",
                    "keywords": [],
                    "threat_families": [],
                    "expected_finds": "",
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_decomposition(target)
    assert len(loaded) == 1
    assert loaded[0].name == "x"
