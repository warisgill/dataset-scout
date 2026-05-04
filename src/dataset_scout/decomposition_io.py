"""decomposition.yaml writer + reader.

Persists the LLM-proposed decomposition directions as a stand-alone
artefact alongside results.json / report.md. The directions are often
the most actionable output for novel briefs (especially when HF
candidate coverage is sparse) — surfacing them as their own file lets
users hand-edit, share, or feed back into a fresh recon via
`--decomposition-from`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from dataset_scout.core import DecompositionDirection

# Bumped when the on-disk shape changes incompatibly.
DECOMPOSITION_FILE_VERSION = "1"


def _to_payload(directions: list[DecompositionDirection]) -> dict[str, Any]:
    return {
        "decomposition_version": DECOMPOSITION_FILE_VERSION,
        "directions": [d.model_dump(mode="json") for d in directions],
    }


def write_decomposition(directions: list[DecompositionDirection], out_dir: Path) -> Path | None:
    """Write `<out_dir>/decomposition.yaml`. Returns None when empty."""
    if not directions:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "decomposition.yaml"
    target.write_text(
        yaml.safe_dump(_to_payload(directions), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def load_decomposition(path: Path) -> list[DecompositionDirection]:
    """Read a decomposition.yaml back into a list of DecompositionDirection.

    Accepts both the wrapped form (`{decomposition_version, directions}`)
    and a bare list at the top level.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("directions", [])
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [DecompositionDirection.model_validate(item) for item in items]
