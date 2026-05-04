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


def write_decomposition(
    directions: list[DecompositionDirection], out: Path
) -> Path | None:
    """Write decomposition YAML.

    `out` may be either a directory (in which case the file is written
    as ``<out>/decomposition.yaml`` — what `recon` and `tour` pass)
    or an explicit ``.yaml``/``.yml`` file path (what users typically
    pass to ``datascout decompose --out``). Heuristic: a path with a
    ``.yaml``/``.yml`` suffix is treated as a file, anything else as
    a directory. Returns the resolved file path, or None if there
    are no directions to write.
    """
    if not directions:
        return None
    if out.suffix.lower() in (".yaml", ".yml"):
        target = out
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        target = out / "decomposition.yaml"
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
