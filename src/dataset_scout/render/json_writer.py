"""results.json writer."""

from __future__ import annotations

from pathlib import Path

from dataset_scout.core import ReconResult


def write_results_json(result: ReconResult, out_dir: Path) -> Path:
    """Write the recon result as `<out_dir>/results.json` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "results.json"
    target.write_text(
        result.model_dump_json(indent=2, exclude_none=False) + "\n",
        encoding="utf-8",
    )
    return target
