"""Probe namespace.

Each probe is a small class implementing the `Probe` protocol from
`probes.base`. The "cheap" probes in `probes.cheap` consume only the
normalized `CandidateMetadata` envelope — no row sampling. Sample-driven
probes (label_structure, schema_fingerprint) land in M1b once sources
support `stream_sample`.
"""

from __future__ import annotations

from dataset_scout.probes.base import Probe, ProbeRegistry
from dataset_scout.probes.cheap import (
    CardCompletenessProbe,
    FreshnessProbe,
    LanguagesProbe,
    LicenseProbe,
    RecencyProbe,
    SizeProbe,
    cheap_probes,
)

__all__ = [
    "CardCompletenessProbe",
    "FreshnessProbe",
    "LanguagesProbe",
    "LicenseProbe",
    "Probe",
    "ProbeRegistry",
    "RecencyProbe",
    "SizeProbe",
    "cheap_probes",
]
