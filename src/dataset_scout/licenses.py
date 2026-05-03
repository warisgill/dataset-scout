"""SPDX license guesser.

Tiny, deterministic mapping from raw card-declared license strings to
SPDX-style identifiers. We only return values from the (extensible)
canonical set; if we don't recognise the input, we return None and let
the caller decide how to surface the raw string.

Honest by design: when the upstream card says "other" or
"see license file", we don't pretend to know.
"""

from __future__ import annotations

# Map raw lowercase tokens -> SPDX identifiers. Keep this small and
# deterministic. Unknowns return None.
_RAW_TO_SPDX: dict[str, str] = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache2": "Apache-2.0",
    "apache-2": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-3": "BSD-3-Clause",
    "bsd": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc0-1.0": "CC0-1.0",
    "cc0": "CC0-1.0",
    "cc-0": "CC0-1.0",
    "odc-by-1.0": "ODC-BY-1.0",
    "odc-by": "ODC-BY-1.0",
    "gpl-3.0": "GPL-3.0",
    "agpl-3.0": "AGPL-3.0",
    "openrail": "OpenRAIL-M",
    "openrail-m": "OpenRAIL-M",
    "openrail++": "OpenRAIL-M",
    "lgpl-3.0": "LGPL-3.0",
    "mpl-2.0": "MPL-2.0",
}


def guess_spdx(raw: str | None) -> str | None:
    """Return a best-effort SPDX identifier for a raw license string.

    Returns None for unknown / "other" / missing inputs — never fabricates.
    """
    if not raw:
        return None
    norm = raw.strip().lower()
    if not norm or norm in {"other", "unknown", "n/a", "see license"}:
        return None
    return _RAW_TO_SPDX.get(norm)
