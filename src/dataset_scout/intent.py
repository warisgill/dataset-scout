"""Brief → :class:`Intent` parsers.

The :class:`HeuristicIntentParser` is the no-API-key default. It is
deliberately conservative: when a phrase does not match a known
pattern the corresponding :class:`Intent` field is left at its default
rather than guessed at. Flag-supplied values always override
parser-derived ones (the CLI is the source of truth for what the user
explicitly asked for).

The :class:`LLMIntentParser` lands in a later milestone; only the
:class:`IntentParser` :class:`~typing.Protocol` is exposed here so
callers can depend on the shape today.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from dataset_scout.core import Intent, LicensePolicy, SensitiveDomain


@runtime_checkable
class IntentParser(Protocol):
    def parse(self, brief: str, **flags: object) -> Intent: ...


# ─── Mapping tables ──────────────────────────────────────────────────
#
# Patterns are compiled with re.IGNORECASE and word-ish boundaries
# (``\b``) so "malware" doesn't fire on "malwareishly" but "prompt-
# injection" still matches because ``-`` is a word boundary. The lists
# below are the single source of truth — extend them rather than
# threading new keywords through the parser body.

# Language phrase → ISO code. Order doesn't matter here; we resolve
# first-occurrence ordering by scanning the brief.
_LANGUAGE_TOKENS: dict[str, str] = {
    "english": "en",
    "japanese": "ja",
    "chinese": "zh",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "arabic": "ar",
    "russian": "ru",
    "korean": "ko",
}

# A non-English-centric mix covering the four largest first-language
# populations across distinct script families (Latin/EN, Han/ZH,
# Latin/ES, Arabic/AR). The point is breadth of script coverage, not a
# claim about the world's "top N".
_MULTILINGUAL_DEFAULT: tuple[str, ...] = ("en", "zh", "es", "ar")

# Threat family patterns. More-specific patterns appear first so that
# overlap resolution (see ``_match_threat_families``) prefers them —
# e.g. "indirect prompt injection" wins over "prompt injection".
# Each entry: (family_name, regex_pattern).
_THREAT_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("indirect_injection", r"\bindirect(?:[\s\-_]+prompt)?[\s\-_]+injection\b"),
    ("prompt_injection", r"\bprompt[\s\-_]+injection\b"),
    ("jailbreak", r"\bjailbreak(?:s|ing)?\b"),
    ("unsafe_output", r"\b(?:unsafe|harmful)[\s\-]+output(?:s)?\b"),
    ("exfiltration", r"\b(?:data[\s\-]+)?exfiltration\b"),
    ("memory_poisoning", r"\b(?:memory|context)[\s\-]+poisoning\b"),
    ("tool_use_safety", r"\btool[\s\-_]+(?:call|use)(?:s)?\b"),
    ("malware", r"\bmalware\b"),
)

# Deployment-context phrases. Order = priority: the first match wins.
# The captured value is what we surface in ``Intent.deployment_context``.
_DEPLOYMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("mcp_server", r"\bMCP[\s\-]+server\b"),
    ("rag_pipeline", r"\bRAG[\s\-]+pipeline\b"),
    ("x_facing", r"\b([a-z]+-facing)\b"),
    ("agent", r"\bagent(?:s|ic)?\b"),
    ("production", r"\bproduction\b"),
    ("deployed", r"\bdeployed\b"),
)

_DETECTION_TARGET_MAX_LEN = 80
# Capture a noun-ish phrase after the trigger up to a clause boundary.
_DETECTION_TARGET_PATTERNS: tuple[str, ...] = (
    r"\bdetect(?:ing|ion\s+of)?\s+([^.,;:\n]+)",
    r"\bclassifier(?:s)?\s+for\s+([^.,;:\n]+)",
    r"\blabels?\s+for\s+([^.,;:\n]+)",
)


# ─── Helpers ─────────────────────────────────────────────────────────


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _match_languages(brief: str) -> list[str]:
    """Find language codes mentioned in ``brief``, ordered by first
    occurrence. Empty list when no hint is present (caller falls back
    to the Intent default of ["en"])."""
    lower = brief.lower()
    hits: list[tuple[int, str]] = []

    if (idx := lower.find("multilingual")) != -1:
        for code in _MULTILINGUAL_DEFAULT:
            hits.append((idx, code))

    for token, code in _LANGUAGE_TOKENS.items():
        for m in re.finditer(rf"\b{re.escape(token)}\b", lower):
            hits.append((m.start(), code))
            break  # only first occurrence per token contributes

    hits.sort(key=lambda p: p[0])
    return _ordered_unique([code for _, code in hits])


def _match_threat_families(brief: str) -> list[str]:
    """Return threat-family names found in ``brief``, ordered by first
    occurrence. Overlapping matches resolve in favor of the longer
    span (so "indirect prompt injection" suppresses "prompt
    injection")."""
    spans: list[tuple[int, int, str]] = []
    for family, pattern in _THREAT_FAMILY_PATTERNS:
        for m in re.finditer(pattern, brief, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), family))

    # Greedy non-overlap: prefer longer spans, then earlier start.
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    accepted: list[tuple[int, int, str]] = []
    for start, end, family in spans:
        if any(not (end <= a_start or start >= a_end) for a_start, a_end, _ in accepted):
            continue
        accepted.append((start, end, family))

    accepted.sort(key=lambda s: s[0])
    return _ordered_unique([family for _, _, family in accepted])


def _match_deployment_context(brief: str) -> str | None:
    for label, pattern in _DEPLOYMENT_PATTERNS:
        m = re.search(pattern, brief, flags=re.IGNORECASE)
        if m is None:
            continue
        if label == "x_facing":
            return m.group(1).lower()
        return label
    return None


def _match_detection_target(brief: str) -> str:
    for pattern in _DETECTION_TARGET_PATTERNS:
        m = re.search(pattern, brief, flags=re.IGNORECASE)
        if m is None:
            continue
        captured = m.group(1).strip()
        if captured:
            return captured[:_DETECTION_TARGET_MAX_LEN].rstrip()
    return brief.strip()[:_DETECTION_TARGET_MAX_LEN].rstrip()


def _match_license_policy(brief: str) -> LicensePolicy:
    lower = brief.lower()
    # "any license" / "no license restriction" → no filtering.
    if re.search(r"\bany\s+license\b", lower) or re.search(
        r"\bno\s+license\s+restriction(?:s)?\b", lower
    ):
        return LicensePolicy(allow=frozenset(), warn_only=frozenset())

    extras: set[str] = set()
    if re.search(r"\bresearch\s+use\s+ok\b", lower):
        extras.update({"CC-BY-SA-4.0", "GPL-3.0"})
    if re.search(r"\bshare[\s\-]?alike\s+ok\b", lower):
        extras.update({"CC-BY-SA-4.0", "GPL-3.0"})

    if extras:
        default = LicensePolicy()
        return LicensePolicy(
            allow=default.allow | frozenset(extras),
            warn_only=default.warn_only,
        )

    # "permissive" / "permissive license" / no hint → default policy.
    return LicensePolicy()


# ─── Flag coercion ───────────────────────────────────────────────────


def _as_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"flag {name!r} must be a string, got {type(value).__name__}")
    return value


def _as_str_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise TypeError(f"flag {name!r} must be a list[str]")
    return list(value)


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"flag {name!r} must be a number")
    return float(value)


# ─── Parsers ─────────────────────────────────────────────────────────


class HeuristicIntentParser:
    """Default parser. No API key required. Fast and transparent.

    Maps natural-language phrases in the brief to :class:`Intent`
    fields via small regex tables. Flag-supplied values always
    override parser-derived values.

    ``must_have`` / ``nice_to_have`` / ``must_not`` are intentionally
    left empty in M1a: extracting them reliably from free-form prose
    requires either an LLM or a brittle DSL, and downstream tools
    already let users express such constraints in the recipe. The
    fields exist on :class:`Intent` so the LLM parser can populate
    them later without breaking the schema.
    """

    def parse(self, brief: str, **flags: object) -> Intent:
        languages = _match_languages(brief) or ["en"]
        threat_families = _match_threat_families(brief)
        sensitive_domain = (
            SensitiveDomain.GATED if "malware" in threat_families else SensitiveDomain.NONE
        )
        deployment_context = _match_deployment_context(brief)
        detection_target = _match_detection_target(brief)
        license_policy = _match_license_policy(brief)
        min_strategy_confidence = 0.5

        # Apply flag overrides (CLI flag names, hyphens → underscores).
        if "language" in flags:
            languages = _as_str_list(flags["language"], "language")
        if "threat_families" in flags:
            threat_families = _as_str_list(flags["threat_families"], "threat_families")
            sensitive_domain = (
                SensitiveDomain.GATED if "malware" in threat_families else SensitiveDomain.NONE
            )
        if "license" in flags:
            allow = _as_str_list(flags["license"], "license")
            license_policy = LicensePolicy(allow=frozenset(allow))
        if "detection_target" in flags:
            detection_target = _as_str(flags["detection_target"], "detection_target")
        if "deployment_context" in flags:
            deployment_context = _as_str(flags["deployment_context"], "deployment_context")
        if "min_strategy_confidence" in flags:
            min_strategy_confidence = _as_float(
                flags["min_strategy_confidence"], "min_strategy_confidence"
            )

        return Intent(
            raw_brief=brief,
            detection_target=detection_target or None,
            threat_families=threat_families,
            deployment_context=deployment_context,
            languages=languages,
            license_policy=license_policy,
            sensitive_domain=sensitive_domain,
            min_strategy_confidence=min_strategy_confidence,
        )


__all__ = ["HeuristicIntentParser", "IntentParser"]
