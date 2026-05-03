"""ProgressEvent — the unified event protocol the pipeline emits.

The CLI subscribes to these and renders rich progress; a future HTTP
server would forward them as SSE / WebSocket frames. Same code path.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProgressEventKind(StrEnum):
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    CANDIDATE_FOUND = "candidate_found"
    CANDIDATE_SCORED = "candidate_scored"
    DIRECTION_PROPOSED = "direction_proposed"
    STRATEGY_ASSESSED = "strategy_assessed"
    NOTICE = "notice"
    WARNING = "warning"


class ProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ProgressEventKind
    stage: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
