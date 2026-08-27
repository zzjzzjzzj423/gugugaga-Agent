from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Exchange:
    turn_id: str
    user_content: str
    assistant_content: str
    completed_at: str


@dataclass(frozen=True)
class Batch:
    id: str
    exchanges: tuple[Exchange, ...]
    attempt_count: int
    lease_expires_at: str


@dataclass(frozen=True)
class FactCandidate:
    subject: str
    content: str


@dataclass(frozen=True)
class ConsolidationResult:
    facts: tuple[FactCandidate, ...] = field(default_factory=tuple)
    episode: str | None = None


@dataclass(frozen=True)
class SaveNoteResult:
    status: str
    fact_id: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        messages = {
            "added": "Memory saved.",
            "duplicate": "This memory already exists.",
            "rejected": "Memory was rejected by validation.",
            "failed": "Memory could not be saved.",
        }
        value: dict[str, Any] = {
            "status": self.status,
            "message": messages.get(self.status, self.status),
        }
        if self.fact_id:
            value["fact_id"] = self.fact_id
        if self.error_code:
            value["error_code"] = self.error_code
        return value

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)
