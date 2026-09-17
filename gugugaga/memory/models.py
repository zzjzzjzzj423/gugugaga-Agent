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
    importance: float = 1.0
    conflict_ids: tuple[str, ...] = field(default_factory=tuple)
    conflict_reason: str = ""
    reviewed_candidates: tuple[tuple[str, str, str], ...] | None = None


@dataclass(frozen=True)
class EpisodeCandidate:
    summary: str
    importance: float = 1.0


@dataclass(frozen=True)
class ConflictProbe:
    existing_fact_id: str
    subject: str
    content: str
    reason: str
    turn_id: str
    evidence: str


@dataclass(frozen=True)
class BatchConflictReview:
    query: str
    reviewed_candidates: tuple[tuple[str, str, str], ...]
    conflicts: tuple[ConflictProbe, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ConsolidationResult:
    facts: tuple[FactCandidate, ...] = field(default_factory=tuple)
    episodes: tuple[EpisodeCandidate, ...] = field(default_factory=tuple)
    conflict_review: BatchConflictReview | None = None


@dataclass(frozen=True)
class SaveNoteResult:
    status: str
    fact_id: str | None = None
    error_code: str | None = None
    conflict_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        messages = {
            "added": "Memory saved.",
            "duplicate": "This memory already exists.",
            "rejected": "Memory was rejected by validation.",
            "failed": "Memory could not be saved.",
            "pending": "This memory conflicts with an existing record and needs confirmation in Memory. Neither version is treated as a current fact until resolved.",
        }
        value: dict[str, Any] = {
            "status": self.status,
            "message": messages.get(self.status, self.status),
        }
        if self.fact_id:
            value["fact_id"] = self.fact_id
        if self.error_code:
            value["error_code"] = self.error_code
        if self.conflict_id:
            value["conflict_id"] = self.conflict_id
        return value

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


@dataclass(frozen=True)
class RecallItem:
    """One structured candidate selected for a turn-scoped recall."""

    memory_key: str
    kind: str
    subject: str
    text: str
    occurred_at: str | None = None
    retrieval_sources: tuple[str, ...] = field(default_factory=tuple)
    source_ranks: dict[str, int] = field(default_factory=dict)
    relevance_score: float = 0.0
    final_score: float = 0.0
    source_turn_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def feedback_enabled(self) -> bool:
        return self.kind in {"fact", "episode"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_key": self.memory_key,
            "kind": self.kind,
            "subject": self.subject,
            "text": self.text,
            "occurred_at": self.occurred_at,
            "retrieval_sources": list(self.retrieval_sources),
            "source_ranks": dict(self.source_ranks),
            "relevance_score": self.relevance_score,
            "final_score": self.final_score,
            "source_turn_ids": list(self.source_turn_ids),
            "feedback_enabled": self.feedback_enabled,
        }


@dataclass(frozen=True)
class RecallResult:
    """One turn-scoped Retrieval Gate decision and its prompt payload."""

    content: str = ""
    decision: str = "skip"
    reason: str = "no_relevant_memory"
    hit_count: int = 0
    kinds: tuple[str, ...] = field(default_factory=tuple)
    memory_keys: tuple[str, ...] = field(default_factory=tuple)
    items: tuple[RecallItem, ...] = field(default_factory=tuple)
    strategy: str = "none"
    route: str = "mixed"
    route_source: str = "default"
    route_confidence: float | None = None
    pending_conflicts: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def should_inject(self) -> bool:
        return self.decision == "retrieve" and bool(self.content)
