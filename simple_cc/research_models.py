from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol


class TaskKind(str, Enum):
    NORMAL = "normal"
    RESEARCH = "research"


def normalize_task_kind(task_type: str | None) -> TaskKind:
    value = str(task_type or "").strip().lower()
    return (
        TaskKind.RESEARCH
        if value in {"research", "research_analysis"}
        else TaskKind.NORMAL
    )


class ResearchRank(str, Enum):
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass(frozen=True)
class RankPolicy:
    max_research_rounds: int
    distinct_source_count: int
    authoritative_source_count: int
    research_direction_count: int


RANK_POLICIES: Mapping[ResearchRank, RankPolicy] = MappingProxyType({
    ResearchRank.LIGHT: RankPolicy(10, 2, 1, 1),
    ResearchRank.STANDARD: RankPolicy(20, 3, 1, 2),
    ResearchRank.DEEP: RankPolicy(30, 4, 2, 3),
})


@dataclass(frozen=True)
class ResearchPlan:
    rank: ResearchRank
    directions: tuple[str, ...]
    reason: str
    used_fallback: bool = False
    validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cleaned = tuple(item.strip() for item in self.directions if item.strip())
        required = self.policy.research_direction_count
        if len(cleaned) != required:
            raise ValueError(f"{self.rank.value} requires {required} research directions")
        normalized = tuple(" ".join(item.casefold().split()) for item in cleaned)
        if len(set(normalized)) != len(normalized):
            raise ValueError("research directions must be unique")
        object.__setattr__(self, "directions", cleaned)

    @property
    def policy(self) -> RankPolicy:
        return RANK_POLICIES[self.rank]


@dataclass
class ResearchBudget:
    max_rounds: int
    used_rounds: int = 0

    @property
    def remaining_rounds(self) -> int:
        return self.max_rounds - self.used_rounds

    def consume(self, rounds: int) -> None:
        if rounds < 0 or rounds > self.remaining_rounds:
            raise ValueError("rounds exceed remaining research budget")
        self.used_rounds += rounds


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    canonical_url: str
    domain: str
    title: str | None
    content_excerpt: str
    published_at: str | None
    date_status: str | None
    cutoff: str | None
    tool_name: str
    authoritative: bool = False
    authority_reason: str | None = None


class EvidenceRegistry:
    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records.values())

    def register(self, record: EvidenceRecord) -> EvidenceRecord:
        return self._records.setdefault(record.canonical_url, record)

    def get_by_id(self, source_id: str) -> EvidenceRecord | None:
        return next(
            (item for item in self._records.values() if item.source_id == source_id),
            None,
        )

    def clear_authority(self) -> None:
        self._records = {
            url: replace(item, authoritative=False, authority_reason=None)
            for url, item in self._records.items()
        }

    def mark_authority(self, source_id: str, authoritative: bool, reason: str) -> None:
        record = self.get_by_id(source_id)
        if record is None:
            raise ValueError(f"unknown evidence source id: {source_id}")
        self._records[record.canonical_url] = replace(
            record,
            authoritative=authoritative,
            authority_reason=reason.strip(),
        )


class ResearchExecutionOutcome(Protocol):
    status: str
    final_text: str
    failure_class: str | None
    failure_message: str | None
    rounds_used: int


@dataclass(frozen=True)
class DirectionAssessment:
    direction: str
    covered: bool
    source_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ResearchGateDecision:
    passed: bool
    source_count: int
    domain_count: int
    authoritative_source_ids: tuple[str, ...]
    directions: tuple[DirectionAssessment, ...]
    gaps: tuple[str, ...]
    validation_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchWorkflowResult:
    final_text: str
    plan: ResearchPlan
    research_rounds_used: int
    supplemental_research_used: bool
    writing_repair_used: bool
