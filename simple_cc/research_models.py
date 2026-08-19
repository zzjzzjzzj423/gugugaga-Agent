from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol

from .trace import ArtifactRef


EVIDENCE_CONTENT_CHARS_MAX = 6000
EVIDENCE_PDF_FRAGMENT_CHARS_MAX = 700
EVIDENCE_PDF_FRAGMENTS_MAX = 8
EVIDENCE_ARTIFACT_REFERENCES_MAX = 16


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
class EvidenceFragment:
    key: str
    content: str


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
    content_fragments: tuple[EvidenceFragment, ...] = ()
    artifact_references: tuple[ArtifactRef, ...] = ()


class EvidenceRegistrationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class EvidenceRegistry:
    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records.values())

    def register(self, record: EvidenceRecord) -> EvidenceRecord:
        existing = self._records.get(record.canonical_url)
        if existing is None:
            self._records[record.canonical_url] = record
            return record

        for field_name in ("published_at", "date_status", "cutoff"):
            old_value = getattr(existing, field_name)
            new_value = getattr(record, field_name)
            if old_value is not None and new_value is not None and old_value != new_value:
                raise EvidenceRegistrationError(
                    f"conflicting_{field_name}",
                    f"repeated source has conflicting {field_name} metadata",
                )

        fragments = {item.key: item for item in existing.content_fragments}
        for item in record.content_fragments:
            fragments.setdefault(item.key, item)
        merged_fragments = tuple(
            fragments[key]
            for key in sorted(fragments)[:EVIDENCE_PDF_FRAGMENTS_MAX]
        )
        if merged_fragments:
            content_excerpt = "\n\n".join(
                item.content for item in merged_fragments
            )[:EVIDENCE_CONTENT_CHARS_MAX]
        else:
            content_excerpt = existing.content_excerpt

        artifact_references = list(existing.artifact_references)
        known_artifacts = {item.sha256 for item in artifact_references}
        for item in record.artifact_references:
            if len(artifact_references) >= EVIDENCE_ARTIFACT_REFERENCES_MAX:
                break
            if item.sha256 in known_artifacts:
                continue
            artifact_references.append(item)
            known_artifacts.add(item.sha256)

        merged = replace(
            existing,
            title=existing.title or record.title,
            content_excerpt=content_excerpt,
            published_at=existing.published_at or record.published_at,
            date_status=existing.date_status or record.date_status,
            cutoff=existing.cutoff or record.cutoff,
            content_fragments=merged_fragments,
            artifact_references=tuple(artifact_references),
        )
        if merged == existing:
            return existing
        self._records[record.canonical_url] = merged
        return merged

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
