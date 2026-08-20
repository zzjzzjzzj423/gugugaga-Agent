from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol
from urllib.parse import urlsplit

from .trace import ArtifactRef


EVIDENCE_CONTENT_CHARS_MAX = 6000
EVIDENCE_PDF_FRAGMENT_CHARS_MAX = 700
EVIDENCE_PDF_FRAGMENTS_MAX = 8
EVIDENCE_ARTIFACT_REFERENCES_MAX = 16


def _evidence_fragment_sort_key(key: str) -> tuple[int, int, str]:
    prefix = "pdf_page:"
    page_number = key.removeprefix(prefix)
    if (
        key.startswith(prefix)
        and page_number.isascii()
        and page_number.isdecimal()
    ):
        return (0, int(page_number), key)
    return (1, 0, key)


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

    @staticmethod
    def _reject(code: str, message: str) -> None:
        raise EvidenceRegistrationError(code, message)

    @classmethod
    def _validate_record(cls, record: EvidenceRecord) -> None:
        # Local import keeps evidence extraction dependent on these models while
        # making the registry the single runtime trust boundary for all callers.
        from .evidence import (
            EVIDENCE_URL_CHARS_MAX,
            canonicalize_url,
            source_id_for_url,
        )

        if not isinstance(record, EvidenceRecord):
            cls._reject("invalid_record", "evidence must be an EvidenceRecord")
        try:
            canonical = canonicalize_url(record.canonical_url)
        except (TypeError, ValueError) as error:
            cls._reject("invalid_url", str(error))
        if (
            canonical != record.canonical_url
            or len(canonical) > EVIDENCE_URL_CHARS_MAX
        ):
            cls._reject(
                "invalid_url",
                "evidence URL must be exact canonical HTTP(S) identity",
            )
        if (
            not isinstance(record.source_id, str)
            or record.source_id != source_id_for_url(canonical)
        ):
            cls._reject(
                "source_id_mismatch",
                "evidence source_id does not match canonical URL",
            )
        expected_domain = urlsplit(canonical).hostname or ""
        if not isinstance(record.domain, str) or record.domain != expected_domain:
            cls._reject(
                "domain_mismatch",
                "evidence domain does not match canonical URL",
            )
        if record.tool_name not in {"web_fetch", "pdf_fetch"}:
            cls._reject(
                "invalid_tool_name",
                "registered evidence must come from a fetch tool",
            )
        content = record.content_excerpt
        if (
            not isinstance(content, str)
            or not content
            or content != content.strip()
            or len(content) > EVIDENCE_CONTENT_CHARS_MAX
            or not any(
                not character.isspace()
                and not unicodedata.category(character).startswith("C")
                for character in content
            )
        ):
            cls._reject(
                "invalid_content",
                "evidence content must be usable, normalized, and bounded",
            )
        if record.title is not None and (
            not isinstance(record.title, str)
            or not record.title
            or record.title != record.title.strip()
            or len(record.title) > 4096
            or any(
                unicodedata.category(character).startswith("C")
                for character in record.title
            )
        ):
            cls._reject("invalid_metadata", "evidence title is invalid")
        if record.date_status not in {"unknown", "verified"}:
            cls._reject("invalid_metadata", "evidence date_status is invalid")

        def parsed_iso(value: str | None, field: str) -> date | None:
            if value is None:
                return None
            if not isinstance(value, str):
                cls._reject("invalid_metadata", f"evidence {field} is invalid")
            try:
                parsed = date.fromisoformat(value)
            except ValueError:
                cls._reject("invalid_metadata", f"evidence {field} is invalid")
            if parsed.isoformat() != value:
                cls._reject("invalid_metadata", f"evidence {field} is invalid")
            return parsed

        published = parsed_iso(record.published_at, "published_at")
        cutoff = parsed_iso(record.cutoff, "cutoff")
        if (
            (record.date_status == "verified" and published is None)
            or (record.date_status == "unknown" and published is not None)
            or (
                published is not None
                and cutoff is not None
                and published > cutoff
            )
        ):
            cls._reject("invalid_metadata", "evidence date metadata conflicts")
        if (
            not isinstance(record.authoritative, bool)
            or record.authoritative
            or record.authority_reason is not None
        ):
            cls._reject(
                "untrusted_authority",
                "authority may only be assigned by the research gate",
            )
        if not isinstance(record.content_fragments, tuple) or not all(
            isinstance(fragment, EvidenceFragment)
            for fragment in record.content_fragments
        ):
            cls._reject("invalid_content", "evidence fragments are invalid")
        if len(record.content_fragments) > EVIDENCE_PDF_FRAGMENTS_MAX:
            cls._reject("invalid_content", "too many evidence fragments")
        if not isinstance(record.artifact_references, tuple) or not all(
            isinstance(artifact, ArtifactRef)
            for artifact in record.artifact_references
        ):
            cls._reject("invalid_record", "evidence artifacts are invalid")
        if len(record.artifact_references) > EVIDENCE_ARTIFACT_REFERENCES_MAX:
            cls._reject("invalid_record", "too many evidence artifacts")
        fragment_keys: set[str] = set()
        for fragment in record.content_fragments:
            page_number = fragment.key.removeprefix("pdf_page:")
            if (
                not fragment.key.startswith("pdf_page:")
                or not page_number.isascii()
                or not page_number.isdecimal()
                or fragment.key in fragment_keys
                or not fragment.content
                or len(fragment.content) > EVIDENCE_PDF_FRAGMENT_CHARS_MAX
            ):
                cls._reject("invalid_content", "evidence fragment is invalid")
            fragment_keys.add(fragment.key)
        if record.tool_name == "web_fetch" and record.content_fragments:
            cls._reject("invalid_content", "web evidence cannot have PDF fragments")
        if record.tool_name == "pdf_fetch":
            if not record.content_fragments:
                cls._reject("invalid_content", "PDF evidence requires fragments")
            expected_excerpt = "\n\n".join(
                fragment.content for fragment in record.content_fragments
            )[:EVIDENCE_CONTENT_CHARS_MAX]
            if record.content_excerpt != expected_excerpt:
                cls._reject(
                    "invalid_content",
                    "PDF evidence excerpt does not match its fragments",
                )

    def register(self, record: EvidenceRecord) -> EvidenceRecord:
        self._validate_record(record)
        colliding = self.get_by_id(record.source_id)
        if (
            colliding is not None
            and colliding.canonical_url != record.canonical_url
        ):
            raise EvidenceRegistrationError(
                "source_id_collision",
                "evidence source_id is already registered to another URL",
            )
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
            for key in sorted(fragments, key=_evidence_fragment_sort_key)[
                :EVIDENCE_PDF_FRAGMENTS_MAX
            ]
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
