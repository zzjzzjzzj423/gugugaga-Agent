from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .research_models import (
    EVIDENCE_ARTIFACT_REFERENCES_MAX,
    EVIDENCE_CONTENT_CHARS_MAX,
    EVIDENCE_PDF_FRAGMENT_CHARS_MAX,
    EVIDENCE_PDF_FRAGMENTS_MAX,
    EvidenceFragment,
    EvidenceRecord,
    EvidenceRegistry,
    ResearchPlan,
)
from .trace import ArtifactRef, RunContext


RESEARCH_TOOLS = {"web_search", "web_fetch", "pdf_fetch"}
EVIDENCE_EXCERPT_CHARS = EVIDENCE_CONTENT_CHARS_MAX


class CutoffMismatch(ValueError):
    pass


@dataclass(frozen=True)
class PreparedToolArguments:
    arguments: dict[str, Any]
    decision: str
    supplied_cutoff: str | None


@dataclass(frozen=True)
class EvidenceIngestionResult:
    record: EvidenceRecord | None
    rejection_code: str | None = None
    rejection_reason: str | None = None


def _rejected(code: str, reason: str) -> EvidenceIngestionResult:
    return EvidenceIngestionResult(None, code, reason)


def prepare_research_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    required_cutoff: str | None,
) -> PreparedToolArguments:
    copied = dict(arguments or {})
    supplied = copied.get("cutoff")
    if tool_name not in RESEARCH_TOOLS or required_cutoff is None:
        return PreparedToolArguments(copied, "not_required", supplied)
    if supplied is None:
        copied["cutoff"] = required_cutoff
        return PreparedToolArguments(copied, "injected", None)
    if supplied != required_cutoff:
        raise CutoffMismatch(
            f"tool cutoff {supplied!r} does not match task cutoff {required_cutoff!r}"
        )
    return PreparedToolArguments(copied, "matched", supplied)


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(str(url).strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"not a canonicalizable HTTP URL: {url}")
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    if port is not None and port != (443 if scheme == "https" else 80):
        host = f"{host}:{port}"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, parsed.path or "/", query, ""))


def source_id_for_url(url: str) -> str:
    canonical = canonicalize_url(url)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"src_{digest[:16]}"


_URL_PATTERN = re.compile(r"https?://[^\s<>\])]+", re.IGNORECASE)


def link_final_answer_sources(
    final_text: str, registered_sources: dict[str, str]
) -> dict[str, list[str]]:
    cited_urls: list[str] = []
    matched: list[str] = []
    unmatched: list[str] = []
    for raw in _URL_PATTERN.findall(final_text or ""):
        raw = raw.rstrip(".,;:!?\"'")
        try:
            canonical = canonicalize_url(raw)
        except ValueError:
            continue
        if canonical in cited_urls:
            continue
        cited_urls.append(canonical)
        source_id = registered_sources.get(canonical)
        if source_id is None:
            unmatched.append(canonical)
        elif source_id not in matched:
            matched.append(source_id)
    return {
        "cited_urls": cited_urls,
        "matched_source_ids": matched,
        "unmatched_citations": unmatched,
    }


def _parse_iso_date(
    value: Any,
    *,
    field_name: str,
) -> tuple[str | None, date | None, EvidenceIngestionResult | None]:
    if value is None:
        return None, None, None
    if not isinstance(value, str) or not value.strip():
        return None, None, _rejected(
            f"invalid_{field_name}",
            f"{field_name} must be null or an ISO date string",
        )
    normalized = value.strip()
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError:
        return None, None, _rejected(
            f"invalid_{field_name}",
            f"{field_name} must use YYYY-MM-DD",
        )
    if parsed.isoformat() != normalized:
        return None, None, _rejected(
            f"invalid_{field_name}",
            f"{field_name} must use YYYY-MM-DD",
        )
    return normalized, parsed, None


def _pdf_page_fragment(page_number: int, content: str) -> EvidenceFragment:
    start_marker = f"--- PAGE {page_number} START ---"
    end_marker = f"--- PAGE {page_number} END ---"
    body = content.strip()
    if body.startswith(start_marker) and body.endswith(end_marker):
        body = body[len(start_marker):-len(end_marker)].strip()
    fixed_chars = len(start_marker) + len(end_marker) + 2
    body = body[:max(0, EVIDENCE_PDF_FRAGMENT_CHARS_MAX - fixed_chars)].rstrip()
    fragment = f"{start_marker}\n{body}\n{end_marker}"
    return EvidenceFragment(f"pdf_page:{page_number:010d}", fragment)


def _pdf_fragments(payload: dict[str, Any]) -> tuple[
    tuple[EvidenceFragment, ...] | None,
    EvidenceIngestionResult | None,
]:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        return None, _rejected(
            "invalid_pdf_pages",
            "successful pdf_fetch payload must contain a non-empty pages list",
        )

    by_page: dict[int, EvidenceFragment] = {}
    for page in pages:
        if not isinstance(page, dict):
            return None, _rejected(
                "invalid_pdf_pages", "each PDF page must be an object"
            )
        page_number = page.get("page_number")
        content = page.get("content")
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            or not isinstance(content, str)
        ):
            return None, _rejected(
                "invalid_pdf_pages",
                "PDF pages require a positive integer page_number and string content",
            )
        if page_number in by_page:
            return None, _rejected(
                "invalid_pdf_pages", "PDF payload contains a repeated page number"
            )
        by_page[page_number] = _pdf_page_fragment(page_number, content)

    page_numbers = sorted(by_page)
    for field_name, expected in (
        ("start_page", page_numbers[0]),
        ("end_page", page_numbers[-1]),
    ):
        value = payload.get(field_name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value != expected
        ):
            return None, _rejected(
                "invalid_pdf_pages",
                f"{field_name} does not match the returned PDF pages",
            )
    return tuple(
        by_page[number]
        for number in page_numbers[:EVIDENCE_PDF_FRAGMENTS_MAX]
    ), None


def inspect_evidence_result(
    tool_name: str,
    output: str,
    *,
    excerpt_chars: int = EVIDENCE_EXCERPT_CHARS,
    required_cutoff: str | None = None,
) -> EvidenceIngestionResult:
    if tool_name not in {"web_fetch", "pdf_fetch"}:
        return EvidenceIngestionResult(None)
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        return _rejected("invalid_json", "fetch result is not valid JSON")
    if not isinstance(payload, dict):
        return _rejected("invalid_payload", "fetch result must be a JSON object")
    if payload.get("ok") is not True:
        error = payload.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        return _rejected(
            code if isinstance(code, str) and code else "fetch_unsuccessful",
            message if isinstance(message, str) and message else "fetch was unsuccessful",
        )

    operation = payload.get("operation")
    expected_operation = "fetch" if tool_name == "web_fetch" else "pdf_fetch"
    if operation is not None:
        if not isinstance(operation, str):
            return _rejected(
                "invalid_operation", "fetch operation metadata must be a string"
            )
        if operation != expected_operation:
            return _rejected(
                "operation_mismatch",
                f"{tool_name} returned operation {operation!r}",
            )

    raw_url = payload.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        return _rejected("invalid_url", "fetch URL must be a non-empty string")
    try:
        canonical = canonicalize_url(raw_url)
    except (TypeError, ValueError) as error:
        return _rejected("invalid_url", str(error))

    title = payload.get("title")
    if title is not None and not isinstance(title, str):
        return _rejected("invalid_title", "fetch title must be null or a string")
    title = title.strip() if isinstance(title, str) and title.strip() else None

    normalized_required_cutoff, required_cutoff_date, error = _parse_iso_date(
        required_cutoff,
        field_name="required_cutoff",
    )
    if error is not None:
        return error
    normalized_cutoff, cutoff_date, error = _parse_iso_date(
        payload.get("cutoff"),
        field_name="cutoff",
    )
    if error is not None:
        return error
    if normalized_required_cutoff is not None:
        if normalized_cutoff is None or normalized_cutoff != normalized_required_cutoff:
            return _rejected(
                "cutoff_mismatch",
                "fetch result cutoff does not match the active research cutoff",
            )
        cutoff_date = required_cutoff_date

    published_at, published_date, error = _parse_iso_date(
        payload.get("published_at"),
        field_name="published_at",
    )
    if error is not None:
        return error
    publication_dates = payload.get("publication_dates")
    if publication_dates is not None:
        if not isinstance(publication_dates, list):
            return _rejected(
                "invalid_publication_dates",
                "publication_dates must be a list of ISO date strings",
            )
        normalized_publication_dates: list[str] = []
        for value in publication_dates:
            normalized, _, item_error = _parse_iso_date(
                value,
                field_name="publication_date",
            )
            if item_error is not None or normalized is None:
                return _rejected(
                    "invalid_publication_dates",
                    "publication_dates must contain only ISO date strings",
                )
            normalized_publication_dates.append(normalized)
        if (
            len(set(normalized_publication_dates)) != 1
            or normalized_publication_dates[0] != published_at
        ):
            return _rejected(
                "date_conflict",
                "publication_dates conflict with published_at",
            )
    date_status = payload.get("date_status")
    if date_status is not None and (
        not isinstance(date_status, str) or not date_status.strip()
    ):
        return _rejected(
            "invalid_date_status", "date_status must be null or a non-empty string"
        )
    date_status = date_status.strip() if isinstance(date_status, str) else None
    normalized_date_status = date_status.casefold() if date_status else None
    if (
        normalized_date_status in {"verified", "known"}
        and published_date is None
    ) or (normalized_date_status == "unknown" and published_date is not None):
        return _rejected(
            "date_status_conflict",
            "date_status is inconsistent with published_at",
        )
    if cutoff_date is not None and published_date is not None and published_date > cutoff_date:
        return _rejected(
            "published_after_cutoff",
            "fetch publication date is later than the active cutoff",
        )

    fragments: tuple[EvidenceFragment, ...] = ()
    if tool_name == "pdf_fetch":
        parsed_fragments, error = _pdf_fragments(payload)
        if error is not None:
            return error
        assert parsed_fragments is not None
        fragments = parsed_fragments
        content = "\n\n".join(item.content for item in fragments)
    else:
        content = payload.get("content", payload.get("text", ""))
        if not isinstance(content, str):
            return _rejected(
                "invalid_content", "fetch content must be a string"
            )

    parsed = urlsplit(canonical)
    record = EvidenceRecord(
        source_id=source_id_for_url(canonical),
        canonical_url=canonical,
        domain=parsed.hostname or "",
        title=title,
        content_excerpt=content[:min(excerpt_chars, EVIDENCE_CONTENT_CHARS_MAX)],
        published_at=published_at,
        date_status=date_status,
        cutoff=normalized_cutoff,
        tool_name=tool_name,
        content_fragments=fragments,
    )
    return EvidenceIngestionResult(record)


def evidence_record_from_result(
    tool_name: str,
    output: str,
    *,
    excerpt_chars: int = EVIDENCE_EXCERPT_CHARS,
    required_cutoff: str | None = None,
) -> EvidenceRecord | None:
    return inspect_evidence_result(
        tool_name,
        output,
        excerpt_chars=excerpt_chars,
        required_cutoff=required_cutoff,
    ).record


def attach_evidence_artifacts(
    record: EvidenceRecord,
    artifacts: list[ArtifactRef] | tuple[ArtifactRef, ...],
) -> EvidenceRecord:
    unique: list[ArtifactRef] = []
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.sha256 in seen:
            continue
        unique.append(artifact)
        seen.add(artifact.sha256)
        if len(unique) >= EVIDENCE_ARTIFACT_REFERENCES_MAX:
            break
    return replace(record, artifact_references=tuple(unique))


def registered_source_map(registry: EvidenceRegistry) -> dict[str, str]:
    return {item.canonical_url: item.source_id for item in registry.records}


def validate_research_final(
    final_text: str,
    registry: EvidenceRegistry,
    plan: ResearchPlan,
) -> list[str]:
    policy = plan.policy
    records = registry.records
    linkage = link_final_answer_sources(final_text, registered_source_map(registry))
    errors: list[str] = []

    if len(records) < policy.distinct_source_count:
        errors.append(f"read at least {policy.distinct_source_count} distinct sources")
    if len({item.domain for item in records if item.domain}) < policy.distinct_source_count:
        errors.append(
            f"use at least {policy.distinct_source_count} independent domains"
        )
    if sum(item.authoritative for item in records) < policy.authoritative_source_count:
        errors.append(
            f"use at least {policy.authoritative_source_count} authoritative source"
        )
    if not linkage["matched_source_ids"]:
        errors.append("cite fetched sources in the final answer")
    if linkage["unmatched_citations"]:
        errors.append("final answer contains unfetched citations")
    return errors


def record_research_evidence(
    run: RunContext,
    tool_name: str,
    output: str,
    *,
    output_artifact: ArtifactRef,
    raw_artifacts: list[ArtifactRef],
    span_id: str,
    registered_sources: dict[str, str],
    ingestion: EvidenceIngestionResult | None = None,
) -> None:
    if tool_name not in RESEARCH_TOOLS:
        return
    try:
        loaded = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        run.recorder.record(
            "research_result_unparseable",
            {"tool_name": tool_name, "output_artifact": output_artifact.as_dict()},
            span_id=span_id,
            agent_id=run.agent_id,
        )
        return
    payload = loaded if isinstance(loaded, dict) else {}
    if tool_name == "web_search":
        if not isinstance(loaded, dict):
            run.recorder.record(
                "research_result_unparseable",
                {
                    "tool_name": tool_name,
                    "output_artifact": output_artifact.as_dict(),
                },
                span_id=span_id,
                agent_id=run.agent_id,
            )
            return
        run.recorder.record(
            "search_result",
            {
                "ok": bool(payload.get("ok")),
                "query": payload.get("query"),
                "cutoff": payload.get("cutoff"),
                "candidate_urls": [
                    item.get("url")
                    for item in payload.get("results", [])
                    if isinstance(item, dict) and item.get("url")
                ],
                "snippet_only": True,
                "error": payload.get("error"),
            },
            span_id=span_id,
            agent_id=run.agent_id,
        )
        return
    if ingestion is None:
        ingestion = inspect_evidence_result(tool_name, output)
    if ingestion.record is None:
        run.recorder.record(
            "source_rejected",
            {
                "tool_name": tool_name,
                "url": payload.get("url")
                if isinstance(payload.get("url"), str)
                else None,
                "cutoff": payload.get("cutoff")
                if isinstance(payload.get("cutoff"), str)
                else None,
                "error": payload.get("error"),
                "reason_code": ingestion.rejection_code,
                "reason": ingestion.rejection_reason,
                "output_artifact": output_artifact.as_dict(),
            },
            span_id=span_id,
            agent_id=run.agent_id,
        )
        return
    record = ingestion.record
    registered_sources[record.canonical_url] = record.source_id
    run.recorder.record(
        "source_registered",
        {
            "source_id": record.source_id,
            "canonical_url": record.canonical_url,
            "published_at": record.published_at,
            "date_status": record.date_status,
            "cutoff": record.cutoff,
            "start_page": payload.get("start_page"),
            "end_page": payload.get("end_page"),
            "model_visible_artifact": output_artifact.as_dict(),
            "raw_artifacts": [artifact.as_dict() for artifact in raw_artifacts],
        },
        span_id=span_id,
        agent_id=run.agent_id,
    )
