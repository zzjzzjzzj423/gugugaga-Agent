from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, replace
from datetime import date
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .pdf_research import PDF_PAGE_COUNT_MAX
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
EVIDENCE_URL_CHARS_MAX = 4096
_HOST_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
_ALLOWED_DATE_STATUSES = {"unknown", "verified"}
_MARKER_LIKE_LINE = re.compile(
    r"^---\s*(P(?:A(?:G(?:E)?)?)?|T(?:A(?:B(?:L(?:E)?)?)?)?)"
    r"(?=\s|[0-9]|$)",
    re.IGNORECASE,
)
_TABLE_MARKER_LINE = re.compile(
    r"--- TABLE ([1-9][0-9]*) (START|END) ---"
)


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
    if (
        not isinstance(url, str)
        or not url
        or any(
            character.isspace()
            or character in {"<", ">"}
            or unicodedata.category(character).startswith("C")
            for character in url
        )
    ):
        raise ValueError("not a canonicalizable HTTP URL")
    try:
        parsed = urlsplit(url)
    except ValueError as error:
        raise ValueError(f"not a canonicalizable HTTP URL: {url}") from error
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"not a canonicalizable HTTP URL: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("HTTP URL credentials are not allowed")
    scheme = parsed.scheme.lower()
    raw_host = parsed.hostname
    if "%" in raw_host:
        raise ValueError("IPv6 scope identifiers are not allowed")
    if raw_host.endswith(".."):
        raise ValueError(f"malformed HTTP URL hostname: {raw_host}")
    if raw_host.endswith("."):
        raw_host = raw_host[:-1]
    try:
        address = ipaddress.ip_address(raw_host.split("%", 1)[0])
    except ValueError:
        if re.fullmatch(r"[0-9.]+", raw_host):
            raise ValueError(f"malformed HTTP URL hostname: {raw_host}")
        try:
            host = raw_host.encode("idna").decode("ascii").lower()
        except UnicodeError as error:
            raise ValueError(f"malformed HTTP URL hostname: {raw_host}") from error
        labels = host.split(".")
        if (
            not host
            or len(host) > 253
            or any(not _HOST_LABEL.fullmatch(label) for label in labels)
        ):
            raise ValueError(f"malformed HTTP URL hostname: {raw_host}")
    else:
        host = address.compressed
        if address.version == 6:
            host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"invalid HTTP URL port: {url}") from error
    if port == 0:
        raise ValueError(f"invalid HTTP URL port: {url}")
    if port is not None and port != (443 if scheme == "https" else 80):
        host = f"{host}:{port}"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, parsed.path or "/", query, ""))


def source_id_for_url(url: str) -> str:
    canonical = canonicalize_url(url)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"src_{digest[:16]}"


_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_INVALID_CITATION_PREFIX = "invalid_http_url:sha256:"


def _citation_identity_candidate(
    answer: str,
    match: re.Match[str],
) -> tuple[str, str]:
    raw_token = match.group(0)
    if answer[max(0, match.start() - 2):match.start()] == "](":
        closing = raw_token.rfind(")")
        if closing >= 0:
            return raw_token[:closing], raw_token
    if match.start() > 0 and answer[match.start() - 1] == "<":
        closing = raw_token.find(">")
        if closing >= 0:
            return raw_token[:closing], raw_token
    return raw_token, raw_token


def _invalid_citation_marker(raw_token: str) -> str:
    digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    return f"{_INVALID_CITATION_PREFIX}{digest}"


def _canonical_citation_identity(candidate: str) -> str | None:
    if len(candidate) > EVIDENCE_URL_CHARS_MAX:
        return None
    try:
        canonical = canonicalize_url(candidate)
    except ValueError:
        return None
    if len(canonical) > EVIDENCE_URL_CHARS_MAX:
        return None
    return canonical


def link_final_answer_sources(
    final_text: str, registered_sources: dict[str, str]
) -> dict[str, list[str]]:
    cited_urls: list[str] = []
    matched: list[str] = []
    unmatched: list[str] = []
    answer = final_text or ""
    for match in _URL_PATTERN.finditer(answer):
        candidate, raw_token = _citation_identity_candidate(answer, match)
        canonical = _canonical_citation_identity(candidate)
        if canonical is None:
            marker = _invalid_citation_marker(raw_token)
            if marker not in unmatched:
                unmatched.append(marker)
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


def _normalize_evidence_text(
    content: str,
    *,
    strip_outer: bool = True,
) -> str:
    normalized = unicodedata.normalize("NFC", content)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    visible: list[str] = []
    for character in normalized:
        if character in {"\n", "\t"}:
            visible.append(character)
        elif character.isspace():
            visible.append(" ")
        elif (
            unicodedata.category(character).startswith("C")
            or not character.isprintable()
        ):
            continue
        else:
            visible.append(character)
    filtered = "".join(visible)
    if strip_outer:
        filtered = filtered.strip()
    return unicodedata.normalize("NFC", filtered)


def _marker_like_kind(line: str) -> str | None:
    match = _MARKER_LIKE_LINE.match(line.strip())
    if match is None:
        return None
    return "PAGE" if match.group(1).upper().startswith("P") else "TABLE"


def _pdf_page_body(page_number: int, content: str) -> str | None:
    start_marker = f"--- PAGE {page_number} START ---"
    end_marker = f"--- PAGE {page_number} END ---"
    normalized = _normalize_evidence_text(content, strip_outer=False)
    lines = normalized.split("\n") if normalized else []

    page_marker_indexes = [
        index
        for index, line in enumerate(lines)
        if _marker_like_kind(line) == "PAGE"
    ]
    if page_marker_indexes:
        if (
            page_marker_indexes != [0, len(lines) - 1]
            or lines[0] != start_marker
            or lines[-1] != end_marker
        ):
            return None
        lines = lines[1:-1]

    real_content: list[str] = []
    open_table: str | None = None
    next_table_number = 1
    for line in lines:
        marker_kind = _marker_like_kind(line)
        if marker_kind == "PAGE":
            return None
        if marker_kind != "TABLE":
            real_content.append(line)
            continue
        match = _TABLE_MARKER_LINE.fullmatch(line)
        if match is None:
            return None
        table_number, boundary = match.groups()
        if boundary == "START":
            if open_table is not None or table_number != str(next_table_number):
                return None
            open_table = table_number
        else:
            if open_table != table_number:
                return None
            open_table = None
            next_table_number += 1
    if open_table is not None:
        return None
    return "\n".join(real_content).strip()


def _has_substantive_evidence(content: str) -> bool:
    return any(
        unicodedata.category(character)[0] in {"L", "N", "S"}
        for character in content
    )


def _pdf_page_fragment(
    page_number: int,
    body: str,
) -> EvidenceFragment | None:
    start_marker = f"--- PAGE {page_number} START ---"
    end_marker = f"--- PAGE {page_number} END ---"
    fixed_chars = len(start_marker) + len(end_marker) + 2
    available_body_chars = EVIDENCE_PDF_FRAGMENT_CHARS_MAX - fixed_chars
    if available_body_chars < 1:
        return None
    bounded_body = body[:available_body_chars].rstrip()
    if not bounded_body or not _has_substantive_evidence(bounded_body):
        return None
    fragment = f"{start_marker}\n{bounded_body}\n{end_marker}"
    if len(fragment) > EVIDENCE_PDF_FRAGMENT_CHARS_MAX:
        return None
    return EvidenceFragment(f"pdf_page:{page_number:010d}", fragment)


def _pdf_fragments(payload: dict[str, Any]) -> tuple[
    tuple[EvidenceFragment, ...] | None,
    EvidenceIngestionResult | None,
]:
    pages = payload.get("pages")
    if (
        not isinstance(pages, list)
        or not pages
        or len(pages) > PDF_PAGE_COUNT_MAX
    ):
        return None, _rejected(
            "invalid_pdf_pages",
            "successful pdf_fetch pages must respect the handler page-count limit",
        )

    start_page = payload.get("start_page")
    end_page = payload.get("end_page")
    if (
        isinstance(start_page, bool)
        or not isinstance(start_page, int)
        or not 1 <= start_page <= sys.maxsize
        or isinstance(end_page, bool)
        or not isinstance(end_page, int)
        or not start_page <= end_page <= sys.maxsize
        or end_page - start_page + 1 != len(pages)
    ):
        return None, _rejected(
            "invalid_pdf_pages",
            "PDF pages must exactly match a handler-valid start_page/end_page range",
        )

    fragments: list[EvidenceFragment] = []
    for offset, page in enumerate(pages):
        if not isinstance(page, dict):
            return None, _rejected(
                "invalid_pdf_pages", "each PDF page must be an object"
            )
        page_number = page.get("page_number")
        content = page.get("content")
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or not 1 <= page_number <= sys.maxsize
            or page_number != start_page + offset
            or not isinstance(content, str)
        ):
            return None, _rejected(
                "invalid_pdf_pages",
                "PDF pages require ordered handler-valid page numbers and string content",
            )
        usable_content = _pdf_page_body(page_number, content)
        if usable_content is None:
            return None, _rejected(
                "invalid_pdf_pages",
                "PDF content contains malformed or mismatched PAGE markers",
            )
        if usable_content:
            fragment = _pdf_page_fragment(page_number, usable_content)
            if fragment is None:
                return None, _rejected(
                    "invalid_pdf_pages",
                    "PDF page markers and text exceed the fragment contract",
                )
            fragments.append(fragment)
    if not fragments:
        return None, _rejected(
            "invalid_pdf_pages",
            "PDF pages must contain non-empty extractable evidence",
        )
    return tuple(fragments[:EVIDENCE_PDF_FRAGMENTS_MAX]), None


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
    if not isinstance(operation, str) or operation != expected_operation:
        return _rejected(
            "invalid_operation",
            f"{tool_name} must return operation {expected_operation!r}",
        )

    raw_url = payload.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        return _rejected("invalid_url", "fetch URL must be a non-empty string")
    if len(raw_url) > EVIDENCE_URL_CHARS_MAX:
        return _rejected(
            "url_too_long",
            f"fetch URL exceeds {EVIDENCE_URL_CHARS_MAX} characters",
        )
    try:
        canonical = canonicalize_url(raw_url)
    except (TypeError, ValueError) as error:
        return _rejected("invalid_url", str(error))
    if len(canonical) > EVIDENCE_URL_CHARS_MAX:
        return _rejected(
            "url_too_long",
            f"canonical fetch URL exceeds {EVIDENCE_URL_CHARS_MAX} characters",
        )

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
    if "cutoff" not in payload:
        return _rejected(
            "invalid_cutoff", "successful fetch payload must include cutoff"
        )
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

    if "published_at" not in payload:
        return _rejected(
            "invalid_published_at",
            "successful fetch payload must include published_at",
        )
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
    if (
        not isinstance(date_status, str)
        or date_status not in _ALLOWED_DATE_STATUSES
    ):
        return _rejected(
            "invalid_date_status",
            "date_status must be exactly 'unknown' or 'verified'",
        )
    normalized_date_status = date_status
    if (
        normalized_date_status == "verified"
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
        content = payload.get("content")
        if not isinstance(content, str):
            return _rejected(
                "invalid_content", "fetch content must be a non-empty string"
            )
        content = _normalize_evidence_text(content)
        if not content:
            return _rejected(
                "invalid_content",
                "fetch content must contain usable printable evidence",
            )

    excerpt_limit = max(0, min(excerpt_chars, EVIDENCE_CONTENT_CHARS_MAX))
    content_excerpt = content[:excerpt_limit].rstrip()
    if not content_excerpt:
        return _rejected(
            "invalid_content", "bounded fetch content must contain usable evidence"
        )

    parsed = urlsplit(canonical)
    record = EvidenceRecord(
        source_id=source_id_for_url(canonical),
        canonical_url=canonical,
        domain=parsed.hostname or "",
        title=title,
        content_excerpt=content_excerpt,
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
    if (
        sum(item.authoritative is True for item in records)
        < policy.authoritative_source_count
    ):
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
        if tool_name == "web_search":
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
        loaded = {}
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
        raw_rejected_url = (
            payload.get("url")
            if isinstance(payload.get("url"), str)
            else None
        )
        run.recorder.record(
            "source_rejected",
            {
                "tool_name": tool_name,
                "url_preview": (
                    raw_rejected_url[:2048]
                    if raw_rejected_url is not None
                    else None
                ),
                "url_preview_truncated": bool(
                    raw_rejected_url is not None
                    and len(raw_rejected_url) > 2048
                ),
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
            "domain": record.domain[:255],
            "title": record.title[:512] if record.title is not None else None,
            "tool_name": record.tool_name[:64],
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
