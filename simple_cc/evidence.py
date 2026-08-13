from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .trace import ArtifactRef, RunContext


RESEARCH_TOOLS = {"web_search", "web_fetch", "pdf_fetch"}


class CutoffMismatch(ValueError):
    pass


@dataclass(frozen=True)
class PreparedToolArguments:
    arguments: dict[str, Any]
    decision: str
    supplied_cutoff: str | None


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


def record_research_evidence(
    run: RunContext,
    tool_name: str,
    output: str,
    *,
    output_artifact: ArtifactRef,
    raw_artifacts: list[ArtifactRef],
    span_id: str,
    registered_sources: dict[str, str],
) -> None:
    if tool_name not in RESEARCH_TOOLS:
        return
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        run.recorder.record(
            "research_result_unparseable",
            {"tool_name": tool_name, "output_artifact": output_artifact.as_dict()},
            span_id=span_id,
            agent_id=run.agent_id,
        )
        return
    if tool_name == "web_search":
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
    if not payload.get("ok"):
        run.recorder.record(
            "source_rejected",
            {
                "tool_name": tool_name,
                "url": payload.get("url"),
                "cutoff": payload.get("cutoff"),
                "error": payload.get("error"),
            },
            span_id=span_id,
            agent_id=run.agent_id,
        )
        return
    url = payload.get("url")
    if not url:
        return
    canonical = canonicalize_url(url)
    source_id = source_id_for_url(canonical)
    registered_sources[canonical] = source_id
    run.recorder.record(
        "source_registered",
        {
            "source_id": source_id,
            "canonical_url": canonical,
            "published_at": payload.get("published_at"),
            "date_status": payload.get("date_status"),
            "cutoff": payload.get("cutoff"),
            "start_page": payload.get("start_page"),
            "end_page": payload.get("end_page"),
            "model_visible_artifact": output_artifact.as_dict(),
            "raw_artifacts": [artifact.as_dict() for artifact in raw_artifacts],
        },
        span_id=span_id,
        agent_id=run.agent_id,
    )
