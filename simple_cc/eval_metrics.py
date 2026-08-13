from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .trace import TERMINAL_STATUSES, read_trace_lines


@dataclass(frozen=True)
class RunMetrics:
    status: str
    trace_valid: bool
    total_duration_ms: float | None
    model_calls: int
    tool_calls: int
    searches: int
    fetches: int
    documents: int
    tool_failures: int
    repeated_query_rate: float | None
    repeated_url_rate: float | None
    independent_domains: int
    unknown_date_sources: int
    rejected_post_cutoff: int
    core_prompt_tokens: int | None
    core_completion_tokens: int | None
    all_in_prompt_tokens: int | None
    all_in_completion_tokens: int | None


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def validate_trace(run_dir: Path | str) -> tuple[bool, list[str]]:
    run_dir = Path(run_dir)
    errors: list[str] = []
    try:
        manifest = _manifest(run_dir)
    except Exception as error:
        return False, [f"manifest: {type(error).__name__}"]
    try:
        rows, incomplete = read_trace_lines(run_dir / "trajectory.jsonl")
    except Exception as error:
        return False, [f"trajectory: {type(error).__name__}"]
    if incomplete:
        errors.append("trajectory has incomplete final line")
    for expected, row in enumerate(rows, 1):
        if row.get("sequence") != expected:
            errors.append(
                f"sequence expected {expected}, got {row.get('sequence')}"
            )
        if row.get("run_id") != manifest.get("run_id"):
            errors.append("run_id mismatch")
        if row.get("task_id") != manifest.get("task_id"):
            errors.append("task_id mismatch")
        for field in (
            "schema_version",
            "event_type",
            "timestamp_utc",
            "elapsed_ms",
            "agent_id",
            "payload",
        ):
            if field not in row:
                errors.append(f"event {expected} missing {field}")
    for row in rows:
        _check_artifacts(run_dir, row.get("payload"), errors)
    if manifest.get("status") not in TERMINAL_STATUSES:
        errors.append("manifest has no terminal status")
    return not errors, errors


def _check_artifacts(run_dir: Path, value: Any, errors: list[str]) -> None:
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            path = run_dir / value["path"]
            if not path.is_file():
                errors.append(f"artifact missing: {value['path']}")
            else:
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != value["sha256"]:
                    errors.append(f"artifact hash mismatch: {value['path']}")
                if len(data) != value["size_bytes"]:
                    errors.append(f"artifact size mismatch: {value['path']}")
        for item in value.values():
            _check_artifacts(run_dir, item, errors)
    elif isinstance(value, list):
        for item in value:
            _check_artifacts(run_dir, item, errors)


def _sum_usage(rows: list[dict], kinds: set[str] | None, field: str) -> int | None:
    values = []
    for row in rows:
        if row.get("event_type") != "llm_response":
            continue
        payload = row.get("payload", {})
        if kinds is not None and payload.get("call_kind") not in kinds:
            continue
        value = payload.get("usage", {}).get(field)
        if value is None:
            return None
        values.append(int(value))
    return sum(values) if values else 0


def _repeat_rate(values: list[str]) -> float | None:
    if not values:
        return None
    normalized = [" ".join(value.lower().split()) for value in values]
    repeats = len(normalized) - len(set(normalized))
    return repeats / len(normalized)


def derive_metrics(run_dir: Path | str) -> RunMetrics:
    run_dir = Path(run_dir)
    valid, _errors = validate_trace(run_dir)
    manifest = _manifest(run_dir)
    rows, _incomplete = read_trace_lines(run_dir / "trajectory.jsonl")
    llm = [row for row in rows if row.get("event_type") == "llm_response"]
    tool_results = [row for row in rows if row.get("event_type") == "tool_result"]
    searches = [row for row in rows if row.get("event_type") == "search_result"]
    sources = [row for row in rows if row.get("event_type") == "source_registered"]
    rejected = [row for row in rows if row.get("event_type") == "source_rejected"]
    tool_names = [
        row.get("payload", {}).get("tool_name")
        for row in rows
        if row.get("event_type") == "tool_requested"
    ]
    urls = [
        row.get("payload", {}).get("canonical_url")
        for row in sources
        if row.get("payload", {}).get("canonical_url")
    ]
    domains = {
        urlsplit(row.get("payload", {}).get("canonical_url", "")).hostname
        for row in sources
    }
    domains.discard(None)
    last_elapsed = rows[-1].get("elapsed_ms") if rows else None
    return RunMetrics(
        status=manifest.get("status", "unknown"),
        trace_valid=valid,
        total_duration_ms=last_elapsed,
        model_calls=len(llm),
        tool_calls=len(tool_names),
        searches=len(searches),
        fetches=tool_names.count("web_fetch"),
        documents=tool_names.count("pdf_fetch"),
        tool_failures=sum(
            not row.get("payload", {}).get("success", False) for row in tool_results
        ),
        repeated_query_rate=_repeat_rate(
            [
                row.get("payload", {}).get("query", "")
                for row in searches
                if row.get("payload", {}).get("query")
            ]
        ),
        repeated_url_rate=_repeat_rate(urls),
        independent_domains=len(domains),
        unknown_date_sources=sum(
            row.get("payload", {}).get("date_status") == "unknown"
            for row in sources
        ),
        rejected_post_cutoff=sum(
            row.get("payload", {}).get("error", {}).get("code")
            in {"post_cutoff", "published_after_cutoff"}
            for row in rejected
        ),
        core_prompt_tokens=_sum_usage(rows, {"agent"}, "prompt_tokens"),
        core_completion_tokens=_sum_usage(
            rows, {"agent"}, "completion_tokens"
        ),
        all_in_prompt_tokens=_sum_usage(rows, None, "prompt_tokens"),
        all_in_completion_tokens=_sum_usage(rows, None, "completion_tokens"),
    )
