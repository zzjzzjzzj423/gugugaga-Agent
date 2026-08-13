from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
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
    cost: float | None
    cost_currency: str | None
    pricing_version: str | None


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def read_trajectory(run_dir: Path | str) -> tuple[list[dict[str, Any]], bool]:
    path = Path(run_dir)
    if path.is_dir():
        path = path / "trajectory.jsonl"
    return read_trace_lines(path)


def validate_trace(run_dir: Path | str) -> tuple[bool, list[str]]:
    run_dir = Path(run_dir)
    errors: list[str] = []
    try:
        manifest = _manifest(run_dir)
    except Exception as error:
        return False, [f"manifest: {type(error).__name__}"]
    try:
        rows, incomplete = read_trajectory(run_dir)
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
    terminal = [row for row in rows if row.get("event_type") == "run_finalized"]
    if not terminal:
        errors.append("trajectory has no terminal event")
    elif terminal[-1].get("payload", {}).get("status") != manifest.get("status"):
        errors.append("terminal event status mismatch")
    return not errors, errors


def validate_completed_run(
    run_dir: Path | str,
    *,
    expected_run_id: str | None = None,
    expected_task_id: str | None = None,
) -> tuple[bool, list[str]]:
    run_dir = Path(run_dir)
    valid, errors = validate_trace(run_dir)
    try:
        manifest = _manifest(run_dir)
    except Exception:
        return False, errors
    if manifest.get("status") != "completed":
        errors.append("manifest is not completed")
    if expected_run_id is not None and manifest.get("run_id") != expected_run_id:
        errors.append("assignment run_id mismatch")
    if expected_task_id is not None and manifest.get("task_id") != expected_task_id:
        errors.append("assignment task_id mismatch")
    try:
        rows, incomplete = read_trajectory(run_dir)
    except Exception:
        return False, errors
    if incomplete:
        return False, errors
    terminal_indexes = [
        index
        for index, row in enumerate(rows)
        if row.get("event_type") == "run_finalized"
    ]
    if len(terminal_indexes) != 1 or terminal_indexes[0] != len(rows) - 1:
        errors.append("terminal event must be unique and final")
    answers = [row for row in rows if row.get("event_type") == "final_answer"]
    if len(answers) != 1:
        errors.append("completed run must have exactly one final answer event")
    answer_path = run_dir / "final_answer.txt"
    if not answer_path.is_file():
        errors.append("completed run has no final answer file")
    elif len(answers) == 1:
        try:
            answer_text = answer_path.read_text(encoding="utf-8")
        except Exception as error:
            errors.append(f"final answer unreadable: {type(error).__name__}")
        else:
            if answer_text != answers[0].get("payload", {}).get("text"):
                errors.append("final answer mismatch")
    if not any(row.get("event_type") == "run_completed" for row in rows):
        errors.append("completed run has no run_completed event")
    return valid and not errors, errors


def _check_artifacts(run_dir: Path, value: Any, errors: list[str]) -> None:
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            raw_path = value["path"]
            artifact_root = (run_dir / "artifacts").resolve()
            try:
                relative = Path(raw_path)
                if relative.is_absolute() or not relative.parts or relative.parts[0] != "artifacts":
                    raise ValueError
                path = (run_dir / relative).resolve()
                path.relative_to(artifact_root)
                current = run_dir.resolve()
                for part in relative.parts:
                    current = current / part
                    if current.is_symlink():
                        raise ValueError
            except (TypeError, ValueError, OSError):
                errors.append(f"artifact path outside artifacts: {raw_path}")
                path = None
            if path is None:
                pass
            elif not path.is_file():
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


def _calculate_cost(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    pricing: dict[str, Any] | None,
) -> tuple[float | None, str | None, str | None]:
    if not isinstance(pricing, dict):
        return None, None, None
    version = pricing.get("version")
    currency = pricing.get("currency")
    models = pricing.get("models")
    if not isinstance(version, str) or not version:
        return None, None, None
    if not isinstance(currency, str) or not currency:
        return None, None, version
    if not isinstance(models, dict):
        return None, currency, version
    total = 0.0
    for row in rows:
        if row.get("event_type") != "llm_response":
            continue
        payload = row.get("payload", {})
        model = payload.get("model") or manifest.get("model")
        usage = payload.get("usage", {})
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        rate = models.get(model) if model else None
        if prompt is None or completion is None or not isinstance(rate, dict):
            return None, currency, version
        effective_date = rate.get("effective_date")
        try:
            date.fromisoformat(effective_date)
            input_rate = float(rate["input_per_million"])
            output_rate = float(rate["output_per_million"])
        except (TypeError, ValueError, KeyError):
            return None, currency, version
        total += int(prompt) * input_rate / 1_000_000
        total += int(completion) * output_rate / 1_000_000
    return total, currency, version


def derive_metrics(
    run_dir: Path | str, *, pricing: dict[str, Any] | None = None
) -> RunMetrics:
    run_dir = Path(run_dir)
    valid, _errors = validate_trace(run_dir)
    manifest = _manifest(run_dir)
    rows, _incomplete = read_trajectory(run_dir)
    llm = [row for row in rows if row.get("event_type") == "llm_response"]
    tool_results = [row for row in rows if row.get("event_type") == "tool_result"]
    searches = [row for row in rows if row.get("event_type") == "search_result"]
    sources = [row for row in rows if row.get("event_type") == "source_registered"]
    rejected = [row for row in rows if row.get("event_type") == "source_rejected"]
    tool_errors = [row for row in rows if row.get("event_type") == "tool_error"]
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
    cost, currency, pricing_version = _calculate_cost(rows, manifest, pricing)
    return RunMetrics(
        status=manifest.get("status", "unknown"),
        trace_valid=valid,
        total_duration_ms=last_elapsed,
        model_calls=len(llm),
        tool_calls=len(tool_names),
        searches=len(searches),
        fetches=tool_names.count("web_fetch"),
        documents=tool_names.count("pdf_fetch"),
        tool_failures=len(tool_errors) + sum(
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
        cost=cost,
        cost_currency=currency,
        pricing_version=pricing_version,
    )
