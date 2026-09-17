from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from .models import ConsolidationResult, EpisodeCandidate, FactCandidate


class MemoryValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_CREDENTIAL_PATTERNS = (
    re.compile(
        r"(?i)\b(?:api[_ -]?key|authorization|access[_ -]?token|"
        r"refresh[_ -]?token|password|passwd|secret|credential)\b\s*[:=]\s*\S+"
    ),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def contains_credential(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CREDENTIAL_PATTERNS)


def redact_credentials(text: str) -> str:
    value = text
    for pattern in _CREDENTIAL_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def validate_fact(subject: Any, content: Any) -> tuple[str, str]:
    if not isinstance(subject, str) or not isinstance(content, str):
        raise MemoryValidationError("schema_invalid", "subject and content must be strings")
    clean_subject = subject.strip()
    clean_content = content.strip()
    if not 1 <= len(clean_subject) <= 120:
        raise MemoryValidationError("subject_length", "subject must contain 1-120 characters")
    if not 1 <= len(clean_content) <= 1000:
        raise MemoryValidationError("content_length", "content must contain 1-1000 characters")
    if contains_credential(clean_subject) or contains_credential(clean_content):
        raise MemoryValidationError("sensitive_content", "credentials cannot be stored in memory")
    return clean_subject, clean_content


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def fact_hash(subject: str, content: str) -> str:
    value = f"{normalize_text(subject)}\n{normalize_text(content)}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _importance(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryValidationError("schema_invalid", "importance must be a number")
    if not 0 <= value <= 1:
        raise MemoryValidationError("schema_invalid", "importance must be finite and between 0 and 1")
    return float(value)


def _future_value(value: Any) -> str:
    if not isinstance(value, str):
        raise MemoryValidationError("schema_invalid", "future_value must be a string")
    cleaned = value.strip()
    if not 1 <= len(cleaned) <= 300:
        raise MemoryValidationError(
            "schema_invalid", "future_value must contain 1-300 characters"
        )
    if contains_credential(cleaned):
        raise MemoryValidationError(
            "sensitive_content", "credentials cannot be stored in future_value"
        )
    return cleaned


def _require_fields(value: Any, required: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise MemoryValidationError("schema_invalid", f"{path}: must be an object")
    missing = sorted(required - set(value))
    extra_count = len(set(value) - required)
    if missing or extra_count:
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if extra_count:
            details.append(f"unexpected fields: {extra_count}")
        raise MemoryValidationError("schema_invalid", f"{path}: {'; '.join(details)}")


def _parse_json_object(raw: str) -> Any:
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text)
    if fence is not None:
        text = fence.group(1)

    def reject_constant(_constant: str) -> None:
        # json.loads accepts NaN and Infinity by default. Find the first such
        # token outside strings so diagnostics expose a position, never text.
        tokens = re.finditer(r'"(?:\\.|[^"\\])*"|(?P<constant>-?Infinity|NaN)', text)
        position = next(match.start() for match in tokens if match.group("constant"))
        raise json.JSONDecodeError("invalid numeric constant", text, position)

    try:
        return json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise MemoryValidationError(
            "schema_invalid",
            f"output: JSON syntax error at line {error.lineno}, column {error.colno}",
        ) from None


def parse_consolidation_result(
    raw: str,
    *,
    max_facts: int = 10,
    min_importance: float = 0.8,
    max_episodes: int = 5,
    episode_min_importance: float = 0.6,
    admission_stats: dict[str, int] | None = None,
) -> ConsolidationResult:
    if not 0 <= min_importance <= 1:
        raise ValueError("min_importance must be between 0 and 1")
    if not 0 <= max_episodes <= 5:
        raise ValueError("max_episodes must be between 0 and 5")
    if not 0 <= episode_min_importance <= 1:
        raise ValueError("episode_min_importance must be between 0 and 1")
    value = _parse_json_object(raw)
    _require_fields(value, {"facts", "episodes"}, "output")
    facts_value = value["facts"]
    if not isinstance(facts_value, list):
        raise MemoryValidationError("schema_invalid", "facts: must be an array")
    if len(facts_value) > max_facts:
        raise MemoryValidationError("schema_invalid", f"facts: must contain at most {max_facts} items")
    facts_filtered_importance = 0
    facts_filtered_temporary = 0
    facts: list[FactCandidate] = []
    for index, item in enumerate(facts_value):
        path = f"facts[{index}]"
        required = {"subject", "content", "importance", "durability", "future_value"}
        _require_fields(item, required, path)
        for field in ("subject", "content"):
            if not isinstance(item[field], str):
                raise MemoryValidationError("schema_invalid", f"{path}.{field}: must be a string")
        try:
            subject, content = validate_fact(item["subject"], item["content"])
        except MemoryValidationError as error:
            field = (
                "subject" if error.code == "subject_length"
                else "content" if error.code == "content_length"
                else "subject" if contains_credential(item["subject"])
                else "content"
            )
            raise MemoryValidationError(error.code, f"{path}.{field}: {error}") from None
        try:
            importance = _importance(item["importance"])
        except MemoryValidationError as error:
            raise MemoryValidationError(error.code, f"{path}.importance: {error}") from None
        if not isinstance(item["durability"], str) or item["durability"] not in {"long_term", "temporary"}:
            raise MemoryValidationError(
                "schema_invalid", f"{path}.durability: must be long_term or temporary"
            )
        try:
            _future_value(item["future_value"])
        except MemoryValidationError as error:
            raise MemoryValidationError(error.code, f"{path}.future_value: {error}") from None
        facts_filtered_importance += int(importance < min_importance)
        facts_filtered_temporary += int(item["durability"] == "temporary")
        if importance >= min_importance and item["durability"] == "long_term":
            facts.append(FactCandidate(subject, content, importance))
    episodes_value = value["episodes"]
    if not isinstance(episodes_value, list):
        raise MemoryValidationError("schema_invalid", "episodes: must be an array")
    if len(episodes_value) > max_episodes:
        raise MemoryValidationError(
            "schema_invalid", f"episodes: must contain at most {max_episodes} items"
        )
    episodes: list[EpisodeCandidate] = []
    for index, episode_value in enumerate(episodes_value):
        path = f"episodes[{index}]"
        required = {"summary", "importance", "future_value"}
        _require_fields(episode_value, required, path)
        summary = episode_value["summary"]
        if not isinstance(summary, str):
            raise MemoryValidationError("schema_invalid", f"{path}.summary: must be a string")
        summary = summary.strip()
        if not 1 <= len(summary) <= 2000:
            raise MemoryValidationError("episode_length", f"{path}.summary: must contain 1-2000 characters")
        if contains_credential(summary):
            raise MemoryValidationError("sensitive_content", f"{path}.summary: credentials cannot be stored in an episode")
        try:
            importance = _importance(episode_value["importance"])
        except MemoryValidationError as error:
            raise MemoryValidationError(error.code, f"{path}.importance: {error}") from None
        try:
            _future_value(episode_value["future_value"])
        except MemoryValidationError as error:
            raise MemoryValidationError(error.code, f"{path}.future_value: {error}") from None
        if importance >= episode_min_importance:
            episodes.append(EpisodeCandidate(summary, importance))
    if admission_stats is not None:
        # Publish counts only after the entire response has passed validation.
        # A fact can fail both admission checks, so reason counts can overlap.
        admission_stats.update(
            facts_received=len(facts_value),
            facts_admitted=len(facts),
            facts_filtered_importance=facts_filtered_importance,
            facts_filtered_temporary=facts_filtered_temporary,
            episodes_received=len(episodes_value),
            episodes_admitted=len(episodes),
            episodes_filtered_importance=len(episodes_value) - len(episodes),
        )
    return ConsolidationResult(tuple(facts), tuple(episodes))
