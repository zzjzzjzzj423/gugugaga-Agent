from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from .models import ConsolidationResult, FactCandidate


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
    score = float(value)
    if not 0 <= score <= 1:
        raise MemoryValidationError("schema_invalid", "importance must be between 0 and 1")
    return score


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


def parse_consolidation_result(
    raw: str, *, max_facts: int = 10, min_importance: float = 0.8
) -> ConsolidationResult:
    if not 0 <= min_importance <= 1:
        raise ValueError("min_importance must be between 0 and 1")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise MemoryValidationError("schema_invalid", "consolidation output must be one JSON object") from error
    if not isinstance(value, dict) or set(value) != {"facts", "episode"}:
        raise MemoryValidationError("schema_invalid", "output keys must be exactly facts and episode")
    facts_value = value["facts"]
    if not isinstance(facts_value, list) or len(facts_value) > max_facts:
        raise MemoryValidationError("schema_invalid", f"facts must contain at most {max_facts} items")
    facts: list[FactCandidate] = []
    for item in facts_value:
        required = {"subject", "content", "importance", "durability", "future_value"}
        if not isinstance(item, dict) or set(item) != required:
            raise MemoryValidationError(
                "schema_invalid",
                "each fact must contain only subject, content, importance, durability, and future_value",
            )
        subject, content = validate_fact(item["subject"], item["content"])
        importance = _importance(item["importance"])
        if item["durability"] not in {"long_term", "temporary"}:
            raise MemoryValidationError(
                "schema_invalid", "durability must be long_term or temporary"
            )
        _future_value(item["future_value"])
        if importance >= min_importance and item["durability"] == "long_term":
            facts.append(FactCandidate(subject, content))
    episode_value = value["episode"]
    if episode_value is None:
        episode = None
    elif isinstance(episode_value, dict):
        required = {"summary", "importance", "completed", "future_value"}
        if set(episode_value) != required:
            raise MemoryValidationError(
                "schema_invalid",
                "episode must contain only summary, importance, completed, and future_value",
            )
        summary = episode_value["summary"]
        if not isinstance(summary, str):
            raise MemoryValidationError("schema_invalid", "episode summary must be a string")
        summary = summary.strip()
        if not 1 <= len(summary) <= 2000:
            raise MemoryValidationError("episode_length", "episode must contain at most 2000 characters")
        if contains_credential(summary):
            raise MemoryValidationError("sensitive_content", "credentials cannot be stored in an episode")
        importance = _importance(episode_value["importance"])
        if not isinstance(episode_value["completed"], bool):
            raise MemoryValidationError("schema_invalid", "episode completed must be a boolean")
        _future_value(episode_value["future_value"])
        episode = (
            summary
            if importance >= min_importance and episode_value["completed"]
            else None
        )
    else:
        raise MemoryValidationError("schema_invalid", "episode must be an object or null")
    return ConsolidationResult(tuple(facts), episode)
