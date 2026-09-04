from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from gugugaga.memory import MemoryService, RecallResult


LOCOMO_DATE_FORMAT = "%I:%M %p on %d %B, %Y"


@dataclass(frozen=True)
class LocomoExchange:
    session_id: str
    turn_id: str
    user_content: str
    assistant_content: str
    completed_at: str
    partial: bool = False


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LoCoMo-Refined data was not found at {source}")
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{source} does not contain any records")
    return rows


def normalize_timestamp(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("LoCoMo session date_time is required")
    try:
        parsed = datetime.strptime(raw, LOCOMO_DATE_FORMAT)
    except ValueError:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def message_content(message: dict[str, Any], *, include_image_context: bool = True) -> str:
    text = str(message.get("text", "")).strip()
    lines = [text] if text else []
    if include_image_context:
        for field, label in (("blip_caption", "caption"), ("query", "query")):
            value = str(message.get(field, "") or "").strip()
            if value:
                lines.append(f"[{label}] {value}")
    return "\n".join(lines)


def session_to_exchanges(
    sample_id: str,
    session: dict[str, Any],
    *,
    include_image_context: bool = True,
) -> list[LocomoExchange]:
    messages = list(session.get("messages") or [])
    session_index = int(session.get("session_index", 0))
    timestamp = normalize_timestamp(str(session.get("date_time", "")))
    exchanges: list[LocomoExchange] = []
    for offset in range(0, len(messages), 2):
        first = messages[offset]
        second = messages[offset + 1] if offset + 1 < len(messages) else None
        first_speaker = str(first.get("speaker", "unknown")).strip() or "unknown"
        first_content = message_content(first, include_image_context=include_image_context)
        second_content = ""
        if second is not None:
            second_speaker = str(second.get("speaker", "unknown")).strip() or "unknown"
            second_text = message_content(second, include_image_context=include_image_context)
            second_content = f"{second_speaker}: {second_text}"
        exchange_index = offset // 2
        exchanges.append(
            LocomoExchange(
                session_id=sample_id,
                turn_id=f"{sample_id}:s{session_index:04d}:e{exchange_index:04d}",
                user_content=f"{first_speaker}: {first_content}",
                assistant_content=second_content,
                completed_at=timestamp,
                partial=second is None,
            )
        )
    return exchanges


def conversation_exchanges(
    conversation: dict[str, Any], *, include_image_context: bool = True
) -> Iterable[LocomoExchange]:
    sample_id = str(conversation["sample_id"])
    sessions = sorted(
        conversation.get("sessions") or [],
        key=lambda item: int(item.get("session_index", 0)),
    )
    for session in sessions:
        yield from session_to_exchanges(
            sample_id,
            session,
            include_image_context=include_image_context,
        )


class GugugagaMemoryAdapter:
    """Replay one LoCoMo conversation through the production memory boundary."""

    def __init__(
        self,
        database: Path | str,
        provider: Any,
        *,
        threshold: int = 6,
        model: str | None = None,
        timeout_seconds: int = 30,
        lease_seconds: int = 600,
        max_facts: int = 10,
        min_importance: float = 0.8,
        max_episodes: int = 5,
        episode_min_importance: float = 0.6,
        recall_token_budget: int = 2000,
        embedding_model: str | None = None,
        retrieval_candidate_limit: int = 20,
        retrieval_final_limit: int = 5,
        retrieval_min_score: float = 0.20,
        max_consolidation_attempts: int = 3,
        allow_existing: bool = False,
    ):
        if not 1 <= max_consolidation_attempts <= 10:
            raise ValueError("max_consolidation_attempts must be between 1 and 10")
        self.database = Path(database)
        database_exists = self.database.exists()
        if database_exists and not allow_existing:
            raise FileExistsError(
                f"refusing to reuse LoCoMo evaluation database {self.database}"
            )
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.threshold = threshold
        self.max_consolidation_attempts = max_consolidation_attempts
        self._service_options = {
            "model": model,
            "timeout_seconds": timeout_seconds,
            "lease_seconds": lease_seconds,
            "max_facts": max_facts,
            "min_importance": min_importance,
            "max_episodes": max_episodes,
            "episode_min_importance": episode_min_importance,
            "recall_token_budget": recall_token_budget,
            "embedding_model": embedding_model,
            "retrieval_candidate_limit": retrieval_candidate_limit,
            "retrieval_final_limit": retrieval_final_limit,
            "retrieval_min_score": retrieval_min_score,
            "start_worker": False,
        }
        self.service = self._new_service(threshold)
        self._recorded_turn_ids = self._load_recorded_turn_ids()
        self._seen_input_turn_ids: set[str] = set()
        self.exchange_count = len(self._recorded_turn_ids)
        self.partial_exchange_count = 0
        self.batch_count = self._load_consolidated_batch_count()
        self.consolidation_retry_count = 0
        self.final_flush_size = 0

    def _new_service(self, threshold: int) -> MemoryService:
        return MemoryService(
            self.database,
            self.provider,
            threshold=threshold,
            **self._service_options,
        )

    def _load_recorded_turn_ids(self) -> set[str]:
        with sqlite3.connect(self.database) as connection:
            return {
                str(row[0])
                for row in connection.execute("SELECT DISTINCT turn_id FROM chat_log")
            }

    def _load_consolidated_batch_count(self) -> int:
        with sqlite3.connect(self.database) as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM consolidation_batches WHERE status='consolidated'"
                ).fetchone()[0]
            )

    @property
    def recorded_turn_ids(self) -> frozenset[str]:
        return frozenset(self._recorded_turn_ids)

    def ingest_exchange(self, exchange: LocomoExchange) -> bool:
        if exchange.turn_id in self._recorded_turn_ids:
            if exchange.turn_id not in self._seen_input_turn_ids:
                self.partial_exchange_count += int(exchange.partial)
                self._seen_input_turn_ids.add(exchange.turn_id)
            return False
        self.service.repository.record_exchange(
            session_id=exchange.session_id,
            turn_id=exchange.turn_id,
            user_content=exchange.user_content,
            assistant_content=exchange.assistant_content,
            source="locomo_refined",
            completed_at=exchange.completed_at,
        )
        self.exchange_count += 1
        self.partial_exchange_count += int(exchange.partial)
        self._recorded_turn_ids.add(exchange.turn_id)
        self._seen_input_turn_ids.add(exchange.turn_id)
        self._drain_full_batches()
        return True

    @staticmethod
    def _failure_description(status: dict[str, Any]) -> str:
        failure = status.get("last_failure")
        if not isinstance(failure, dict):
            return "unknown consolidation failure"
        code = str(failure.get("error_code") or "unknown")
        attempts = int(failure.get("attempt_count") or 0)
        return f"{code} (provider attempt {attempts})"

    def _process_one_with_retries(
        self, service: MemoryService, *, required_pending: int
    ) -> int:
        for adapter_attempt in range(1, self.max_consolidation_attempts + 1):
            processed = service.process_pending(max_batches=1)
            if processed:
                return processed
            status = service.status()
            if int(status.get("pending", 0)) < required_pending:
                return 0
            if adapter_attempt >= self.max_consolidation_attempts:
                raise RuntimeError(
                    "consolidation failed after "
                    f"{self.max_consolidation_attempts} benchmark attempts: "
                    f"{self._failure_description(status)}"
                )
            changed = service.retry_failed()
            if changed <= 0:
                raise RuntimeError(
                    "consolidation is blocked and could not be retried: "
                    f"{self._failure_description(status)}"
                )
            self.consolidation_retry_count += 1
        return 0

    def _drain_full_batches(self) -> None:
        while int(self.service.status().get("pending", 0)) >= self.threshold:
            processed = self._process_one_with_retries(
                self.service,
                required_pending=self.threshold,
            )
            if not processed:
                break
            self.batch_count += processed

    def ingest_conversation(
        self, conversation: dict[str, Any], *, include_image_context: bool = True
    ) -> int:
        before = self.exchange_count
        for exchange in conversation_exchanges(
            conversation, include_image_context=include_image_context
        ):
            self.ingest_exchange(exchange)
        return self.exchange_count - before

    def flush(self) -> int:
        """Consolidate the final sub-threshold tail without changing production defaults."""
        self._drain_full_batches()
        pending = int(self.service.status().get("pending", 0))
        if pending >= self.threshold:
            status = self.service.status()
            raise RuntimeError(f"{pending} exchanges remain pending: {self._failure_description(status)}")
        if pending > 0:
            tail_service = self._new_service(pending)
            try:
                processed = self._process_one_with_retries(
                    tail_service,
                    required_pending=pending,
                )
            finally:
                tail_service.close()
            if processed != 1:
                raise RuntimeError(f"failed to consolidate final tail of {pending} exchanges")
            self.batch_count += processed
            self.final_flush_size = pending
        while self.service.process_index_pending():
            pass
        return pending

    def recall(self, question: str) -> RecallResult:
        return self.service.recall_for_turn(question)

    def status(self) -> dict[str, Any]:
        return {
            **self.service.status(),
            "exchange_count": self.exchange_count,
            "partial_exchange_count": self.partial_exchange_count,
            "batch_count": self.batch_count,
            "consolidation_retry_count": self.consolidation_retry_count,
            "final_flush_size": self.final_flush_size,
        }

    def close(self) -> None:
        self.service.close()
