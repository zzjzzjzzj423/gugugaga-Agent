from __future__ import annotations

import contextlib
import contextvars
import json
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Event = dict[str, Any]
Subscriber = Callable[[Event], None]

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|"
    r"password|passwd|secret|credential|private[_-]?key)",
    re.IGNORECASE,
)
_REDACTED = "[REDACTED]"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _SENSITIVE_KEY.search(key):
        return _REDACTED
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): _json_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sanitize(value: Any) -> Any:
    """Return a JSON-safe value with common credential fields redacted."""

    return _json_value(value)


class Observer:
    """Thread-safe in-process event bus used by persistence and live UIs."""

    def __init__(self):
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[Subscriber]] = {}

    def subscribe(self, event_type: str, callback: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

        def unsubscribe() -> None:
            with self._lock:
                callbacks = self._subscribers.get(event_type, [])
                if callback in callbacks:
                    callbacks.remove(callback)

        return unsubscribe

    def notify(self, event_type: str, event: Mapping[str, Any] | None = None) -> Event:
        context = _current_event_context.get()
        payload = {
            **context,
            **dict(event or {}),
            "type": event_type,
            "timestamp": _utc_now(),
        }
        immutable_for_dispatch = sanitize(payload)
        with self._lock:
            callbacks = [
                *self._subscribers.get(event_type, ()),
                *self._subscribers.get("*", ()),
            ]
        for callback in callbacks:
            try:
                callback(dict(immutable_for_dispatch))
            except Exception:
                # Observability is deliberately best effort. A broken Dashboard or
                # full disk must never change the Agent's control flow.
                continue
        return immutable_for_dispatch


_default_observer = Observer()
_current_observer: contextvars.ContextVar[Observer | None] = contextvars.ContextVar(
    "gugugaga_observer", default=None
)
_current_event_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "gugugaga_event_context", default={}
)


def set_default_observer(observer: Observer) -> None:
    global _default_observer
    _default_observer = observer


def get_observer() -> Observer:
    return _current_observer.get() or _default_observer


def current_event_context() -> dict[str, Any]:
    """Return a copy of the active trace context for structured child work."""

    return dict(_current_event_context.get())


def notify(event_type: str, event: Mapping[str, Any] | None = None) -> Event:
    return get_observer().notify(event_type, event)


@contextlib.contextmanager
def event_scope(observer: Observer | None = None, **values: Any) -> Iterator[None]:
    observer_token = None
    if observer is not None:
        observer_token = _current_observer.set(observer)
    current = _current_event_context.get()
    context_token = _current_event_context.set({**current, **values})
    try:
        yield
    finally:
        _current_event_context.reset(context_token)
        if observer_token is not None:
            _current_observer.reset(observer_token)


class JsonlWriter:
    def __init__(self):
        self._lock = threading.RLock()

    def append(self, path: Path, value: Mapping[str, Any]) -> None:
        line = json.dumps(sanitize(value), ensure_ascii=False, default=str)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


class Tracer:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._writer = JsonlWriter()

    def event(self, event: Event) -> None:
        date = datetime.now().astimezone().date().isoformat()
        self._writer.append(self.directory / f"{date}.jsonl", event)


class UsageLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._writer = JsonlWriter()

    def event(self, event: Event) -> None:
        if event.get("type") != "llm" or event.get("status") != "ok":
            return
        usage = event.get("usage") or {}
        self._writer.append(
            self.path,
            {
                "timestamp": event.get("timestamp"),
                "session_id": event.get("session_id"),
                "turn_id": event.get("turn_id"),
                "agent_type": event.get("agent_type", "main"),
                "provider": event.get("provider"),
                "model": event.get("model"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "call_type": event.get("call_type", "agent"),
            },
        )


class ChatLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def _initialize(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        meta TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_log_session "
                    "ON chat_log(session_id, id)"
                )
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(chat_log)")
                }
                additions = (
                    "is_final INTEGER NOT NULL DEFAULT 1",
                    "consolidation_status TEXT NOT NULL DEFAULT 'incomplete'",
                    "batch_id TEXT NULL",
                    "attempt_count INTEGER NOT NULL DEFAULT 0",
                    "lease_expires_at TEXT NULL",
                    "next_retry_at TEXT NULL",
                    "last_error_code TEXT NULL",
                    "completed_at TEXT NULL",
                    "consolidated_at TEXT NULL",
                )
                for definition in additions:
                    name = definition.split()[0]
                    if name not in columns:
                        connection.execute(
                            f"ALTER TABLE chat_log ADD COLUMN {definition}"
                        )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_log_consolidation "
                    "ON chat_log(consolidation_status, next_retry_at, "
                    "completed_at, turn_id)"
                )

    def append(
        self,
        *,
        session_id: str,
        turn_id: str,
        role: str,
        content: str,
        source: str,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        encoded_meta = json.dumps(sanitize(meta or {}), ensure_ascii=False, default=str)
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_log "
                "(session_id, turn_id, role, content, source, meta, created_at, "
                "is_final, consolidation_status, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    session_id,
                    turn_id,
                    role,
                    content,
                    source,
                    encoded_meta,
                    now,
                    "pending" if role == "assistant" else "incomplete",
                    now if role == "assistant" else None,
                ),
            )
            if role == "assistant":
                connection.execute(
                    """
                    UPDATE chat_log
                    SET consolidation_status='pending', completed_at=?
                    WHERE session_id=? AND turn_id=? AND is_final=1
                      AND role IN ('user', 'assistant')
                    """,
                    (now, session_id, turn_id),
                )


class TurnCapture:
    def __init__(self, turn_id: str):
        self.turn_id = turn_id
        self.iterations = 0
        self.tools: list[dict[str, Any]] = []
        self.model: str | None = None
        self.provider: str | None = None
        self.errors: list[str] = []

    def event(self, event: Event) -> None:
        if event.get("turn_id") != self.turn_id:
            return
        iteration = event.get("iteration")
        if isinstance(iteration, int):
            self.iterations = max(self.iterations, iteration)
        if event.get("type") == "llm":
            self.model = event.get("model") or self.model
            self.provider = event.get("provider") or self.provider
            if event.get("status") == "error":
                self.errors.append(str(event.get("error", "LLM call failed")))
        if event.get("type") == "tool":
            self.tools.append(
                {
                    "tool": event.get("tool"),
                    "args": event.get("args", {}),
                    "output": event.get("output"),
                    "status": event.get("status"),
                    "latency_ms": event.get("latency_ms"),
                    "agent_type": event.get("agent_type", "main"),
                }
            )

    def meta(self, latency_ms: int, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "gate": {},
            "graph": {},
            "iterations": self.iterations,
            "latency_ms": latency_ms,
            "tools": self.tools,
            "model": self.model,
            "provider": self.provider,
            "errors": self.errors,
            **dict(extra or {}),
        }


class TurnRecording:
    def __init__(
        self,
        system: "RecordingSystem",
        *,
        session_id: str,
        user_message: str,
        source: str,
    ):
        self.system = system
        self.session_id = session_id
        self.user_message = user_message
        self.source = source
        self.turn_id = f"turn_{uuid.uuid4().hex}"
        self.capture = TurnCapture(self.turn_id)
        self._started = time.monotonic()
        self._unsubscribe: Callable[[], None] | None = None
        self._scope: contextlib.AbstractContextManager | None = None
        self._finished = False

    def __enter__(self) -> "TurnRecording":
        self._unsubscribe = self.system.observer.subscribe("*", self.capture.event)
        self._scope = event_scope(
            self.system.observer,
            session_id=self.session_id,
            turn_id=self.turn_id,
            source=self.source,
            agent_type="main",
        )
        self._scope.__enter__()
        self.system._safe_chat_append(
            session_id=self.session_id,
            turn_id=self.turn_id,
            role="user",
            content=self.user_message,
            source=self.source,
        )
        notify("turn_start", {"user_message": self.user_message})
        return self

    def finish(self, reply: str, *, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
        latency_ms = round((time.monotonic() - self._started) * 1000)
        assistant_meta = self.capture.meta(latency_ms, meta)
        notify(
            "turn_end",
            {
                "reply": reply,
                "iterations": assistant_meta["iterations"],
                "latency_ms": latency_ms,
            },
        )
        self.system._safe_chat_append(
            session_id=self.session_id,
            turn_id=self.turn_id,
            role="assistant",
            content=reply,
            source=self.source,
            meta=assistant_meta,
        )
        self._finished = True
        return assistant_meta

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None and not self._finished:
            notify(
                "turn_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_ms": round((time.monotonic() - self._started) * 1000),
                },
            )
        if self._scope is not None:
            self._scope.__exit__(exc_type, exc, traceback)
        if self._unsubscribe is not None:
            self._unsubscribe()
        return False


class RecordingSystem:
    def __init__(self, state_dir: Path):
        state_dir = Path(state_dir)
        self.observer = Observer()
        self.tracer: Tracer | None = None
        self.usage: UsageLedger | None = None
        self.chat_log: ChatLog | None = None
        try:
            self.tracer = Tracer(state_dir / "traces")
            self.observer.subscribe("*", self.tracer.event)
        except Exception:
            pass
        try:
            self.usage = UsageLedger(state_dir / "usage.jsonl")
            self.observer.subscribe("llm", self.usage.event)
        except Exception:
            pass
        try:
            self.chat_log = ChatLog(state_dir / "state.db")
        except Exception:
            pass

    def start_turn(self, *, session_id: str, user_message: str, source: str) -> TurnRecording:
        return TurnRecording(
            self,
            session_id=session_id,
            user_message=user_message,
            source=source,
        )

    def _safe_chat_append(self, **values: Any) -> None:
        if self.chat_log is None:
            return
        try:
            self.chat_log.append(**values)
        except Exception:
            pass


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return None


def response_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": _usage_value(usage, "input_tokens", "prompt_tokens"),
        "output_tokens": _usage_value(usage, "output_tokens", "completion_tokens"),
    }


def record_llm_call(provider: Any, /, *args: Any, call_type: str = "agent", **kwargs: Any) -> Any:
    """Call a provider and emit one safe event without request or system contents."""

    started = time.monotonic()
    model = kwargs.get("model") or getattr(getattr(provider, "settings", None), "model", None)
    provider_name = getattr(provider, "provider_name", type(provider).__name__)
    try:
        response = provider.create(*args, **kwargs)
    except Exception as error:
        notify(
            "llm",
            {
                "provider": provider_name,
                "model": model,
                "call_type": call_type,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "latency_ms": round((time.monotonic() - started) * 1000),
            },
        )
        raise
    notify(
        "llm",
        {
            "provider": getattr(response, "provider", None) or provider_name,
            "model": getattr(response, "model", None) or model,
            "call_type": call_type,
            "status": "ok",
            "stop_reason": getattr(response, "stop_reason", None),
            "usage": response_usage(response),
            "latency_ms": round((time.monotonic() - started) * 1000),
        },
    )
    return response
