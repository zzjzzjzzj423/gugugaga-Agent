from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "1.0"
TERMINAL_STATUSES = {
    "completed",
    "failed",
    "max_rounds",
    "cancelled",
    "timed_out",
    "worker_crashed",
    "trace_invalid",
}
SUPERVISOR_STATUSES = {"timed_out", "worker_crashed", "trace_invalid"}
_SENSITIVE_KEYS = {
    "apikey": "api_key",
    "authorization": "authorization",
    "cookie": "cookie",
    "setcookie": "cookie",
    "password": "password",
    "secret": "configured_secret",
    "accesstoken": "configured_secret",
    "refreshtoken": "configured_secret",
    "authtoken": "configured_secret",
}


class TraceWriteError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    media_type: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunContext:
    recorder: "TraceRecorder"
    run_id: str
    task_id: str
    cutoff: str | None
    agent_id: str = "root"
    parent_span_id: str | None = None

    def child(
        self, agent_id: str, parent_span_id: str | None = None
    ) -> "RunContext":
        return replace(
            self,
            agent_id=agent_id,
            parent_span_id=parent_span_id or self.parent_span_id,
        )


_ACTIVE_RUN: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "simple_cc_active_run", default=None
)


@contextlib.contextmanager
def bind_run_context(context: RunContext) -> Iterator[RunContext]:
    token = _ACTIVE_RUN.set(context)
    try:
        yield context
    finally:
        _ACTIVE_RUN.reset(token)


def current_run_context() -> RunContext | None:
    return _ACTIVE_RUN.get()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "type"):
        result = {"type": getattr(value, "type")}
        for name in ("text", "id", "name", "input"):
            if hasattr(value, name):
                result[name] = _jsonable(getattr(value, name))
        return result
    return str(value)


def _key_reason(key: str) -> str | None:
    normalized = str(key).lower().replace("_", "").replace("-", "")
    for candidate, reason in _SENSITIVE_KEYS.items():
        if normalized == candidate or normalized.endswith(candidate):
            return reason
    return None


def redact_value(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    value = _jsonable(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            reason = _key_reason(key)
            result[key] = (
                f"[REDACTED:{reason}]"
                if reason is not None and item not in (None, "")
                else redact_value(item, secrets)
            )
        return result
    if isinstance(value, list):
        return [redact_value(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED:configured_secret]")
        return value
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class TraceRecorder:
    def __init__(
        self,
        run_dir: Path | str,
        *,
        run_id: str | None = None,
        secrets: list[str] | tuple[str, ...] = (),
        required: bool = True,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.run_id = run_id or str(uuid.uuid4())
        self.secrets = tuple(str(item) for item in secrets if item)
        self.required = required
        self.trajectory_path = self.run_dir / "trajectory.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.artifacts_dir = self.run_dir / "artifacts"
        self._lock = threading.RLock()
        self._sequence = 0
        self._started_monotonic = time.monotonic()
        self._task_id: str | None = None
        self._status: str | None = None
        self._manifest: dict[str, Any] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        *,
        task_id: str,
        question: str,
        cutoff: str | None,
        metadata: dict[str, Any],
    ) -> None:
        with self._lock:
            if self._task_id is not None:
                raise ValueError("run already started")
            self._task_id = str(task_id)
            self._status = "running"
            self._started_monotonic = time.monotonic()
            self._manifest = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "task_id": self._task_id,
                "question": question,
                "cutoff": cutoff,
                "started_at": _utc_now(),
                "ended_at": None,
                "status": "running",
                **redact_value(metadata, self.secrets),
            }
            try:
                _atomic_json(self.manifest_path, self._manifest)
            except Exception as error:
                raise TraceWriteError(str(error)) from error
            self.record(
                "run_started",
                {
                    "question": question,
                    "cutoff": cutoff,
                    "task_metadata": metadata,
                },
            )

    def _append_line(self, serialized: str) -> None:
        with self.trajectory_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        agent_id: str = "root",
    ) -> dict[str, Any]:
        with self._lock:
            if self._task_id is None:
                raise TraceWriteError("run has not started")
            self._sequence += 1
            event = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "task_id": self._task_id,
                "sequence": self._sequence,
                "event_type": str(event_type),
                "timestamp_utc": _utc_now(),
                "elapsed_ms": round(
                    (time.monotonic() - self._started_monotonic) * 1000, 3
                ),
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "agent_id": agent_id,
                "payload": redact_value(payload, self.secrets),
            }
            serialized = json.dumps(
                event, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            try:
                self._append_line(serialized)
            except Exception as error:
                self._sequence -= 1
                raise TraceWriteError(str(error)) from error
            return event

    def store_artifact(
        self,
        content: Any,
        *,
        media_type: str,
        source: str,
        suffix: str,
    ) -> ArtifactRef:
        if isinstance(content, bytes):
            data = content
        elif isinstance(content, str):
            data = redact_value(content, self.secrets).encode("utf-8")
        else:
            data = json.dumps(
                redact_value(content, self.secrets),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = self.artifacts_dir / f"{digest}{safe_suffix}"
        with self._lock:
            if not path.exists():
                fd, temp_name = tempfile.mkstemp(prefix=".artifact.", dir=self.artifacts_dir)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temp_name, path)
                except Exception as error:
                    try:
                        os.unlink(temp_name)
                    except OSError:
                        pass
                    raise TraceWriteError(str(error)) from error
        return ArtifactRef(
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=digest,
            media_type=media_type,
            size_bytes=len(data),
        )

    def finalize(self, status: str, details: dict[str, Any] | None = None) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"unknown terminal status: {status}")
        with self._lock:
            if self._status in TERMINAL_STATUSES:
                raise ValueError(f"manifest already terminal: {self._status}")
            if self._task_id is None:
                raise ValueError("run has not started")
            manifest = {
                **self._manifest,
                **redact_value(details or {}, self.secrets),
                "ended_at": _utc_now(),
                "status": status,
            }
            try:
                _atomic_json(self.manifest_path, manifest)
            except Exception as error:
                raise TraceWriteError(str(error)) from error
            self._manifest = manifest
            self._status = status


class NullTraceRecorder:
    def record(self, *args: Any, **kwargs: Any) -> None:
        return None


def read_trace_lines(path: Path | str) -> tuple[list[dict[str, Any]], bool]:
    data = Path(path).read_bytes()
    terminated = data.endswith(b"\n")
    parts = data.splitlines()
    rows: list[dict[str, Any]] = []
    incomplete = False
    for index, raw in enumerate(parts):
        try:
            rows.append(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index == len(parts) - 1 and not terminated:
                incomplete = True
                break
            raise
    return rows, incomplete


def supervisor_finalize_manifest(
    run_dir: Path | str,
    status: str,
    task_metadata: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> None:
    if status not in SUPERVISOR_STATUSES:
        raise ValueError("supervisor may only finalize failure states")
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") in TERMINAL_STATUSES:
            return
        if manifest.get("status") != "running":
            raise ValueError("supervisor found invalid manifest status")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "started_at": None,
            **redact_value(task_metadata),
        }
    manifest.update(redact_value(details or {}))
    manifest.update({"status": status, "ended_at": _utc_now()})
    _atomic_json(manifest_path, manifest)
