from __future__ import annotations

import contextlib
import contextvars
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


MutationObserver = Callable[[str, dict], None]


@dataclass
class _Request:
    request_id: str
    paths: frozenset[str]
    global_write: bool


_observer: contextvars.ContextVar[MutationObserver | None] = contextvars.ContextVar(
    "gugugaga_mutation_observer", default=None
)
_require_hash: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gugugaga_mutation_require_hash", default=False
)


@contextlib.contextmanager
def mutation_actor_scope(
    *, observer: MutationObserver | None = None, require_hash: bool = False
) -> Iterator[None]:
    observer_token = _observer.set(observer)
    hash_token = _require_hash.set(bool(require_hash))
    try:
        yield
    finally:
        _require_hash.reset(hash_token)
        _observer.reset(observer_token)


def mutation_requires_hash() -> bool:
    return _require_hash.get()


def emit_mutation_state(state: str, **details) -> None:
    callback = _observer.get()
    if callback is not None:
        callback(state, details)


class WorkspaceMutationCoordinator:
    """FIFO hierarchical lock: one global writer or independent path writers."""

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._queue: list[_Request] = []
        self._global_owner: str | None = None
        self._path_owners: dict[str, str] = {}

    @staticmethod
    def normalize_paths(workspace: Path, paths: list[str]) -> tuple[str, ...]:
        base = Path(workspace).resolve()
        values = []
        for raw in paths:
            target = (base / str(raw)).resolve()
            if not target.is_relative_to(base):
                raise ValueError(f"Path escapes workspace: {raw}")
            values.append(str(target).casefold())
        return tuple(sorted(set(values)))

    @staticmethod
    def _requests_conflict(left: _Request, right: _Request) -> bool:
        return (
            left.global_write
            or right.global_write
            or bool(left.paths & right.paths)
        )

    def _can_acquire(self, request: _Request) -> bool:
        if self._global_owner is not None:
            return False
        if request.global_write:
            if self._path_owners:
                return False
        elif any(path in self._path_owners for path in request.paths):
            return False
        for queued in self._queue:
            if queued is request:
                break
            if self._requests_conflict(queued, request):
                return False
        return True

    @contextlib.contextmanager
    def acquire(
        self,
        workspace: Path,
        *,
        paths: list[str] | None = None,
        global_write: bool = False,
        cancel_event: threading.Event | None = None,
        deadline: float | Callable[[], float] | None = None,
    ) -> Iterator[tuple[str, ...]]:
        normalized = self.normalize_paths(workspace, paths or [])
        request = _Request(
            request_id=f"mutation_{uuid.uuid4().hex}",
            paths=frozenset(normalized),
            global_write=bool(global_write),
        )
        emit_mutation_state(
            "waiting_resource",
            paths=list(normalized),
            global_write=request.global_write,
        )
        with self._condition:
            self._queue.append(request)
            try:
                while not self._can_acquire(request):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Cancelled while waiting for mutation lock")
                    value = deadline() if callable(deadline) else deadline
                    if value is not None and time.monotonic() >= value:
                        if cancel_event is not None:
                            cancel_event.set()
                        raise RuntimeError("Timed out while waiting for mutation lock")
                    remaining = None if value is None else max(0.0, value - time.monotonic())
                    self._condition.wait(
                        timeout=0.1 if remaining is None else min(0.1, remaining)
                    )
                self._queue.remove(request)
                if request.global_write:
                    self._global_owner = request.request_id
                else:
                    for path in request.paths:
                        self._path_owners[path] = request.request_id
            except Exception:
                if request in self._queue:
                    self._queue.remove(request)
                self._condition.notify_all()
                raise
        emit_mutation_state(
            "writing",
            paths=list(normalized),
            global_write=request.global_write,
        )
        try:
            yield normalized
        finally:
            with self._condition:
                if request.global_write and self._global_owner == request.request_id:
                    self._global_owner = None
                for path in request.paths:
                    if self._path_owners.get(path) == request.request_id:
                        self._path_owners.pop(path, None)
                self._condition.notify_all()
            emit_mutation_state(
                "released",
                paths=list(normalized),
                global_write=request.global_write,
            )


MUTATIONS = WorkspaceMutationCoordinator()
