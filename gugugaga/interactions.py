from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import config
from .observability import notify
from .stateio import atomic_write_text, interprocess_lock


INTERACTION_ACTIONS = frozenset({"steer", "queue", "redirect", "stop"})
INTERACTION_PHASES = frozenset(
    {"idle", "llm_running", "tool_running", "finalizing", "error", "stopped"}
)
_TARGET_PATTERN = re.compile(r"(?:main|team:[A-Za-z0-9][A-Za-z0-9_-]{0,63})\Z")
_PRIORITY = {"stop": 0, "redirect": 1, "steer": 2, "queue": 3}


@dataclass
class AgentInteraction:
    id: str
    target: str
    action: str
    content: str
    status: str
    created_at: float
    updated_at: float
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InteractionBroker:
    """Persistent, user-owned intervention queue for Main and Team Agents."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.path = self.workspace / ".gugugaga" / "agent-interactions.json"
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._items: dict[str, AgentInteraction] = {}
        self._phases: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._load()

    def _load(self) -> None:
        with interprocess_lock(self.path.with_suffix(".lock")):
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, TypeError, ValueError):
                payload = {}
        values = payload.get("items", []) if isinstance(payload, dict) else []
        for raw in values if isinstance(values, list) else []:
            try:
                item = AgentInteraction(**raw)
                self._validate_target(item.target)
                self._validate_action(item.action)
            except (TypeError, ValueError):
                continue
            # A process restart cannot retain an in-flight call. Put durable work
            # back at a safe boundary and leave completed audit rows untouched.
            if item.status in {"received", "pending", "injecting", "running"}:
                item.status = "queued" if item.action == "queue" else "pending"
            self._items[item.id] = item

    def _persist_locked(self) -> None:
        payload = {
            "version": 1,
            "items": [
                asdict(item)
                for item in sorted(
                    self._items.values(), key=lambda value: value.created_at
                )
            ],
        }
        with interprocess_lock(self.path.with_suffix(".lock")):
            atomic_write_text(
                self.path,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )

    @staticmethod
    def _validate_target(target: str) -> str:
        value = str(target or "").strip()
        if not _TARGET_PATTERN.fullmatch(value):
            raise ValueError(f"invalid interaction target: {target}")
        return value

    @staticmethod
    def _validate_action(action: str) -> str:
        value = str(action or "").strip().lower()
        if value not in INTERACTION_ACTIONS:
            raise ValueError(f"invalid interaction action: {action}")
        return value

    def set_phase(
        self,
        target: str,
        phase: str,
        *,
        task_id: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        target = self._validate_target(target)
        value = str(phase or "").strip().lower()
        if value not in INTERACTION_PHASES:
            raise ValueError(f"invalid interaction phase: {phase}")
        with self._condition:
            state = {
                "target": target,
                "phase": value,
                "task_id": task_id,
                "summary": str(summary or "").strip(),
                "updated_at": time.time(),
            }
            self._phases[target] = state
            if value in {"idle", "stopped", "error"}:
                self._cancel_events.setdefault(target, threading.Event()).clear()
            self._condition.notify_all()
        notify("agent_phase", state)
        return dict(state)

    def phase(self, target: str) -> dict[str, Any]:
        target = self._validate_target(target)
        with self._lock:
            return dict(
                self._phases.get(
                    target,
                    {
                        "target": target,
                        "phase": "idle",
                        "task_id": None,
                        "summary": "",
                        "updated_at": None,
                    },
                )
            )

    def submit(
        self,
        target: str,
        action: str,
        content: str = "",
        *,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInteraction:
        target = self._validate_target(target)
        action = self._validate_action(action)
        content = str(content or "").strip()
        if action != "stop" and not content:
            raise ValueError("interaction content is required")
        if len(content) > 20_000:
            raise ValueError("interaction content is too long")
        phase = self.phase(target)["phase"]
        if action in {"steer", "redirect"} and phase in {
            "idle",
            "stopped",
            "error",
        }:
            raise ValueError(f"cannot {action} {target} while it is {phase}")
        now = time.time()
        item = AgentInteraction(
            id=f"interaction_{uuid.uuid4().hex}",
            target=target,
            action=action,
            content=content,
            status="queued" if action == "queue" else "pending",
            created_at=now,
            updated_at=now,
            task_id=task_id,
            metadata=dict(metadata or {}),
        )
        with self._condition:
            self._items[item.id] = item
            if action == "stop":
                self._cancel_events.setdefault(target, threading.Event()).set()
            self._persist_locked()
            self._condition.notify_all()
        notify(
            "agent_interaction",
            {
                "interaction_id": item.id,
                "target": target,
                "action": action,
                "status": item.status,
                "task_id": task_id,
                "content": content,
            },
        )
        return item

    def update(
        self,
        interaction_id: str,
        status: str,
        *,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInteraction:
        with self._condition:
            item = self._items.get(interaction_id)
            if item is None:
                raise KeyError(interaction_id)
            item.status = str(status)
            item.updated_at = time.time()
            if task_id is not None:
                item.task_id = task_id
            if metadata:
                item.metadata.update(metadata)
            self._persist_locked()
            self._condition.notify_all()
            value = AgentInteraction(**asdict(item))
        notify(
            "agent_interaction",
            {
                "interaction_id": value.id,
                "target": value.target,
                "action": value.action,
                "status": value.status,
                "task_id": value.task_id,
            },
        )
        return value

    def consume(
        self,
        target: str,
        actions: Iterable[str],
        *,
        all_steers: bool = True,
    ) -> list[AgentInteraction]:
        target = self._validate_target(target)
        allowed = {self._validate_action(action) for action in actions}
        with self._condition:
            candidates = [
                item
                for item in self._items.values()
                if item.target == target
                and item.action in allowed
                and item.status in {"pending", "queued"}
            ]
            candidates.sort(
                key=lambda item: (_PRIORITY[item.action], item.created_at)
            )
            if not candidates:
                return []
            first = candidates[0]
            if first.action == "steer" and all_steers:
                selected = [item for item in candidates if item.action == "steer"]
            else:
                selected = [first]
            now = time.time()
            for item in selected:
                item.status = "injecting" if item.action != "queue" else "running"
                item.updated_at = now
            self._persist_locked()
            return [AgentInteraction(**asdict(item)) for item in selected]

    def pending(self, target: str, *actions: str) -> bool:
        target = self._validate_target(target)
        allowed = set(actions or INTERACTION_ACTIONS)
        with self._lock:
            return any(
                item.target == target
                and item.action in allowed
                and item.status in {"pending", "queued"}
                for item in self._items.values()
            )

    def cancel_event(self, target: str) -> threading.Event:
        target = self._validate_target(target)
        with self._lock:
            return self._cancel_events.setdefault(target, threading.Event())

    def clear_cancel(self, target: str) -> None:
        self.cancel_event(target).clear()

    def wait(self, target: str, timeout: float) -> bool:
        target = self._validate_target(target)
        with self._condition:
            if self.pending(target):
                return True
            self._condition.wait(max(0.0, timeout))
            return self.pending(target)

    def list(
        self,
        target: str | None = None,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if target is not None:
            target = self._validate_target(target)
        with self._lock:
            values = [
                asdict(item)
                for item in self._items.values()
                if target is None or item.target == target
            ]
        values.sort(key=lambda item: item["created_at"], reverse=True)
        return values[: max(1, min(int(limit), 2000))]


_brokers_lock = threading.RLock()
_brokers: dict[str, InteractionBroker] = {}


def interaction_broker(workspace: Path | None = None) -> InteractionBroker:
    root = Path(workspace or config.WORKDIR).resolve()
    key = str(root).casefold()
    with _brokers_lock:
        broker = _brokers.get(key)
        if broker is None:
            broker = InteractionBroker(root)
            _brokers[key] = broker
        return broker


def reset_interaction_brokers() -> None:
    with _brokers_lock:
        _brokers.clear()
