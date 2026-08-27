from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


_TASK_ID_PATTERN = re.compile(r"task_[0-9]+_[0-9a-f]{6}\Z", re.ASCII)


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(data, encoding="utf-8")
    os.replace(temp, path)


class TodoStore:
    def __init__(self):
        self.items: list[dict] = []

    def update(self, items: list[dict]) -> str:
        normalized = []
        active_seen = False
        for raw in items:
            status = raw.get("status", "pending")
            if status not in {"pending", "in_progress", "completed"}:
                status = "pending"
            if status == "in_progress":
                if active_seen:
                    status = "pending"
                active_seen = True
            normalized.append({"content": str(raw.get("content", "")).strip(), "status": status})
        self.items = normalized
        return json.dumps(self.items, ensure_ascii=False)


@dataclass
class Task:
    id: str
    subject: str
    description: str = ""
    status: str = "pending"
    owner: str = ""
    blocked_by: list[str] = field(default_factory=list)


class TaskStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, task_id: str) -> Path:
        if not isinstance(task_id, str) or not _TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(f"invalid task id: {task_id}")
        return self.directory / f"{task_id}.json"

    def _save(self, task: Task) -> None:
        _atomic_write(self._path(task.id), json.dumps(asdict(task), ensure_ascii=False, indent=2))

    def create(self, subject: str, description: str = "", blocked_by: list[str] | None = None) -> Task:
        with self._lock:
            for dependency_id in blocked_by or []:
                self._path(dependency_id)
            task = Task(f"task_{int(time.time())}_{uuid.uuid4().hex[:6]}", subject, description, blocked_by=blocked_by or [])
            self._save(task)
            return task

    def get(self, task_id: str) -> Task:
        task = Task(**json.loads(self._path(task_id).read_text(encoding="utf-8")))
        if task.id != task_id:
            raise ValueError(f"task id mismatch: expected {task_id}, got {task.id}")
        return task

    def list(self) -> list[Task]:
        return [Task(**json.loads(path.read_text(encoding="utf-8"))) for path in sorted(self.directory.glob("task_*.json")) if _TASK_ID_PATTERN.fullmatch(path.stem)]

    def can_start(self, task: Task) -> bool:
        return all(self._path(dep).exists() and self.get(dep).status == "completed" for dep in task.blocked_by)

    def claim(self, task_id: str, owner: str) -> str:
        with self._lock:
            task = self.get(task_id)
            if task.status != "pending" or task.owner:
                return f"Task {task_id} cannot claim: status={task.status}, owner={task.owner}"
            if not self.can_start(task):
                return f"Blocked: {', '.join(task.blocked_by)}"
            task.status, task.owner = "in_progress", owner
            self._save(task)
            return f"Claimed {task.id} ({task.subject}) by {owner}"

    def complete(self, task_id: str) -> str:
        with self._lock:
            task = self.get(task_id)
            if task.status != "in_progress":
                return f"Task {task_id} cannot complete from {task.status}"
            task.status = "completed"
            self._save(task)
            unblocked = [t.id for t in self.list() if t.status == "pending" and task.id in t.blocked_by and self.can_start(t)]
            suffix = f"; unblocked: {', '.join(unblocked)}" if unblocked else ""
            return f"Completed {task.id}{suffix}"
