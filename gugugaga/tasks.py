from __future__ import annotations

import ast
import json
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config
from .stateio import atomic_write_text, interprocess_lock


CURRENT_TODOS: list[dict] = []
_task_state_lock = threading.RLock()
_TASK_ID_PATTERN = re.compile(r"task_[0-9]+_[0-9]{4}\Z", re.ASCII)


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]
    assignee: str | None = None


def _task_path(task_id: str) -> Path:
    if not isinstance(task_id, str) or not _TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"invalid task id: {task_id}")
    return config.TASKS_DIR / f"{task_id}.json"


def create_task(
    subject: str,
    description: str = "",
    blockedBy: list[str] | None = None,
) -> Task:
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        for dependency_id in blockedBy or []:
            _task_path(dependency_id)
        while True:
            task_id = f"task_{int(time.time())}_{random.randint(0, 9999):04d}"
            if not _task_path(task_id).exists():
                break
        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            status="pending",
            owner=None,
            blockedBy=list(blockedBy or []),
        )
        _save_task_unlocked(task)
        return task


def _save_task_unlocked(task: Task) -> None:
    path = _task_path(task.id)
    atomic_write_text(path, json.dumps(asdict(task), indent=2))


def save_task(task: Task) -> None:
    _task_path(task.id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        _save_task_unlocked(task)


def _load_task_unlocked(task_id: str) -> Task:
    task = Task(**json.loads(_task_path(task_id).read_text()))
    if task.id != task_id:
        raise ValueError(f"task id mismatch: expected {task_id}, got {task.id}")
    return task


def load_task(task_id: str) -> Task:
    _task_path(task_id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        return _load_task_unlocked(task_id)


def _list_tasks_unlocked() -> list[Task]:
    return [
        Task(**json.loads(path.read_text()))
        for path in sorted(config.TASKS_DIR.glob("task_*.json"))
        if _TASK_ID_PATTERN.fullmatch(path.stem)
    ]


def list_tasks() -> list[Task]:
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        return _list_tasks_unlocked()


def get_task_json(task_id: str) -> str:
    _task_path(task_id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        return json.dumps(asdict(_load_task_unlocked(task_id)), indent=2)


def _can_start_unlocked(task_id: str) -> bool:
    task = _load_task_unlocked(task_id)
    for dependency_id in task.blockedBy:
        if not _task_path(dependency_id).exists():
            return False
        if _load_task_unlocked(dependency_id).status != "completed":
            return False
    return True


def can_start(task_id: str) -> bool:
    _task_path(task_id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        return _can_start_unlocked(task_id)


def _claim_task_unlocked(task_id: str, owner: str = "agent") -> str:
    task = _load_task_unlocked(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if task.owner:
        return f"Task {task_id} already owned by {task.owner}"
    if task.assignee and task.assignee != owner:
        return f"Task {task_id} is assigned to {task.assignee}, not {owner}"
    active = next(
        (
            candidate
            for candidate in _list_tasks_unlocked()
            if candidate.id != task_id
            and candidate.status == "in_progress"
            and candidate.owner == owner
        ),
        None,
    )
    if active is not None:
        return f"Owner {owner} is already working on {active.id}"
    if not _can_start_unlocked(task_id):
        dependencies = [
            dependency_id
            for dependency_id in task.blockedBy
            if _task_path(dependency_id).exists()
            and _load_task_unlocked(dependency_id).status != "completed"
        ]
        missing = [
            dependency_id
            for dependency_id in task.blockedBy
            if not _task_path(dependency_id).exists()
        ]
        parts = []
        if dependencies:
            parts.append(f"blocked by: {dependencies}")
        if missing:
            parts.append(f"missing deps: {missing}")
        return "Cannot start — " + ", ".join(parts)
    task.owner = owner
    task.assignee = owner
    task.status = "in_progress"
    _save_task_unlocked(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def claim_task(task_id: str, owner: str = "agent") -> str:
    # Autonomous teammates share one task directory. The full read/check/write
    # transition must be indivisible between their polling threads.
    _task_path(task_id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        return _claim_task_unlocked(task_id, owner)


def assign_task(task_id: str, assignee: str) -> Task:
    """Reserve one ready task for an idle Team Agent without claiming it."""

    value = str(assignee or "").strip()
    if not value:
        raise ValueError("assignee is required")
    _task_path(task_id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        task = _load_task_unlocked(task_id)
        if task.status != "pending" or task.owner:
            raise ValueError(f"task {task_id} is not pending")
        if not _can_start_unlocked(task_id):
            raise ValueError(f"task {task_id} is blocked")
        reserved = next(
            (
                candidate
                for candidate in _list_tasks_unlocked()
                if candidate.id != task_id
                and candidate.assignee == value
                and candidate.status in {"pending", "in_progress"}
            ),
            None,
        )
        if reserved is not None:
            raise ValueError(
                f"teammate {value} already has task {reserved.id}"
            )
        if task.assignee and task.assignee != value:
            raise ValueError(
                f"task {task_id} is already assigned to {task.assignee}"
            )
        task.assignee = value
        _save_task_unlocked(task)
        return task


def unassign_task(task_id: str) -> Task:
    """Remove a pending reservation. Claimed work must be completed by its owner."""

    _task_path(task_id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        task = _load_task_unlocked(task_id)
        if task.status != "pending" or task.owner:
            raise ValueError(f"task {task_id} has already been claimed")
        if not task.assignee:
            raise ValueError(f"task {task_id} is not assigned")
        task.assignee = None
        _save_task_unlocked(task)
        return task


def release_task(task_id: str) -> Task:
    """Return abandoned or incorrectly claimed work to the pending queue."""

    _task_path(task_id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        task = _load_task_unlocked(task_id)
        if task.status != "in_progress" or not task.owner:
            raise ValueError(f"task {task_id} is not claimed")
        task.status = "pending"
        task.owner = None
        task.assignee = None
        _save_task_unlocked(task)
        return task


def complete_task(task_id: str, owner: str = "agent") -> str:
    _task_path(task_id)
    with _task_state_lock, interprocess_lock(config.TASKS_DIR / ".state.lock"):
        task = _load_task_unlocked(task_id)
        if task.status != "in_progress":
            return f"Task {task_id} is {task.status}, cannot complete"
        if task.owner != owner:
            return f"Task {task_id} is owned by {task.owner}, not {owner}"
        task.status = "completed"
        _save_task_unlocked(task)
        unblocked = [
            candidate.subject
            for candidate in _list_tasks_unlocked()
            if candidate.status == "pending"
            and candidate.blockedBy
            and _can_start_unlocked(candidate.id)
        ]
        print(f"  \033[32m[complete] {task.subject} ✓\033[0m")
        message = f"Completed {task.id} ({task.subject})"
        if unblocked:
            message += f"\nUnblocked: {', '.join(unblocked)}"
        return message


def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for index, todo in enumerate(todos):
        if not isinstance(todo, dict):
            return None, f"Error: todos[{index}] must be an object"
        if "content" not in todo or "status" not in todo:
            return None, f"Error: todos[{index}] missing 'content' or 'status'"
        if todo["status"] not in ("pending", "in_progress", "completed"):
            return None, (
                f"Error: todos[{index}] has invalid status '{todo['status']}'"
            )
    return todos, None


def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    print(f"  \033[33m[todo] updated {len(CURRENT_TODOS)} item(s)\033[0m")
    return f"Updated {len(CURRENT_TODOS)} todos"


def run_create_task(
    subject: str,
    description: str = "",
    blockedBy: list[str] | None = None,
) -> str:
    task = create_task(subject, description, blockedBy)
    dependencies = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{dependencies}\033[0m")
    return f"Created {task.id}: {task.subject}{dependencies}"


def run_list_tasks() -> str:
    tasks = list_tasks()
    if not tasks:
        return "No tasks."
    return "\n".join(
        f"  {task.id}: {task.subject} [{task.status}]" for task in tasks
    )


def run_get_task(task_id: str) -> str:
    try:
        return get_task_json(task_id)
    except FileNotFoundError:
        return f"Error: task {task_id} not found"
    except ValueError as error:
        return f"Error: {error}"

def run_claim_task(task_id: str) -> str:
    try:
        return claim_task(task_id, owner="agent")
    except FileNotFoundError:
        return f"Error: task {task_id} not found"
    except ValueError as error:
        return f"Error: {error}"


def run_complete_task(task_id: str) -> str:
    try:
        return complete_task(task_id, owner="agent")
    except FileNotFoundError:
        return f"Error: task {task_id} not found"
    except ValueError as error:
        return f"Error: {error}"
