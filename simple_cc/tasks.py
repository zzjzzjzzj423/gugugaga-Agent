from __future__ import annotations

import ast
import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config


CURRENT_TODOS: list[dict] = []
_task_claim_lock = threading.RLock()


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]


def _task_path(task_id: str) -> Path:
    return config.TASKS_DIR / f"{task_id}.json"


def create_task(
    subject: str,
    description: str = "",
    blockedBy: list[str] | None = None,
) -> Task:
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject,
        description=description,
        status="pending",
        owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    config.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    return [
        Task(**json.loads(path.read_text()))
        for path in sorted(config.TASKS_DIR.glob("task_*.json"))
    ]


def get_task_json(task_id: str) -> str:
    return json.dumps(asdict(load_task(task_id)), indent=2)


def can_start(task_id: str) -> bool:
    task = load_task(task_id)
    for dependency_id in task.blockedBy:
        if not _task_path(dependency_id).exists():
            return False
        if load_task(dependency_id).status != "completed":
            return False
    return True


def _claim_task_unlocked(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if task.owner:
        return f"Task {task_id} already owned by {task.owner}"
    if not can_start(task_id):
        dependencies = [
            dependency_id
            for dependency_id in task.blockedBy
            if _task_path(dependency_id).exists()
            and load_task(dependency_id).status != "completed"
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
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def claim_task(task_id: str, owner: str = "agent") -> str:
    # Autonomous teammates share one task directory. The full read/check/write
    # transition must be indivisible between their polling threads.
    with _task_claim_lock:
        return _claim_task_unlocked(task_id, owner)


def complete_task(task_id: str) -> str:
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    unblocked = [
        candidate.subject
        for candidate in list_tasks()
        if candidate.status == "pending"
        and candidate.blockedBy
        and can_start(candidate.id)
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

def run_claim_task(task_id: str) -> str:
    try:
        return claim_task(task_id, owner="agent")
    except FileNotFoundError:
        return f"Error: task {task_id} not found"


def run_complete_task(task_id: str) -> str:
    try:
        return complete_task(task_id)
    except FileNotFoundError:
        return f"Error: task {task_id} not found"
