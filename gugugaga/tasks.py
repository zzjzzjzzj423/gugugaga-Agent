from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config
from .stateio import atomic_write_text, interprocess_lock


CURRENT_TODOS: list[dict] = []
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
    queue_position: int | None = None
    dispatch_type: str | None = None
    interventions: list[dict] = field(default_factory=list)
    interrupted_by_user: bool = False
    candidate_members: list[str] = field(default_factory=list)
    assignment_reason: str = ""
    matching_status: str = "pending"
    matching_revision: int = 1
    matching_notified_revision: int = 0
    matching_events: list[str] = field(default_factory=lambda: ["task_created"])
    mismatch_reports: list[dict] = field(default_factory=list)


def _task_path(task_id: str) -> Path:
    if not isinstance(task_id, str) or not _TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"invalid task id: {task_id}")
    return config.TASKS_DIR / f"{task_id}.json"


def _task_lock(task_id: str) -> AbstractContextManager[None]:
    """Serialize contenders for one task, across threads and local processes."""
    _task_path(task_id)
    # Keep the lock file separate from the atomically replaced JSON. Never
    # unlink it on task deletion: existing waiters must keep the same lock.
    return interprocess_lock(config.TASKS_DIR / ".locks" / f"{task_id}.lock")


@contextmanager
def _task_locks(task_ids: Iterable[str]) -> Iterator[None]:
    """Coordinate dependency reference edits in a consistent lock order."""
    ids = sorted(set(task_ids))
    for task_id in ids:
        _task_path(task_id)
    with ExitStack() as stack:
        for task_id in ids:
            stack.enter_context(_task_lock(task_id))
        yield


def _queue_lock(assignee: str) -> AbstractContextManager[None]:
    """Coordinate FIFO creation for this assignee, not other agents' tasks."""
    key = hashlib.sha256(assignee.encode("utf-8")).hexdigest()
    return interprocess_lock(config.TASKS_DIR / ".locks" / f"queue-{key}.lock")


def _create_task_record(
    subject: str, description: str, blocked_by: list[str], **fields,
) -> Task:
    for dependency_id in blocked_by:
        _task_path(dependency_id)
    while True:
        task_id = f"task_{int(time.time())}_{random.randint(0, 9999):04d}"
        if task_id in blocked_by:
            continue
        # Contention is limited to an ID collision or a shared dependency.
        # Publish a complete JSON only after checking the ID under its lock.
        with _task_locks([task_id, *blocked_by]):
            if _task_path(task_id).exists():
                continue
            task = Task(
                id=task_id, subject=subject, description=description,
                status="pending", owner=None, blockedBy=list(blocked_by), **fields,
            )
            _save_task_unlocked(task)
            return task


def create_task(
    subject: str,
    description: str = "",
    blockedBy: list[str] | None = None,
) -> Task:
    task = _create_task_record(subject, description, list(blockedBy or []))
    _notify_matching([task])
    return task


def create_queued_task(
    subject: str,
    description: str,
    assignee: str,
    *,
    interaction_id: str,
) -> Task:
    """Create one visible FIFO task reserved for a specific Team Agent."""

    value = str(assignee or "").strip()
    if not value:
        raise ValueError("assignee is required")
    title = str(subject or "").strip()[:120]
    if not title:
        raise ValueError("subject is required")
    with _queue_lock(value):
        positions = [
            task.queue_position or 0
            for task in _list_tasks_unlocked()
            if task.assignee == value
            and task.dispatch_type == "queued"
            and task.status in {"pending", "in_progress"}
        ]
        now = time.time()
        task = _create_task_record(
            subject=title,
            description=str(description or "").strip(),
            blocked_by=[],
            assignee=value,
            queue_position=max(positions, default=0) + 1,
            dispatch_type="queued",
            interventions=[
                {
                    "id": interaction_id,
                    "action": "queue",
                    "content": str(description or "").strip(),
                    "status": "task_created",
                    "created_at": now,
                }
            ],
        )
        return task


def append_task_intervention(
    task_id: str,
    *,
    interaction_id: str,
    action: str,
    content: str,
    status: str,
) -> Task:
    """Record an injected message from the owning agent's serial worker only."""
    _task_path(task_id)
    task = _load_task_unlocked(task_id)
    if task.status != "in_progress" or not task.owner:
        raise ValueError(f"task {task_id} has no executing owner")
    existing = next(
        (item for item in task.interventions if item.get("id") == interaction_id),
        None,
    )
    if existing is None:
        task.interventions.append({
            "id": interaction_id, "action": action, "content": content,
            "status": status, "created_at": time.time(),
        })
    else:
        existing["status"] = status
        existing["updated_at"] = time.time()
    _save_task_unlocked(task)
    return task


def _save_task_unlocked(task: Task) -> None:
    path = _task_path(task.id)
    atomic_write_text(path, json.dumps(asdict(task), indent=2))


def save_task(task: Task) -> None:
    """Save pending requirements while preserving newer dispatch metadata."""
    _task_path(task.id)
    if task.status != "pending" or task.owner:
        raise ValueError("use claim_task or complete_task to change execution state")
    with _task_locks([task.id, *task.blockedBy]):
        if _task_path(task.id).exists():
            previous = _load_task_unlocked(task.id)
            if previous.status != "pending" or previous.owner:
                raise ValueError("use the existing intervention flow for claimed tasks")
            if (previous.subject, previous.description, previous.blockedBy) != (
                task.subject, task.description, task.blockedBy
            ):
                # Preserve concurrent ownership/candidate decisions. Requirement
                # edits must not overwrite them from a stale Task snapshot.
                previous.subject = task.subject
                previous.description = task.description
                previous.blockedBy = list(task.blockedBy)
                if _is_matchable(previous):
                    _request_rematch_unlocked(previous, "task_requirements_changed")
            task = previous
        _save_task_unlocked(task)
    _notify_matching([task])


def _is_matchable(task: Task) -> bool:
    return task.status == "pending" and not task.owner and not task.assignee


def _notify_matching(tasks: list[Task], *, updated: bool = False) -> None:
    # Notify outside storage locks: observers may inspect tasks or team state.
    from .observability import notify
    from .teams import _lead_inbox_event

    pending = [task for task in tasks if _is_matchable(task)]
    if not pending:
        return
    if not updated:
        _lead_inbox_event.set()
    notify(
        "task_matching_updated" if updated else "task_matching_requested",
        {"task_ids": [task.id for task in pending]},
    )


def _request_rematch_unlocked(task: Task, reason: str) -> None:
    task.matching_revision += 1
    if task.matching_status != "rematch_required":
        task.matching_status = "pending"
    if reason not in task.matching_events:
        task.matching_events.append(reason)


def request_task_rematch(task_id: str, reason: str) -> Task:
    _task_path(task_id)
    with _task_lock(task_id):
        task = _load_task_unlocked(task_id)
        if not _is_matchable(task):
            return task
        _request_rematch_unlocked(task, reason)
        _save_task_unlocked(task)
    _notify_matching([task])
    return task


def request_all_task_rematches(reason: str) -> list[Task]:
    changed = []
    for snapshot in _list_tasks_unlocked():
        with _task_lock(snapshot.id):
            try:
                task = _load_task_unlocked(snapshot.id)
            except FileNotFoundError:
                continue
            if _is_matchable(task):
                _request_rematch_unlocked(task, reason)
                _save_task_unlocked(task)
                changed.append(task)
    _notify_matching(changed)
    return changed


def pending_matching_requests() -> list[dict]:
    """A durable, coalesced outbox; reads and idle polls never call a model."""
    return [
        asdict(task) for task in _list_tasks_unlocked()
        if _is_matchable(task)
        and task.matching_status != "matched"
        and task.matching_revision > task.matching_notified_revision
    ]


def ack_matching_requests(revisions: dict[str, int]) -> None:
    for task_id, revision in revisions.items():
        with _task_lock(task_id):
            try:
                task = _load_task_unlocked(task_id)
            except FileNotFoundError:
                continue
            # A late Lead acknowledgement must not write a running task. Once
            # claimed, only its serial worker can update the record.
            if task.status != "pending" or task.owner:
                continue
            task.matching_notified_revision = max(
                task.matching_notified_revision,
                min(revision, task.matching_revision),
            )
            _save_task_unlocked(task)


def set_task_candidates(
    task_id: str,
    candidate_members: list[str],
    assignment_reason: str,
    *,
    expected_revision: int,
) -> Task:
    _task_path(task_id)
    if not isinstance(candidate_members, list) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
        for name in candidate_members
    ):
        raise ValueError("candidate_members must be an array of agent names")
    if not isinstance(assignment_reason, str) or not assignment_reason.strip():
        raise ValueError("assignment_reason is required, including for empty candidates")
    if type(expected_revision) is not int:
        raise ValueError("expected_revision must be an integer")
    with _task_lock(task_id):
        task = _load_task_unlocked(task_id)
        if not _is_matchable(task):
            raise ValueError("cannot rematch a claimed or manually assigned task")
        if task.matching_revision != expected_revision:
            raise ValueError("stale matching revision; reload task and member context")
        task.candidate_members = list(dict.fromkeys(candidate_members))
        task.assignment_reason = assignment_reason.strip()
        task.matching_status = "matched"
        task.matching_notified_revision = task.matching_revision
        task.matching_events = []
        _save_task_unlocked(task)
    _notify_matching([task], updated=True)
    return task


def update_task(
    task_id: str,
    subject: str | None = None,
    description: str | None = None,
    blockedBy: list[str] | None = None,
) -> Task:
    _task_path(task_id)
    if subject is not None and (not isinstance(subject, str) or not subject.strip()):
        raise ValueError("subject is required")
    if description is not None and not isinstance(description, str):
        raise ValueError("description must be a string")
    if blockedBy is not None:
        if not isinstance(blockedBy, list):
            raise ValueError("blockedBy must be an array")
        for dependency in blockedBy:
            _task_path(dependency)
            if dependency == task_id:
                raise ValueError("a task cannot depend on itself")
    with _task_locks([task_id, *(blockedBy or [])]):
        task = _load_task_unlocked(task_id)
        if task.status != "pending" or task.owner:
            raise ValueError("use the existing intervention flow for claimed tasks")
        previous = (task.subject, task.description, task.blockedBy)
        if subject is not None:
            task.subject = subject.strip()
        if description is not None:
            task.description = description
        if blockedBy is not None:
            task.blockedBy = list(dict.fromkeys(blockedBy))
        changed = previous != (task.subject, task.description, task.blockedBy)
        if changed:
            if _is_matchable(task):
                _request_rematch_unlocked(task, "task_requirements_changed")
            _save_task_unlocked(task)
    if changed:
        _notify_matching([task])
    return task


def report_task_mismatch(
    task_id: str, owner: str, reason: str, work_summary: str = ""
) -> Task:
    _task_path(task_id)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("mismatch reason is required")
    if not isinstance(work_summary, str):
        raise ValueError("work_summary must be a string")
    task = _load_task_unlocked(task_id)
    if task.status != "in_progress" or task.owner != owner:
        raise ValueError(f"task {task_id} is not owned by {owner}")
    task.mismatch_reports.append({
        "agent": owner, "reason": reason.strip(),
        "work_summary": work_summary, "created_at": time.time(),
    })
    task.status = "pending"
    task.owner = None
    task.assignee = None
    task.matching_status = "rematch_required"
    task.interrupted_by_user = False
    _request_rematch_unlocked(task, "task_mismatch_reported")
    # This is the worker's last write for this ownership. Publish all fields
    # together before another idle worker can see and claim the pending task.
    _save_task_unlocked(task)
    _notify_matching([task])
    return task


def _load_task_unlocked(task_id: str) -> Task:
    task = Task(**json.loads(_task_path(task_id).read_text(encoding="utf-8")))
    if task.id != task_id:
        raise ValueError(f"task id mismatch: expected {task_id}, got {task.id}")
    return task


def load_task(task_id: str) -> Task:
    # Atomic replacement publishes either the old or new complete record.
    return _load_task_unlocked(task_id)


def _list_tasks_unlocked() -> list[Task]:
    result = []
    for path in sorted(config.TASKS_DIR.glob("task_*.json")):
        if not _TASK_ID_PATTERN.fullmatch(path.stem):
            continue
        try:
            result.append(_load_task_unlocked(path.stem))
        except FileNotFoundError:
            # A directory scan is not a transaction over all tasks; deletion
            # may finish after the path was enumerated.
            continue
    return result


def list_tasks() -> list[Task]:
    return _list_tasks_unlocked()


def get_task_json(task_id: str) -> str:
    return json.dumps(asdict(load_task(task_id)), indent=2)


def _can_start_unlocked(task_id: str) -> bool:
    try:
        task = _load_task_unlocked(task_id)
        for dependency_id in task.blockedBy:
            if _load_task_unlocked(dependency_id).status != "completed":
                return False
    except FileNotFoundError:
        return False
    return True


def can_start(task_id: str) -> bool:
    _task_path(task_id)
    return _can_start_unlocked(task_id)


def _claim_task_unlocked(
    task_id: str,
    owner: str = "agent",
    *,
    eligibility_check: Callable[[Task], str | None] | None = None,
) -> str:
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
    dependencies, missing = [], []
    for dependency_id in task.blockedBy:
        try:
            if _load_task_unlocked(dependency_id).status != "completed":
                dependencies.append(dependency_id)
        except FileNotFoundError:
            missing.append(dependency_id)
    if dependencies or missing:
        parts = []
        if dependencies:
            parts.append(f"blocked by: {dependencies}")
        if missing:
            parts.append(f"missing deps: {missing}")
        return "Cannot start — " + ", ".join(parts)
    if not task.assignee:
        if task.matching_status != "matched":
            return f"Task {task_id} is waiting for candidate matching"
        if owner not in task.candidate_members:
            return f"Owner {owner} is not a candidate for task {task_id}"
    if eligibility_check is not None:
        rejection = eligibility_check(task)
        if rejection:
            return rejection
    task.owner = owner
    task.status = "in_progress"
    task.interrupted_by_user = False
    _save_task_unlocked(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def claim_task(
    task_id: str,
    owner: str = "agent",
    *,
    eligibility_check: Callable[[Task], str | None] | None = None,
) -> str:
    # Each agent has a serial worker. Competing agents only serialize when
    # claiming the same task; the scan above is a guard, not an owner lock.
    _task_path(task_id)
    with _task_lock(task_id):
        return _claim_task_unlocked(
            task_id, owner, eligibility_check=eligibility_check
        )


def assign_task(task_id: str, assignee: str) -> Task:
    """Reserve one ready task for an idle Team Agent without claiming it."""

    value = str(assignee or "").strip()
    if not value:
        raise ValueError("assignee is required")
    _task_path(task_id)
    with _queue_lock(value), _task_lock(task_id):
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
                and (candidate.assignee == value or candidate.owner == value)
                and candidate.status in {"pending", "in_progress"}
            ),
            None,
        )
        if reserved is not None:
            raise ValueError(
                f"teammate {value} already has task {reserved.id}"
            )
        task.assignee = value
        _save_task_unlocked(task)
        return task


def unassign_task(task_id: str) -> Task:
    """Remove a pending reservation. Claimed work must be completed by its owner."""

    _task_path(task_id)
    with _task_lock(task_id):
        task = _load_task_unlocked(task_id)
        if task.status != "pending" or task.owner:
            raise ValueError(f"task {task_id} has already been claimed")
        if not task.assignee:
            raise ValueError(f"task {task_id} is not assigned")
        task.assignee = None
        _request_rematch_unlocked(task, "manual_assignment_removed")
        _save_task_unlocked(task)
    _notify_matching([task])
    return task


def release_task(task_id: str, *, expected_task: Task | None = None) -> Task:
    """Manually release abandoned work after its worker has stopped."""

    _task_path(task_id)
    # Unlike the worker's own interrupt, multiple administrative requests can
    # arrive together. Serialize them with claim using only this task's lock.
    with _task_lock(task_id):
        task = _load_task_unlocked(task_id)
        if expected_task is not None and task != expected_task:
            raise ValueError(f"task {task_id} changed; reload before releasing it")
        if task.status != "in_progress" or not task.owner:
            raise ValueError(f"task {task_id} is not claimed")
        task.status = "pending"
        task.owner = None
        task.assignee = None
        _request_rematch_unlocked(task, "task_released")
        _save_task_unlocked(task)
    _notify_matching([task])
    return task


def interrupt_task(task_id: str, owner: str) -> Task:
    """Release interrupted work while preserving its Team Agent reservation."""

    _task_path(task_id)
    task = _load_task_unlocked(task_id)
    if task.status != "in_progress" or task.owner != owner:
        raise ValueError(f"task {task_id} is not owned by {owner}")
    task.status = "pending"
    task.owner = None
    task.assignee = owner
    task.interrupted_by_user = True
    # The owning worker has stopped executing and makes no further writes
    # after publishing this complete pending record.
    _save_task_unlocked(task)
    return task


def delete_task(task_id: str) -> Task:
    """Delete a non-running, unreferenced task and return its final record."""

    path = _task_path(task_id)
    while True:
        snapshot = _load_task_unlocked(task_id)
        queue = snapshot.assignee if snapshot.dispatch_type == "queued" else None
        with ExitStack() as stack:
            if queue:
                stack.enter_context(_queue_lock(queue))
            with _task_lock(task_id):
                task = _load_task_unlocked(task_id)
                if (task.assignee if task.dispatch_type == "queued" else None) != queue:
                    continue
                if task.status == "in_progress":
                    raise ValueError(f"task {task_id} is running and cannot be deleted")
                dependents = [
                    candidate.id for candidate in _list_tasks_unlocked()
                    if candidate.id != task_id and task_id in candidate.blockedBy
                ]
                if dependents:
                    raise ValueError(
                        f"task {task_id} is still required by: {', '.join(dependents)}"
                    )
                path.unlink()
            if queue:
                _renumber_pending_queue(queue)
            return task


def _renumber_pending_queue(assignee: str) -> None:
    """Compact waiting positions under the queue lock; never write running work."""
    queued = sorted(
        (
            task for task in _list_tasks_unlocked()
            if task.assignee == assignee and task.dispatch_type == "queued"
            and task.status in {"pending", "in_progress"}
        ),
        key=lambda task: (task.queue_position is None, task.queue_position or 0, task.id),
    )
    position = max(
        (task.queue_position or 0 for task in queued if task.status == "in_progress"),
        default=0,
    )
    for snapshot in queued:
        with _task_lock(snapshot.id):
            try:
                task = _load_task_unlocked(snapshot.id)
            except FileNotFoundError:
                continue
            if task.assignee != assignee or task.dispatch_type != "queued":
                continue
            if task.status == "in_progress":
                position = max(position, task.queue_position or 0)
                continue
            if task.status != "pending":
                continue
            position += 1
            if task.queue_position != position:
                task.queue_position = position
                _save_task_unlocked(task)


def complete_task(task_id: str, owner: str = "agent") -> str:
    _task_path(task_id)
    task = _load_task_unlocked(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    if task.owner != owner:
        return f"Task {task_id} is owned by {task.owner}, not {owner}"
    # Completion is called by the same serial worker that owns execution.
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
