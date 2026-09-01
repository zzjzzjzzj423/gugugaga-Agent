from __future__ import annotations

import contextlib
import json
import os
import random
import re
import threading
import time
import uuid
from contextvars import copy_context
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from . import config
from .hooks import trigger_hooks
from .models import ToolCall
from .observability import (
    current_event_context,
    event_scope,
    notify,
    record_llm_call,
)
from .permissions import PermissionPolicy
from .mutations import mutation_actor_scope
from .provider import is_context_length_error
from .stateio import atomic_write_text, interprocess_lock
from .tasks import (
    assign_task,
    can_start,
    claim_task,
    complete_task,
    get_task_json,
    list_tasks,
    load_task,
)
from .workspace import run_bash, run_read, run_write

if TYPE_CHECKING:
    from .context_modes import SessionContextCoordinator


# S15-S17 source-compatible team communication. Paths are resolved from
# config at call time so every teammate sees the selected shared workspace.
_mailbox_lock = threading.RLock()
_AGENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_RESERVED_TEAMMATE_NAMES = frozenset({"lead"})
_lead_inbox_event = threading.Event()
_LEAD_WAKE_MESSAGE_TYPES = frozenset(
    {"result", "error", "plan_approval_request"}
)


def _validate_agent_name(name: str) -> str:
    if not isinstance(name, str) or not _AGENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid agent name: {name}")
    return name


def _validate_teammate_name(name: str) -> str:
    value = _validate_agent_name(name)
    if value.casefold() in _RESERVED_TEAMMATE_NAMES:
        raise ValueError(f"reserved teammate name: {name}")
    return value


@dataclass(frozen=True)
class InboxBatch:
    batch_id: str
    agent: str
    messages: tuple[dict, ...]
    path: Path | None = field(default=None, repr=False, compare=False)


class MessageBus:
    def __init__(self):
        self._claimed: dict[str, Path] = {}

    @staticmethod
    def _lock_path(agent: str) -> Path:
        return config.MAILBOX_DIR / f".{agent}.lock"

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: dict | None = None,
    ) -> str:
        _validate_agent_name(from_agent)
        _validate_agent_name(to_agent)
        msg = {
            "id": f"msg_{uuid.uuid4().hex}",
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "ts": time.time(),
            "metadata": metadata or {},
        }
        config.MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
        inbox = config.MAILBOX_DIR / f"{to_agent}.jsonl"
        with _mailbox_lock, interprocess_lock(self._lock_path(to_agent)), inbox.open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(msg, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if to_agent.casefold() == "lead" and msg_type in _LEAD_WAKE_MESSAGE_TYPES:
            _lead_inbox_event.set()
            notify(
                "team_inbox_unread",
                {
                    "message_id": msg["id"],
                    "from_agent": from_agent,
                    "message_type": msg_type,
                    "task_id": msg["metadata"].get("task_id"),
                    "status": "unread",
                },
            )
        return msg["id"]

    def claim_inbox(self, agent: str) -> InboxBatch:
        _validate_agent_name(agent)
        config.MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
        with _mailbox_lock, interprocess_lock(self._lock_path(agent)):
            claimed_paths = set(self._claimed.values())
            recoverable = [
                path
                for path in sorted(config.MAILBOX_DIR.glob(f".{agent}.*.inflight.jsonl"))
                if path not in claimed_paths
            ]
            if recoverable:
                processing = recoverable[0]
                batch_id = processing.name.split(".")[2]
            else:
                inbox = config.MAILBOX_DIR / f"{agent}.jsonl"
                if not inbox.exists():
                    return InboxBatch("", agent, ())
                batch_id = f"batch_{uuid.uuid4().hex}"
                processing = config.MAILBOX_DIR / f".{agent}.{batch_id}.inflight.jsonl"
                os.replace(inbox, processing)

            messages: list[dict] = []
            malformed: list[str] = []
            for line in processing.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("mailbox entry is not an object")
                    messages.append(value)
                except (json.JSONDecodeError, ValueError):
                    malformed.append(line)
            if malformed:
                dead_letter = config.MAILBOX_DIR / f"{agent}.dead-letter.jsonl"
                with dead_letter.open("a", encoding="utf-8") as handle:
                    for line in malformed:
                        handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                atomic_write_text(
                    processing,
                    "".join(
                        json.dumps(message, ensure_ascii=False) + "\n"
                        for message in messages
                    ),
                )
            self._claimed[batch_id] = processing
            return InboxBatch(batch_id, agent, tuple(messages), processing)

    def ack_inbox(self, batch: InboxBatch) -> None:
        if not batch.batch_id or batch.path is None:
            return
        with _mailbox_lock, interprocess_lock(self._lock_path(batch.agent)):
            path = self._claimed.pop(batch.batch_id, batch.path)
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def nack_inbox(self, batch: InboxBatch) -> None:
        if not batch.batch_id or batch.path is None:
            return
        with _mailbox_lock, interprocess_lock(self._lock_path(batch.agent)):
            path = self._claimed.pop(batch.batch_id, batch.path)
            if not path.exists():
                return
            inbox = config.MAILBOX_DIR / f"{batch.agent}.jsonl"
            older = path.read_text(encoding="utf-8")
            newer = inbox.read_text(encoding="utf-8") if inbox.exists() else ""
            atomic_write_text(inbox, older + newer)
            path.unlink()

    @contextlib.contextmanager
    def consume(self, agent: str):
        batch = self.claim_inbox(agent)
        try:
            yield list(batch.messages)
        except BaseException:
            self.nack_inbox(batch)
            raise
        else:
            self.ack_inbox(batch)

    def read_inbox(self, agent: str) -> list[dict]:
        with self.consume(agent) as messages:
            return messages


BUS = MessageBus()
active_teammates: dict[str, bool] = {}
_teammate_lock = threading.RLock()
_teammate_threads: dict[str, threading.Thread] = {}
_teammate_stop_events: dict[str, threading.Event] = {}
_teammate_states: dict[str, dict[str, Any]] = {}
_teammate_stop_event = threading.Event()
_team_accepting = False


def _team_settings_path() -> Path:
    return config.WORKDIR / ".gugugaga" / "team-settings.json"


def _team_profiles_path() -> Path:
    return config.WORKDIR / ".gugugaga" / "team-agents.json"


def _load_teammate_profiles() -> dict[str, dict[str, Any]]:
    path = _team_profiles_path()
    with interprocess_lock(path.with_suffix(".lock")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            payload = {}
    raw_agents = payload.get("agents", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_agents, dict):
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for name, raw in raw_agents.items():
        if not isinstance(raw, dict):
            continue
        try:
            _validate_teammate_name(name)
        except ValueError:
            continue
        role = str(raw.get("role", "")).strip()
        prompt = str(raw.get("prompt", "")).strip()
        if not role or not prompt:
            continue
        profiles[name] = {
            "name": name,
            "role": role,
            "prompt": prompt,
            "created_at": raw.get("created_at"),
            "updated_at": raw.get("updated_at"),
        }
    return profiles


def _persist_teammate_profile(name: str, role: str, prompt: str) -> None:
    path = _team_profiles_path()
    with interprocess_lock(path.with_suffix(".lock")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        agents = payload.get("agents")
        if not isinstance(agents, dict):
            agents = {}
        now = time.time()
        previous = agents.get(name, {})
        agents[name] = {
            "name": name,
            "role": role,
            "prompt": prompt,
            "created_at": (
                previous.get("created_at", now)
                if isinstance(previous, dict)
                else now
            ),
            "updated_at": now,
        }
        atomic_write_text(
            path,
            json.dumps(
                {"version": 1, "agents": agents},
                ensure_ascii=False,
                indent=2,
            ),
        )


def get_team_settings() -> dict[str, Any]:
    path = _team_settings_path()
    with interprocess_lock(path.with_suffix(".lock")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            value = {}
    return {
        "auto_claim_enabled": bool(value.get("auto_claim_enabled", False)),
        "updated_at": value.get("updated_at"),
    }


def update_team_settings(auto_claim_enabled: bool) -> dict[str, Any]:
    if not isinstance(auto_claim_enabled, bool):
        raise ValueError("auto_claim_enabled must be boolean")
    value = {
        "auto_claim_enabled": auto_claim_enabled,
        "updated_at": time.time(),
    }
    path = _team_settings_path()
    with interprocess_lock(path.with_suffix(".lock")):
        atomic_write_text(
            path, json.dumps(value, ensure_ascii=False, indent=2)
        )
    return value


def _set_teammate_state(name: str, **updates: Any) -> None:
    with _teammate_lock:
        state = _teammate_states.get(name)
        if state is None:
            return
        state.update(updates)
        state["last_active_at"] = time.time()


def list_teammate_states() -> list[dict[str, Any]]:
    tasks = list_tasks()
    active_by_owner = {
        task.owner: task.id
        for task in tasks
        if task.status == "in_progress" and task.owner
    }
    reserved_by_assignee = {
        task.assignee: task.id
        for task in tasks
        if task.status == "pending" and task.assignee
    }
    profiles = _load_teammate_profiles()
    with _teammate_lock:
        values = []
        names = set(profiles) | set(_teammate_states)
        for name in names:
            state = _teammate_states.get(name) or {
                "name": name,
                "role": profiles.get(name, {}).get("role", "teammate"),
                "status": "stopped",
                "online": False,
                "current_task_id": None,
                "started_at": None,
                "last_active_at": profiles.get(name, {}).get("updated_at"),
            }
            current_task_id = active_by_owner.get(name) or reserved_by_assignee.get(name)
            status = str(state.get("status", "idle"))
            online = name in active_teammates
            if online and active_by_owner.get(name):
                status = "running"
            values.append(
                {
                    **state,
                    "name": name,
                    "role": state.get("role") or profiles.get(name, {}).get("role"),
                    "online": online,
                    "status": status,
                    "current_task_id": current_task_id,
                }
            )
    return sorted(values, key=lambda item: item["name"].casefold())


def assign_task_to_teammate(task_id: str, teammate: str) -> dict[str, Any]:
    _validate_teammate_name(teammate)
    with _teammate_lock:
        state = _teammate_states.get(teammate)
        if teammate not in active_teammates or state is None:
            raise ValueError(f"teammate '{teammate}' is offline")
        if state.get("status") != "idle":
            raise ValueError(f"teammate '{teammate}' is not idle")
        task = assign_task(task_id, teammate)
        try:
            BUS.send(
                "lead",
                teammate,
                f"Task {task.id} has been assigned to you.",
                "task_assignment",
                {"task_id": task.id},
            )
        except Exception:
            from .tasks import unassign_task

            unassign_task(task.id)
            raise
        _set_teammate_state(teammate, current_task_id=task.id)
    return asdict(task)


@dataclass(frozen=True)
class TeammateShutdownOutcome:
    stopped: bool
    live_names: tuple[str, ...]


@dataclass
class ProtocolState:
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: str
    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {}
_protocol_lock = threading.RLock()
_protocol_workspace: Path | None = None
_PROTOCOL_RETENTION_SECONDS = 24 * 60 * 60


def _protocol_path() -> Path:
    return config.MAILBOX_DIR / "protocol-requests.json"


def _prune_protocol_requests_locked(now: float | None = None) -> None:
    cutoff = (now or time.time()) - _PROTOCOL_RETENTION_SECONDS
    for request_id, state in list(pending_requests.items()):
        if state.status != "pending" and state.created_at < cutoff:
            pending_requests.pop(request_id, None)


def _ensure_protocol_state_loaded_locked() -> None:
    global _protocol_workspace
    workspace = config.MAILBOX_DIR.resolve()
    if _protocol_workspace == workspace:
        return
    pending_requests.clear()
    path = _protocol_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw.get("requests", []):
                state = ProtocolState(**item)
                pending_requests[state.request_id] = state
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            corrupt = path.with_name(
                f"{path.stem}.corrupt-{int(time.time())}{path.suffix}"
            )
            try:
                os.replace(path, corrupt)
            except OSError:
                pass
    _protocol_workspace = workspace
    _prune_protocol_requests_locked()


def _persist_protocol_requests_locked() -> None:
    config.MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
    _prune_protocol_requests_locked()
    content = json.dumps(
        {
            "version": 1,
            "requests": [
                asdict(state)
                for state in sorted(
                    pending_requests.values(), key=lambda value: value.created_at
                )
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    with interprocess_lock(config.MAILBOX_DIR / ".protocol.lock"):
        atomic_write_text(_protocol_path(), content)


def _ensure_protocol_state_loaded() -> None:
    with _protocol_lock:
        _ensure_protocol_state_loaded_locked()


def _new_request_id_locked() -> str:
    _ensure_protocol_state_loaded_locked()
    while True:
        request_id = f"req_{random.randint(0, 999999):06d}"
        if request_id not in pending_requests:
            return request_id


def new_request_id() -> str:
    with _protocol_lock:
        return _new_request_id_locked()


def _create_protocol_request(
    request_type: str,
    sender: str,
    target: str,
    payload: str,
) -> ProtocolState:
    # ID selection and reservation are one critical section. Otherwise two
    # concurrent creators can both observe the same random ID as available.
    with _protocol_lock:
        request_id = _new_request_id_locked()
        state = ProtocolState(
            request_id=request_id,
            type=request_type,
            sender=sender,
            target=target,
            status="pending",
            payload=payload,
        )
        pending_requests[request_id] = state
        _persist_protocol_requests_locked()
        return state


def match_response(
    response_type: str,
    request_id: str,
    approve: bool,
    *,
    sender: str = "",
    target: str = "",
) -> bool:
    with _protocol_lock:
        _ensure_protocol_state_loaded_locked()
        state = pending_requests.get(request_id)
        if not state or state.status != "pending":
            return False
        expected_type = {
            "shutdown": "shutdown_response",
            "plan_approval": "plan_approval_response",
        }.get(state.type)
        if response_type != expected_type:
            return False
        if sender != state.target or target != state.sender:
            return False
        state.status = "approved" if approve else "rejected"
        _persist_protocol_requests_locked()
        return True


def _route_protocol_messages(msgs: list[dict]) -> None:
    for msg in msgs:
        meta = msg.get("metadata", {})
        request_id = meta.get("request_id", "")
        msg_type = msg.get("type", "")
        if request_id and msg_type.endswith("_response"):
            match_response(
                msg_type,
                request_id,
                meta.get("approve", False),
                sender=msg.get("from", ""),
                target=msg.get("to", ""),
            )


def _lead_inbox_has_pending() -> bool:
    if (config.MAILBOX_DIR / "lead.jsonl").exists():
        return True
    claimed = set(getattr(BUS, "_claimed", {}).values())
    return any(
        path not in claimed
        for path in config.MAILBOX_DIR.glob(".lead.*.inflight.jsonl")
    )


def signal_pending_lead_inbox() -> bool:
    pending = _lead_inbox_has_pending()
    if pending:
        _lead_inbox_event.set()
    return pending


def wait_for_lead_inbox(
    stop_event: threading.Event | None = None,
    timeout: float = 1.0,
) -> bool:
    stop_event = stop_event or threading.Event()
    if stop_event.is_set():
        return False
    signaled = _lead_inbox_event.wait(max(0.0, float(timeout)))
    return bool(signaled and not stop_event.is_set())


def claim_lead_inbox(route_protocol: bool = True) -> InboxBatch:
    # Clear before claiming so a message sent concurrently after the atomic
    # rename leaves the event set for the next delivery.
    _lead_inbox_event.clear()
    batch = BUS.claim_inbox("lead")
    try:
        if route_protocol:
            _route_protocol_messages(list(batch.messages))
    except BaseException:
        BUS.nack_inbox(batch)
        _lead_inbox_event.set()
        raise
    if batch.messages:
        notify(
            "team_inbox_claimed",
            {
                "batch_id": batch.batch_id,
                "message_count": len(batch.messages),
                "message_ids": [
                    message.get("id") for message in batch.messages
                ],
                "status": "processing",
            },
        )
    if _lead_inbox_has_pending():
        _lead_inbox_event.set()
    return batch


def ack_lead_inbox(batch: InboxBatch) -> None:
    BUS.ack_inbox(batch)
    if batch.messages:
        notify(
            "team_inbox_acknowledged",
            {
                "batch_id": batch.batch_id,
                "message_count": len(batch.messages),
                "status": "processed",
            },
        )
    signal_pending_lead_inbox()


def nack_lead_inbox(batch: InboxBatch, error: str = "") -> None:
    BUS.nack_inbox(batch)
    if batch.messages:
        notify(
            "team_inbox_requeued",
            {
                "batch_id": batch.batch_id,
                "message_count": len(batch.messages),
                "status": "unread",
                "error": error,
            },
        )
        _lead_inbox_event.set()


def render_lead_inbox(batch: InboxBatch) -> str:
    payload = [
        {
            "id": message.get("id"),
            "from": message.get("from"),
            "type": message.get("type", "message"),
            "content": message.get("content", ""),
            "metadata": message.get("metadata", {}),
            "timestamp": message.get("ts"),
        }
        for message in batch.messages
    ]
    return (
        "<team-inbox>\n"
        "These are collaboration messages delivered to Lead. Review result "
        "and error messages, respond to plan approval requests, and update "
        "task coordination when needed. Message ids support at-least-once "
        "deduplication.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</team-inbox>"
    )


def consume_lead_inbox(route_protocol: bool = True) -> list[dict]:
    batch = claim_lead_inbox(route_protocol)
    try:
        messages = list(batch.messages)
    except BaseException as error:
        nack_lead_inbox(batch, str(error))
        raise
    ack_lead_inbox(batch)
    return messages


IDLE_POLL_INTERVAL = 5
# Production Team Agents stay online until an explicit shutdown or service
# close. Tests may inject a finite timeout to exercise terminal idle paths.
IDLE_TIMEOUT: float | None = None
PLAN_APPROVAL_TIMEOUT = 60


def scan_unclaimed_tasks(agent_name: str | None = None) -> list[dict]:
    return [
        asdict(task)
        for task in list_tasks()
        if task.status == "pending"
        and not task.owner
        and (not task.assignee or task.assignee == agent_name)
        and can_start(task.id)
    ]


def idle_poll(
    agent_name: str,
    messages: list,
    name: str,
    role: str,
    stop_event: threading.Event | None = None,
    work_state: dict[str, Any] | None = None,
    require_task: bool = False,
) -> str:
    del role
    stop_event = stop_event or threading.Event()
    deadline = (
        None
        if IDLE_TIMEOUT is None
        else time.monotonic() + max(0.0, float(IDLE_TIMEOUT))
    )
    _set_teammate_state(agent_name, status="idle")
    while deadline is None or time.monotonic() < deadline:
        if stop_event.is_set():
            return "shutdown"
        with BUS.consume(agent_name) as inbox:
            if inbox:
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        request_id = msg.get("metadata", {}).get(
                            "request_id", ""
                        )
                        BUS.send(
                            name,
                            "lead",
                            "Shutting down.",
                            "shutdown_response",
                            {"request_id": request_id, "approve": True},
                        )
                        return "shutdown"
                    if msg.get("type") == "task_assignment":
                        task_id = str(msg.get("metadata", {}).get("task_id", ""))
                        try:
                            assigned = load_task(task_id)
                        except (FileNotFoundError, ValueError):
                            assigned = None
                        result = (
                            claim_task(task_id, agent_name)
                            if assigned is not None
                            and assigned.assignee == agent_name
                            else "assignment was cancelled or replaced"
                        )
                        if not result.startswith("Claimed"):
                            BUS.send(
                                agent_name,
                                "lead",
                                f"Could not accept assigned task {task_id}: {result}",
                                "error",
                                {"task_id": task_id},
                            )
                            continue
                        claimed = load_task(task_id)
                        if work_state is not None:
                            work_state["task_id"] = task_id
                        _set_teammate_state(
                            agent_name,
                            status="running",
                            current_task_id=task_id,
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "<assigned-task>"
                                    + json.dumps(asdict(claimed), ensure_ascii=False)
                                    + "</assigned-task>"
                                ),
                            }
                        )
                        return "work"
                messages.append(
                    {
                        "role": "user",
                        "content": "<inbox>" + json.dumps(inbox) + "</inbox>",
                    }
                )
                if not require_task or (
                    work_state is not None and work_state.get("task_id")
                ):
                    return "work"
        if not get_team_settings()["auto_claim_enabled"]:
            wait_seconds = float(IDLE_POLL_INTERVAL)
            if deadline is not None:
                wait_seconds = min(
                    wait_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            if stop_event.wait(wait_seconds):
                return "shutdown"
            continue
        unclaimed = scan_unclaimed_tasks(agent_name)
        if unclaimed:
            task_data = unclaimed[0]
            result = claim_task(task_data["id"], agent_name)
            if "Claimed" in result:
                claimed = load_task(task_data["id"])
                if work_state is not None:
                    work_state["task_id"] = claimed.id
                _set_teammate_state(
                    agent_name,
                    status="running",
                    current_task_id=claimed.id,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "<auto-claimed>"
                            + json.dumps(asdict(claimed), ensure_ascii=False)
                            + "</auto-claimed>"
                        ),
                    }
                )
                return "work"
        wait_seconds = float(IDLE_POLL_INTERVAL)
        if deadline is not None:
            wait_seconds = min(
                wait_seconds,
                max(0.0, deadline - time.monotonic()),
            )
        if stop_event.wait(wait_seconds):
            return "shutdown"
    return "timeout"


_team_provider: Any | None = None
_team_permissions = PermissionPolicy()
_team_approval_callback: Callable[[ToolCall], bool] | None = None
_team_context_parent_resolver: Callable[[], SessionContextCoordinator] | None = None
_team_max_tokens = config.DEFAULT_MAX_TOKENS
_team_max_rounds_per_burst = 10


def set_team_provider(
    provider: Any | None,
    permissions: PermissionPolicy | None = None,
    approval_callback: Callable[[ToolCall], bool] | None = None,
    *,
    context_parent_resolver: Callable[[], SessionContextCoordinator] | None = None,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    max_rounds_per_burst: int = 10,
) -> None:
    global _team_provider, _team_permissions, _team_approval_callback
    global _team_accepting, _team_context_parent_resolver
    global _team_max_tokens, _team_max_rounds_per_burst
    _team_provider = provider
    _team_permissions = permissions or PermissionPolicy()
    _team_approval_callback = approval_callback
    _team_context_parent_resolver = context_parent_resolver
    _team_max_tokens = max(1, int(max_tokens))
    _team_max_rounds_per_burst = max(1, int(max_rounds_per_burst))
    _ensure_protocol_state_loaded()
    with _teammate_lock:
        _team_accepting = provider is not None
        if provider is not None and not _teammate_threads:
            _teammate_stop_event.clear()


def _dispatch_teammate_tool(
    block, handlers: dict[str, Callable]
) -> tuple[str, str]:
    from .tools import call_tool_handler

    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked), "blocked"
    call = ToolCall(block.id, block.name, block.input)
    if not _team_permissions.approve(call, _team_approval_callback):
        return (
            f"Permission denied for tool '{block.name}'. "
            "Choose a safer approach."
        ), "denied"
    output = call_tool_handler(
        handlers.get(block.name), block.input, block.name
    )
    trigger_hooks("PostToolUse", block, output)
    rendered = str(output)
    status = (
        "error"
        if rendered.startswith(("Error:", "Unknown:", "Conflict:"))
        else "ok"
    )
    return rendered, status


def spawn_teammate_thread(
    name: str,
    role: str,
    prompt: str,
    *,
    await_assignment: bool = False,
    persist_profile: bool = True,
) -> str:
    try:
        _validate_teammate_name(name)
    except ValueError as error:
        return f"Error: {error}"
    if _team_provider is None:
        return "Error: teammate provider is not configured"
    role = str(role or "").strip()
    prompt = str(prompt or "").strip()
    if not role:
        return "Error: teammate role is required"
    if not prompt:
        return "Error: teammate prompt is required"
    with _teammate_lock:
        if not _team_accepting:
            return "Error: teammate manager is not accepting new teammates"
        if name in active_teammates:
            return f"Teammate '{name}' already exists"
        if persist_profile:
            _persist_teammate_profile(name, role, prompt)
        if not _teammate_threads:
            _teammate_stop_event.clear()
        stop_event = threading.Event()
        active_teammates[name] = True
        _teammate_stop_events[name] = stop_event
        now = time.time()
        _teammate_states[name] = {
            "name": name,
            "role": role,
            "status": "running",
            "online": True,
            "current_task_id": None,
            "started_at": now,
            "last_active_at": now,
        }

    protocol_ctx = {"waiting_plan": None, "waiting_since": None}
    system = (
        f"You are '{name}', a {role}. Use tools to complete tasks in the "
        f"shared selected workspace at {config.WORKDIR}. Before overwriting an "
        "existing file, read it with include_hash=true and pass the returned "
        "SHA-256 as expected_sha256. Declare every file a Bash command may "
        "modify in write_paths. The Task System is the authority for work "
        "dispatch. When workspace auto-claim is disabled, you may claim and "
        "execute only a task explicitly assigned to you. Do not start workspace "
        "mutations from your role description or an ordinary inbox message."
    )

    def handle_inbox_message(agent_name: str, msg: dict, messages: list):
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        request_id = meta.get("request_id", "")
        if msg_type == "shutdown_request":
            BUS.send(
                agent_name,
                "lead",
                "Shutting down.",
                "shutdown_response",
                {"request_id": request_id, "approve": True},
            )
            return True
        if msg_type == "plan_approval_response":
            approve = meta.get("approve", False)
            with _protocol_lock:
                _ensure_protocol_state_loaded_locked()
                state = pending_requests.get(request_id)
                valid = (
                    request_id == protocol_ctx["waiting_plan"]
                    and state is not None
                    and state.type == "plan_approval"
                    and state.sender == agent_name
                    and state.target == "lead"
                    and state.status
                    == ("approved" if approve else "rejected")
                    and msg.get("from") == "lead"
                    and msg.get("to") == agent_name
                )
            if not valid:
                notify(
                    "teammate_protocol_error",
                    {
                        "name": agent_name,
                        "request_id": request_id,
                        "message_type": msg_type,
                    },
                )
                return False
            protocol_ctx["waiting_plan"] = None
            protocol_ctx["waiting_since"] = None
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[Plan approved]"
                        if approve
                        else f"[Plan rejected] {msg['content']}"
                    ),
                }
            )
        return False

    def run():
        from .context_modes import (
            CompressionReason,
            ContextModeError,
            RequestContext,
            create_child_context_coordinator,
        )

        scope = event_scope(agent_type="teammate", agent_id=name)
        scope.__enter__()
        notify(
            "teammate_start",
            {"name": name, "role": role, "prompt": prompt},
        )
        messages = [{"role": "user", "content": prompt}]
        sub_tools = [
            {
                "name": "bash",
                "description": "Run a shell command.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "write_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "read_file",
                "description": "Read file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                        "include_hash": {"type": "boolean"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "expected_sha256": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "send_message",
                "description": "Send message to another agent.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["to", "content"],
                },
            },
            {
                "name": "submit_plan",
                "description": "Submit a plan for Lead approval.",
                "input_schema": {
                    "type": "object",
                    "properties": {"plan": {"type": "string"}},
                    "required": ["plan"],
                },
            },
            {
                "name": "list_tasks",
                "description": "List all tasks.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_task",
                "description": "Read the full task, including description and owner.",
                "input_schema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                },
            },
            {
                "name": "claim_task",
                "description": "Claim a pending task.",
                "input_schema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                },
            },
            {
                "name": "complete_task",
                "description": "Mark an in-progress task as completed.",
                "input_schema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                },
            },
        ]

        work_state: dict[str, Any] = {
            "task_id": None,
            "report_task_id": None,
        }

        def send_message(to: str, content: str) -> str:
            try:
                if to.casefold() != "lead":
                    _validate_teammate_name(to)
                    with _teammate_lock:
                        if to not in active_teammates:
                            return f"Error: teammate '{to}' is not active"
                else:
                    _validate_agent_name(to)
            except (AttributeError, ValueError) as error:
                return f"Error: {error}"
            BUS.send(name, to, content)
            return "Sent"

        def list_task_lines() -> str:
            current = list_tasks()
            if not current:
                return "No tasks."
            return "\n".join(
                (
                    f"  {task.id}: {task.subject} [{task.status}] "
                    f"owner={task.owner or '-'} blockedBy={task.blockedBy}\n"
                    f"    {task.description or '(no description)'}"
                )
                for task in current
            )

        def claim_owned_task(task_id: str) -> str:
            try:
                candidate = load_task(task_id)
            except (FileNotFoundError, ValueError) as error:
                return f"Error: {error}"
            if (
                not get_team_settings()["auto_claim_enabled"]
                and candidate.assignee != name
            ):
                return (
                    "Error: manual assignment required while Team auto-claim "
                    f"is disabled; task {task_id} is not assigned to {name}"
                )
            result = claim_task(task_id, owner=name)
            if result.startswith("Claimed"):
                work_state["task_id"] = task_id
            return result

        def run_task_mutation(handler: Callable, **arguments) -> str:
            if await_assignment:
                task_id = work_state.get("task_id")
                try:
                    current = load_task(task_id) if task_id else None
                except (FileNotFoundError, ValueError):
                    current = None
                if (
                    current is None
                    or current.status != "in_progress"
                    or current.owner != name
                ):
                    return (
                        "Error: no active assigned task; workspace mutations "
                        "require a Task System assignment"
                    )
            return handler(**arguments)

        def complete_owned_task(task_id: str) -> str:
            result = complete_task(task_id, owner=name)
            if result.startswith("Completed"):
                work_state["report_task_id"] = task_id
                if work_state["task_id"] == task_id:
                    work_state["task_id"] = None
            return result

        sub_handlers = {
            "bash": lambda **arguments: run_task_mutation(
                run_bash, **arguments
            ),
            "read_file": run_read,
            "write_file": lambda **arguments: run_task_mutation(
                run_write, **arguments
            ),
            "send_message": send_message,
            "list_tasks": list_task_lines,
            "get_task": get_task_json,
            "claim_task": claim_owned_task,
            "complete_task": complete_owned_task,
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
        }

        coordinator = None
        if _team_context_parent_resolver is not None:
            coordinator = create_child_context_coordinator(
                _team_context_parent_resolver(),
                agent_type="teammate",
                agent_id=name,
            )
        request_context = RequestContext(system=system, tools=sub_tools)
        latest_summary = ""
        try:
            iteration = 0
            attempted_recovery = False
            if await_assignment:
                initial_dispatch = idle_poll(
                    name,
                    messages,
                    name,
                    role,
                    stop_event,
                    work_state,
                    require_task=True,
                )
                if initial_dispatch in ("shutdown", "timeout"):
                    return
            while not stop_event.is_set():
                _set_teammate_state(name, status="running")
                if len(messages) <= 3:
                    messages.insert(
                        0,
                        {
                            "role": "user",
                            "content": (
                                f"<identity>You are '{name}', role: {role}. "
                                "Continue your work.</identity>"
                            ),
                        },
                    )
                should_shutdown = False
                burst_complete = False
                rounds_this_burst = 0
                while rounds_this_burst < _team_max_rounds_per_burst:
                    if stop_event.is_set():
                        should_shutdown = True
                        break
                    with BUS.consume(name) as inbox:
                        for msg in inbox:
                            if handle_inbox_message(name, msg, messages):
                                should_shutdown = True
                                break
                    if should_shutdown:
                        break
                    non_protocol = [
                        msg for msg in inbox if msg.get("type") == "message"
                    ]
                    if non_protocol:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "<inbox>"
                                    + json.dumps(non_protocol)
                                    + "</inbox>"
                                ),
                            }
                        )
                    if protocol_ctx["waiting_plan"]:
                        waited = time.monotonic() - protocol_ctx["waiting_since"]
                        if waited >= PLAN_APPROVAL_TIMEOUT:
                            request_id = protocol_ctx["waiting_plan"]
                            with _protocol_lock:
                                state = pending_requests.get(request_id)
                                if state is not None and state.status == "pending":
                                    state.status = "expired"
                                    _persist_protocol_requests_locked()
                            protocol_ctx["waiting_plan"] = None
                            protocol_ctx["waiting_since"] = None
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"[Plan approval timed out: {request_id}] "
                                        "Revise the plan or stop safely."
                                    ),
                                }
                            )
                        else:
                            if stop_event.wait(IDLE_POLL_INTERVAL):
                                should_shutdown = True
                                break
                            continue

                    if coordinator is not None:
                        try:
                            provider_messages = coordinator.prepare_request(
                                messages, request_context
                            )
                        except ContextModeError:
                            if attempted_recovery:
                                raise
                            provider_messages = coordinator.reactive_recover(
                                messages,
                                request_context,
                                reason=CompressionReason.STRATEGY_FAILURE_RECOVERY,
                            )
                            attempted_recovery = True
                    else:
                        provider_messages = messages
                    iteration += 1
                    rounds_this_burst += 1
                    try:
                        response = record_llm_call(
                            _team_provider,
                            model=config.MODEL or None,
                            system=system,
                            messages=provider_messages,
                            tools=sub_tools,
                            max_tokens=_team_max_tokens,
                            call_type="teammate",
                        )
                    except Exception as error:
                        if (
                            coordinator is not None
                            and is_context_length_error(error)
                            and not attempted_recovery
                        ):
                            coordinator.reactive_recover(
                                messages,
                                request_context,
                                reason=CompressionReason.PROVIDER_OVERFLOW,
                            )
                            attempted_recovery = True
                            continue
                        raise RuntimeError(
                            f"Teammate provider failed: {type(error).__name__}: {error}"
                        ) from error
                    if stop_event.is_set():
                        should_shutdown = True
                        break
                    if response.stop_reason == "max_tokens":
                        raise RuntimeError(
                            "Teammate response reached the output-token limit"
                        )
                    messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                    tool_blocks = [
                        block
                        for block in response.content
                        if getattr(block, "type", None) == "tool_use"
                    ]
                    if not tool_blocks:
                        latest_summary = "".join(
                            getattr(block, "text", "")
                            for block in response.content
                            if getattr(block, "type", None) == "text"
                        ).strip()
                        if not latest_summary:
                            raise RuntimeError(
                                "Teammate finished without a text summary"
                            )
                        result_metadata = {}
                        report_task_id = (
                            work_state["report_task_id"] or work_state["task_id"]
                        )
                        if report_task_id:
                            result_metadata["task_id"] = report_task_id
                        BUS.send(
                            name,
                            "lead",
                            latest_summary,
                            "result",
                            result_metadata,
                        )
                        work_state["report_task_id"] = None
                        burst_complete = True
                        break
                    results = []
                    plan_submitted = False
                    for block in tool_blocks:
                        if stop_event.is_set():
                            should_shutdown = True
                            break
                        tool_started = time.monotonic()
                        if plan_submitted:
                            output = (
                                "Tool not executed because this tool group already "
                                "submitted a plan for approval."
                            )
                            tool_status = "blocked"
                        else:
                            with mutation_actor_scope(require_hash=True):
                                output, tool_status = _dispatch_teammate_tool(
                                    block, sub_handlers
                                )
                        if block.name == "submit_plan" and output.startswith(
                            "Plan submitted ("
                        ):
                            match = re.search(r"\((req_\d+)\)", output)
                            protocol_ctx["waiting_plan"] = (
                                match.group(1) if match else output
                            )
                            protocol_ctx["waiting_since"] = time.monotonic()
                            plan_submitted = True
                        notify(
                            "tool",
                            {
                                "iteration": iteration,
                                "tool": block.name,
                                "args": block.input,
                                "output": output,
                                "status": tool_status,
                                "latency_ms": round(
                                    (time.monotonic() - tool_started) * 1000
                                ),
                            },
                        )
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(output),
                            }
                        )
                    if should_shutdown:
                        break
                    messages.append({"role": "user", "content": results})
                    if protocol_ctx["waiting_plan"]:
                        burst_complete = True
                        break
                if should_shutdown:
                    break
                if protocol_ctx["waiting_plan"]:
                    continue
                if not burst_complete:
                    raise RuntimeError(
                        "Teammate exceeded maximum rounds "
                        f"({_team_max_rounds_per_burst})"
                    )
                idle_result = idle_poll(
                    name,
                    messages,
                    name,
                    role,
                    stop_event,
                    work_state,
                    require_task=await_assignment,
                )
                if idle_result in ("shutdown", "timeout"):
                    break

        except Exception as error:
            notify(
                "teammate_error",
                {
                    "name": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            try:
                BUS.send(
                    name,
                    "lead",
                    f"Teammate '{name}' failed: {type(error).__name__}: {error}",
                    "error",
                )
            except Exception:
                pass
        finally:
            if coordinator is not None:
                coordinator.close()
            notify("teammate_end", {"name": name})
            scope.__exit__(None, None, None)
            with _teammate_lock:
                active_teammates.pop(name, None)
                _teammate_threads.pop(name, None)
                _teammate_stop_events.pop(name, None)
                state = _teammate_states.get(name)
                if state is not None:
                    state.update(
                        {
                            "status": "stopped",
                            "online": False,
                            "current_task_id": None,
                            "last_active_at": time.time(),
                        }
                    )

    parent_context = copy_context()
    thread = threading.Thread(
        target=lambda: parent_context.run(run),
        name=f"teammate-{name}",
        daemon=True,
    )
    with _teammate_lock:
        _teammate_threads[name] = thread
    thread.start()
    return f"Teammate '{name}' spawned as {role}"


def stop_teammate(name: str) -> str:
    """Cooperatively stop one teammate without affecting the rest of the team."""

    try:
        _validate_teammate_name(name)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        thread = _teammate_threads.get(name)
        stop_event = _teammate_stop_events.get(name)
        if (
            name not in active_teammates
            or thread is None
            or stop_event is None
        ):
            return f"Error: teammate '{name}' is not active"
        _set_teammate_state(name, status="stopping")
        stop_event.set()
    return f"Stop requested for {name}"


def restart_teammate(name: str) -> str:
    """Restart one stopped teammate from its persisted workspace profile."""

    try:
        _validate_teammate_name(name)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        if name in active_teammates:
            return f"Error: teammate '{name}' is already active"
    profile = _load_teammate_profiles().get(name)
    if profile is None:
        return f"Error: persisted profile for teammate '{name}' was not found"
    return spawn_teammate_thread(
        name,
        profile["role"],
        profile["prompt"],
        await_assignment=True,
        persist_profile=False,
    )


def stop_all_teammates(timeout: float = 5.0) -> TeammateShutdownOutcome:
    """Cancel and join all source-faithful teammate threads."""
    global _team_accepting
    with _teammate_lock:
        _team_accepting = False
    _teammate_stop_event.set()
    with _teammate_lock:
        for stop_event in _teammate_stop_events.values():
            stop_event.set()
        threads = list(_teammate_threads.items())
    deadline = time.monotonic() + timeout
    for _, thread in threads:
        if thread is threading.current_thread() or not thread.is_alive():
            continue
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    with _teammate_lock:
        for name, thread in list(_teammate_threads.items()):
            if not thread.is_alive():
                _teammate_threads.pop(name, None)
                active_teammates.pop(name, None)
                _teammate_stop_events.pop(name, None)
                state = _teammate_states.get(name)
                if state is not None:
                    state.update(
                        {
                            "status": "stopped",
                            "online": False,
                            "current_task_id": None,
                            "last_active_at": time.time(),
                        }
                    )
        live = tuple(
            name
            for name, thread in _teammate_threads.items()
            if thread.is_alive()
        )
    return TeammateShutdownOutcome(not live, live)


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    _validate_teammate_name(from_name)
    state = _create_protocol_request(
        "plan_approval", from_name, "lead", plan
    )
    try:
        BUS.send(
            from_name,
            "lead",
            plan,
            "plan_approval_request",
            {"request_id": state.request_id},
        )
    except Exception:
        with _protocol_lock:
            pending_requests.pop(state.request_id, None)
            _persist_protocol_requests_locked()
        raise
    return f"Plan submitted ({state.request_id})"


def run_request_shutdown(teammate: str) -> str:
    try:
        _validate_teammate_name(teammate)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        if teammate not in active_teammates:
            return f"Error: teammate '{teammate}' is not active"
    state = _create_protocol_request("shutdown", "lead", teammate, "")
    try:
        BUS.send(
            "lead",
            teammate,
            "Shut down.",
            "shutdown_request",
            {"request_id": state.request_id},
        )
    except Exception:
        with _protocol_lock:
            pending_requests.pop(state.request_id, None)
            _persist_protocol_requests_locked()
        raise
    return f"Shutdown request sent to {teammate}"


def run_request_plan(teammate: str, task: str) -> str:
    try:
        _validate_teammate_name(teammate)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        if teammate not in active_teammates:
            return f"Error: teammate '{teammate}' is not active"
    BUS.send("lead", teammate, f"Submit plan for: {task}", "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(
    request_id: str, approve: bool, feedback: str = ""
) -> str:
    with _protocol_lock:
        _ensure_protocol_state_loaded_locked()
        state = pending_requests.get(request_id)
        if not state:
            return f"Request {request_id} not found"
        if state.type != "plan_approval" or state.target != "lead":
            return f"Request {request_id} is not a plan approval request"
        if state.status != "pending":
            return f"Request {request_id} is already {state.status}"
        state.status = "approved" if approve else "rejected"
        try:
            BUS.send(
                "lead",
                state.sender,
                feedback or ("Approved" if approve else "Rejected"),
                "plan_approval_response",
                {"request_id": request_id, "approve": approve},
            )
        except Exception:
            state.status = "pending"
            _persist_protocol_requests_locked()
            raise
        _persist_protocol_requests_locked()
    return f"Plan {'approved' if approve else 'rejected'}"


def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    context = current_event_context()
    if (
        context.get("agent_type") == "main"
        and context.get("source") == "team_inbox"
    ):
        return (
            "Error: Team Agents cannot be created by an automatic Lead inbox "
            "Turn; wait for an explicit user Turn"
        )
    return spawn_teammate_thread(
        name,
        role,
        prompt,
        await_assignment=True,
    )


def _automatic_inbox_lifecycle_error(action: str) -> str | None:
    context = current_event_context()
    if (
        context.get("agent_type") == "main"
        and context.get("source") == "team_inbox"
    ):
        return (
            f"Error: an automatic Lead inbox Turn cannot {action} Team Agents; "
            "wait for an explicit user Turn"
        )
    return None


def run_stop_teammate(teammate: str) -> str:
    blocked = _automatic_inbox_lifecycle_error("stop")
    return blocked or stop_teammate(teammate)


def run_restart_teammate(teammate: str) -> str:
    blocked = _automatic_inbox_lifecycle_error("restart")
    return blocked or restart_teammate(teammate)


def run_send_message(to: str, content: str) -> str:
    try:
        _validate_teammate_name(to)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        if to not in active_teammates:
            return f"Error: teammate '{to}' is not active"
    BUS.send("lead", to, content)
    return f"Sent to {to}"


def run_check_inbox() -> str:
    batch = claim_lead_inbox(route_protocol=True)
    try:
        msgs = list(batch.messages)
        if not msgs:
            rendered = "(inbox empty)"
        else:
            lines = []
            for message in msgs:
                meta = message.get("metadata", {})
                request_id = meta.get("request_id", "")
                tag = (
                    f" [{message['type']} req:{request_id}]"
                    if request_id
                    else f" [{message['type']}]"
                )
                lines.append(
                    f"  [{message['from']}]{tag} {message['content']}"
                )
            rendered = "\n".join(lines)
    except BaseException as error:
        nack_lead_inbox(batch, str(error))
        raise
    ack_lead_inbox(batch)
    return rendered
