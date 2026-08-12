from __future__ import annotations

import json
import random
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config
from .hooks import trigger_hooks
from .models import ToolCall
from .permissions import PermissionDecision, PermissionPolicy
from .planning import Task, TaskStore
from .tasks import can_start, claim_task, complete_task, list_tasks
from .workspace import run_bash, run_read, run_write


# S15-S17 source-compatible team communication. Paths are resolved from
# config at call time so every teammate sees the selected shared workspace.
_mailbox_lock = threading.RLock()
_AGENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _validate_agent_name(name: str) -> str:
    if not isinstance(name, str) or not _AGENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid agent name: {name}")
    return name


class MessageBus:
    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: dict | None = None,
    ):
        _validate_agent_name(from_agent)
        _validate_agent_name(to_agent)
        msg = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "ts": time.time(),
            "metadata": metadata or {},
        }
        config.MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
        inbox = config.MAILBOX_DIR / f"{to_agent}.jsonl"
        with _mailbox_lock, inbox.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def read_inbox(self, agent: str) -> list[dict]:
        _validate_agent_name(agent)
        inbox = config.MAILBOX_DIR / f"{agent}.jsonl"
        with _mailbox_lock:
            if not inbox.exists():
                return []
            msgs = [
                json.loads(line)
                for line in inbox.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            inbox.unlink()
        return msgs


BUS = MessageBus()
active_teammates: dict[str, bool] = {}
_teammate_lock = threading.RLock()
_teammate_threads: dict[str, threading.Thread] = {}
_teammate_stop_event = threading.Event()
_team_accepting = False


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

    tool_call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""

    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {}
_protocol_lock = threading.RLock()
PERMISSION_POLL_SECONDS = 0.5
PERMISSION_TIMEOUT_SECONDS = 60.0

_lead_inbox_lock = threading.RLock()
_lead_inbox_buffer: list[dict] = []

def _new_request_id_locked() -> str:
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
    *,
    tool_call_id: str = "",
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
) -> ProtocolState:
    with _protocol_lock:
        request_id = _new_request_id_locked()
        state = ProtocolState(
            request_id=request_id,
            type=request_type,
            sender=sender,
            target=target,
            status="pending",
            payload=payload,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=dict(arguments or {}),
        )
        pending_requests[request_id] = state
        return state


def match_response(
    response_type: str,
    request_id: str,
    approve: bool,
):
    expected_types = {
        "shutdown": "shutdown_response",
        "plan_approval": "plan_approval_response",
        "permission": "permission_response",
    }

    with _protocol_lock:
        state = pending_requests.get(request_id)
        if state is None:
            return

        expected = expected_types.get(state.type)
        if expected != response_type:
            return

        if state.status != "pending":
            return

        state.status = "approved" if approve else "rejected"


def _route_protocol_messages(messages: list[dict]) -> None:
    for message in messages:
        metadata = message.get("metadata", {})
        request_id = metadata.get("request_id", "")
        message_type = message.get("type", "")

        if request_id and message_type.endswith("_response"):
            match_response(
                message_type,
                request_id,
                metadata.get("approve", False),
            )


def poll_lead_inbox(
    route_protocol: bool = True,
) -> list[dict]:
    """读取新消息，但保留在缓冲区等待主循环消费。"""
    messages = BUS.read_inbox("lead")

    if route_protocol:
        _route_protocol_messages(messages)

    if messages:
        with _lead_inbox_lock:
            _lead_inbox_buffer.extend(messages)

    return messages


def consume_lead_inbox(
    route_protocol: bool = True,
) -> list[dict]:
    """取得所有已轮询和刚到达的 Lead 消息。"""
    poll_lead_inbox(route_protocol=route_protocol)

    with _lead_inbox_lock:
        messages = list(_lead_inbox_buffer)
        _lead_inbox_buffer.clear()

    return messages


IDLE_POLL_INTERVAL = 5
IDLE_TIMEOUT = 60


def scan_unclaimed_tasks() -> list[dict]:
    unclaimed = []
    for path in sorted(config.TASKS_DIR.glob("task_*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if (
            task.get("status") == "pending"
            and not task.get("owner")
            and can_start(task["id"])
        ):
            unclaimed.append(task)
    return unclaimed


def idle_poll(
    agent_name: str,
    messages: list,
    name: str,
    role: str,
    stop_event: threading.Event | None = None,
) -> str:
    del role
    stop_event = stop_event or threading.Event()
    for _ in range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL):
        if stop_event.is_set():
            return "shutdown"
        inbox = BUS.read_inbox(agent_name)
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
            messages.append(
                {
                    "role": "user",
                    "content": "<inbox>" + json.dumps(inbox) + "</inbox>",
                }
            )
            return "work"
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task_data = unclaimed[0]
            result = claim_task(task_data["id"], agent_name)
            if "Claimed" in result:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<auto-claimed>Task {task_data['id']}: "
                            f"{task_data['subject']}</auto-claimed>"
                        ),
                    }
                )
                return "work"
        if stop_event.wait(IDLE_POLL_INTERVAL):
            return "shutdown"
    return "timeout"


_team_provider: Any | None = None
_team_permissions = PermissionPolicy()
_team_approval_callback: Callable[[ToolCall], bool] | None = None


def set_team_provider(
    provider: Any | None,
    permissions: PermissionPolicy | None = None,
    approval_callback: Callable[[ToolCall], bool] | None = None,
) -> None:
    global _team_provider, _team_permissions, _team_approval_callback
    global _team_accepting
    _team_provider = provider
    _team_permissions = permissions or PermissionPolicy()
    _team_approval_callback = approval_callback
    with _teammate_lock:
        _team_accepting = provider is not None
        if provider is not None and not _teammate_threads:
            _teammate_stop_event.clear()

def request_teammate_permission(
    agent_name: str,
    call: ToolCall,
) -> ProtocolState:
    payload_data = {
        "tool": call.name,
        "arguments": call.arguments,
        "tool_call_id": call.id,
    }
    payload = json.dumps(payload_data, ensure_ascii=False)

    state = _create_protocol_request(
        "permission",
        agent_name,
        "lead",
        payload,
        tool_call_id=call.id,
        tool_name=call.name,
        arguments=call.arguments,
    )

    BUS.send(
        agent_name,
        "lead",
        payload,
        "permission_request",
        {
            "request_id": state.request_id,
            "tool_call_id": call.id,
        },
    )

    return state


def await_permission(
    request_id: str,
    *,
    timeout: float = PERMISSION_TIMEOUT_SECONDS,
    stop_event: threading.Event | None = None,
) -> ProtocolState:
    deadline = time.monotonic() + max(0.0, timeout)

    while True:
        with _protocol_lock:
            state = pending_requests.get(request_id)
            if state is None:
                raise RuntimeError(
                    f"permission request disappeared: {request_id}"
                )

            if state.status in {
                "approved",
                "rejected",
                "expired",
            }:
                return state

        if stop_event is not None and stop_event.is_set():
            with _protocol_lock:
                if state.status == "pending":
                    state.status = "expired"
                    state.feedback = "Teammate is shutting down."
            return state

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            with _protocol_lock:
                if state.status == "pending":
                    state.status = "expired"
                    state.feedback = "Approval timed out."
            return state

        wait_time = min(PERMISSION_POLL_SECONDS, remaining)

        if stop_event is None:
            time.sleep(wait_time)
        else:
            stop_event.wait(wait_time)


def run_review_permission(
    request_id: str,
    approve: bool,
    feedback: str = "",
) -> str:
    with _protocol_lock:
        state = pending_requests.get(request_id)

        if state is None:
            return f"Error: request {request_id} not found"

        if state.type != "permission":
            return (
                f"Error: request {request_id} is "
                f"{state.type}, not permission"
            )

        if state.status != "pending":
            return (
                f"Error: request {request_id} "
                f"is already {state.status}"
            )

        state.status = "approved" if approve else "rejected"
        state.feedback = feedback
        target = state.sender

    BUS.send(
        "lead",
        target,
        feedback or state.status,
        "permission_response",
        {
            "request_id": request_id,
            "approve": approve,
            "tool_call_id": state.tool_call_id,
        },
    )

    return (
        f"Permission {state.status}: {request_id} "
        f"({state.tool_name})"
    )


def run_list_permissions() -> str:
    with _protocol_lock:
        requests = [
            state
            for state in pending_requests.values()
            if state.type == "permission"
            and state.status == "pending"
        ]

    if not requests:
        return "No pending permission requests."

    return "\n".join(
        f"  {state.request_id}: {state.sender} requests "
        f"{state.tool_name} {json.dumps(state.arguments, ensure_ascii=False)}"
        for state in requests
    )

def _dispatch_teammate_tool(
    agent_name: str,
    block,
    handlers: dict[str, Callable],
    *,
    permission_timeout: float = PERMISSION_TIMEOUT_SECONDS,
) -> str:
    from .tools import call_tool_handler

    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked)

    call = ToolCall(
        block.id,
        block.name,
        dict(block.input),
    )
    decision = _team_permissions.decide(call)

    if decision is PermissionDecision.DENY:
        return (
            f"Permission denied for tool '{call.name}'. "
            "Choose a safer approach."
        )

    if decision is PermissionDecision.ASK:
        request = request_teammate_permission(
            agent_name,
            call,
        )
        resolved = await_permission(
            request.request_id,
            timeout=permission_timeout,
            stop_event=_teammate_stop_event,
        )

        if resolved.status != "approved":
            detail = (
                f": {resolved.feedback}"
                if resolved.feedback
                else ""
            )
            return (
                f"Permission {resolved.status} for tool "
                f"'{call.name}'{detail}"
            )

    output = call_tool_handler(
        handlers.get(block.name),
        block.input,
        block.name,
    )
    trigger_hooks("PostToolUse", block, output)
    return str(output)


def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    try:
        _validate_agent_name(name)
    except ValueError as error:
        return f"Error: {error}"
    if _team_provider is None:
        return "Error: teammate provider is not configured"
    with _teammate_lock:
        if not _team_accepting:
            return "Error: teammate manager is not accepting new teammates"
        if name in active_teammates:
            return f"Teammate '{name}' already exists"
        if not _teammate_threads:
            _teammate_stop_event.clear()
        active_teammates[name] = True

    protocol_ctx = {"waiting_plan": None}
    system = (
        f"You are '{name}', a {role}. Use tools to complete tasks in the "
        f"shared selected workspace at {config.WORKDIR}."
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
            if request_id == protocol_ctx["waiting_plan"]:
                protocol_ctx["waiting_plan"] = None
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
        messages = [{"role": "user", "content": prompt}]
        sub_tools = [
            {
                "name": "bash",
                "description": "Run a shell command.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
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

        def send_message(to: str, content: str) -> str:
            BUS.send(name, to, content)
            return "Sent"

        def list_task_lines() -> str:
            current = list_tasks()
            if not current:
                return "No tasks."
            return "\n".join(
                f"  {task.id}: {task.subject} [{task.status}]"
                for task in current
            )

        sub_handlers = {
            "bash": run_bash,
            "read_file": run_read,
            "write_file": run_write,
            "send_message": send_message,
            "list_tasks": list_task_lines,
            "claim_task": lambda task_id: claim_task(task_id, owner=name),
            "complete_task": complete_task,
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
        }

        try:
            while not _teammate_stop_event.is_set():
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
                for _ in range(10):
                    if _teammate_stop_event.is_set():
                        should_shutdown = True
                        break
                    inbox = BUS.read_inbox(name)
                    for msg in inbox:
                        if handle_inbox_message(name, msg, messages):
                            should_shutdown = True
                            break
                    if should_shutdown:
                        break
                    if protocol_ctx["waiting_plan"]:
                        if _teammate_stop_event.wait(IDLE_POLL_INTERVAL):
                            should_shutdown = True
                            break
                        continue
                    if inbox:
                        non_protocol = [
                            msg
                            for msg in inbox
                            if msg.get("type") == "message"
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
                    try:
                        response = _team_provider.create(
                            model=config.MODEL or None,
                            system=system,
                            messages=messages[-20:],
                            tools=sub_tools,
                            max_tokens=8000,
                        )
                    except Exception:
                        break
                    if _teammate_stop_event.is_set():
                        should_shutdown = True
                        break
                    messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                    if not any(
                        getattr(block, "type", None) == "tool_use"
                        for block in response.content
                    ):
                        break
                    results = []
                    for block in response.content:
                        if getattr(block, "type", None) != "tool_use":
                            continue
                        if _teammate_stop_event.is_set():
                            should_shutdown = True
                            break
                        output = _dispatch_teammate_tool(
                            name,
                            block,
                            sub_handlers,
                        )
                        if block.name == "submit_plan" and output.startswith(
                            "Plan submitted ("
                        ):
                            match = re.search(r"\((req_\d+)\)", output)
                            protocol_ctx["waiting_plan"] = (
                                match.group(1) if match else output
                            )
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(output),
                            }
                        )
                        if protocol_ctx["waiting_plan"]:
                            break
                    if should_shutdown:
                        break
                    messages.append({"role": "user", "content": results})
                    if protocol_ctx["waiting_plan"]:
                        break
                if should_shutdown:
                    break
                if protocol_ctx["waiting_plan"]:
                    continue
                idle_result = idle_poll(
                    name,
                    messages,
                    name,
                    role,
                    _teammate_stop_event,
                )
                if idle_result in ("shutdown", "timeout"):
                    break

            summary = "Done."
            for message in reversed(messages):
                if message["role"] != "assistant" or not isinstance(
                    message["content"], list
                ):
                    continue
                for block in message["content"]:
                    if getattr(block, "type", None) == "text":
                        summary = block.text
                        break
                else:
                    continue
                break
            BUS.send(name, "lead", summary, "result")
        finally:
            with _teammate_lock:
                active_teammates.pop(name, None)
                _teammate_threads.pop(name, None)

    thread = threading.Thread(
        target=run, name=f"teammate-{name}", daemon=True
    )
    with _teammate_lock:
        _teammate_threads[name] = thread
    thread.start()
    return f"Teammate '{name}' spawned as {role}"


def stop_all_teammates(timeout: float = 5.0) -> TeammateShutdownOutcome:
    """Cancel and join all source-faithful teammate threads."""
    global _team_accepting
    with _teammate_lock:
        _team_accepting = False
    _teammate_stop_event.set()
    with _teammate_lock:
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
        live = tuple(
            name
            for name, thread in _teammate_threads.items()
            if thread.is_alive()
        )
    return TeammateShutdownOutcome(not live, live)


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    _validate_agent_name(from_name)
    state = _create_protocol_request(
        "plan_approval", from_name, "lead", plan
    )
    BUS.send(
        from_name,
        "lead",
        plan,
        "plan_approval_request",
        {"request_id": state.request_id},
    )
    return f"Plan submitted ({state.request_id})"


def run_request_shutdown(teammate: str) -> str:
    try:
        _validate_agent_name(teammate)
    except ValueError as error:
        return f"Error: {error}"
    state = _create_protocol_request("shutdown", "lead", teammate, "")
    BUS.send(
        "lead",
        teammate,
        "Shut down.",
        "shutdown_request",
        {"request_id": state.request_id},
    )
    return f"Shutdown request sent to {teammate}"


def run_request_plan(teammate: str, task: str) -> str:
    try:
        _validate_agent_name(teammate)
    except ValueError as error:
        return f"Error: {error}"
    BUS.send("lead", teammate, f"Submit plan for: {task}", "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(
    request_id: str, approve: bool, feedback: str = ""
) -> str:
    with _protocol_lock:
        state = pending_requests.get(request_id)
        if not state:
            return f"Request {request_id} not found"
        state.status = "approved" if approve else "rejected"
    BUS.send(
        "lead",
        state.sender,
        feedback or ("Approved" if approve else "Rejected"),
        "plan_approval_response",
        {"request_id": request_id, "approve": approve},
    )
    return f"Plan {'approved' if approve else 'rejected'}"


def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    try:
        _validate_agent_name(to)
    except ValueError as error:
        return f"Error: {error}"

    with _teammate_lock:
        active = to in active_teammates

    if not active:
        return (
            f"Error: teammate '{to}' is not active. "
            "Call spawn_teammate first."
        )

    BUS.send("lead", to, content)
    return f"Sent to {to}"


def run_check_inbox() -> str:
    msgs = consume_lead_inbox(route_protocol=True)
    if not msgs:
        return "(inbox empty)"
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
            f"  [{message['from']}]{tag} {message['content'][:200]}"
        )
    return "\n".join(lines)


class Mailbox:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, name: str) -> Path:
        return self.directory / f"{_validate_agent_name(name)}.jsonl"

    def send(
        self,
        sender: str,
        target: str,
        content: str,
        message_type: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        message = {
            "from": sender,
            "to": target,
            "content": content,
            "type": message_type,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        with self._lock, self._path(target).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")

    def drain(self, name: str) -> list[dict[str, Any]]:
        path = self._path(name)
        with self._lock:
            if not path.exists():
                return []
            items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            path.write_text("", encoding="utf-8")
            return items

    def peek(self, name: str) -> bool:
        path = self._path(name)
        with self._lock:
            return path.exists() and bool(path.read_text(encoding="utf-8").strip())


class ProtocolError(RuntimeError):
    pass


@dataclass
class ProtocolRequest:
    id: str
    type: str
    sender: str
    target: str
    payload: str
    status: str = "pending"
    feedback: str = ""


class ProtocolStore:
    RESPONSE_TYPES = {
        "plan_approval": "plan_approval_response",
        "shutdown": "shutdown_response",
        "permission": "permission_response",
    }

    def __init__(self):
        self.requests: dict[str, ProtocolRequest] = {}
        self._lock = threading.Lock()

    def request(self, request_type: str, sender: str, target: str, payload: str) -> ProtocolRequest:
        if request_type not in self.RESPONSE_TYPES:
            raise ProtocolError(f"unknown protocol type: {request_type}")
        item = ProtocolRequest(f"req_{uuid.uuid4().hex[:10]}", request_type, sender, target, payload)
        with self._lock:
            self.requests[item.id] = item
        return item

    def resolve(self, request_id: str, response_type: str, approve: bool, feedback: str = "") -> ProtocolRequest:
        with self._lock:
            item = self.requests.get(request_id)
            if item is None:
                raise ProtocolError(f"unknown request: {request_id}")
            expected = self.RESPONSE_TYPES[item.type]
            if response_type != expected:
                raise ProtocolError(f"expected {expected}, got {response_type}")
            item.status = "approved" if approve else "rejected"
            item.feedback = feedback
            return item

    def get(self, request_id: str) -> ProtocolRequest:
        return self.requests[request_id]


class TeamManager:
    def __init__(
        self,
        mailbox: Mailbox,
        tasks: TaskStore,
        protocols: ProtocolStore,
        runtime_factory: Callable[[str, str], Any],
        poll_seconds: float = 1.0,
        idle_timeout: float = 30.0,
    ):
        self.mailbox = mailbox
        self.tasks = tasks
        self.protocols = protocols
        self.runtime_factory = runtime_factory
        self.poll_seconds = poll_seconds
        self.idle_timeout = idle_timeout
        self.members: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def claim_next(self, name: str) -> Task | None:
        for task in self.tasks.list():
            if task.status == "pending" and not task.owner and self.tasks.can_start(task):
                if self.tasks.claim(task.id, name).startswith("Claimed"):
                    return self.tasks.get(task.id)
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        with self._lock:
            if name in self.members and self.members[name]["status"] != "stopped":
                return f"Error: teammate '{name}' already exists"
            self.members[name] = {"role": role, "status": "working", "stop": threading.Event()}

        def run():
            runtime = self.runtime_factory(name, role)
            try:
                result = runtime.run_turn(prompt)
                self.mailbox.send(name, "lead", result, "result")
                idle_started = time.time()
                while not self.members[name]["stop"].is_set() and time.time() - idle_started < self.idle_timeout:
                    inbox = self.mailbox.drain(name)
                    if inbox:
                        idle_started = time.time()
                    should_stop = False
                    for message in inbox:
                        kind = message.get("type", "message")
                        if kind == "shutdown_request":
                            request_id = message.get("metadata", {}).get("request_id", "")
                            self.protocols.resolve(request_id, "shutdown_response", True)
                            self.mailbox.send(name, "lead", "Shutdown approved", "shutdown_response", {"request_id": request_id, "approve": True})
                            should_stop = True
                            break
                        if kind == "request_plan":
                            plan = runtime.run_turn(f"Create a concise plan for: {message['content']}")
                            request = self.protocols.request("plan_approval", name, "lead", plan)
                            self.mailbox.send(name, "lead", plan, "plan_approval_request", {"request_id": request.id})
                        elif kind == "permission_response":
                            continue
                        else:
                            answer = runtime.run_turn(message["content"])
                            self.mailbox.send(name, "lead", answer, "message")
                    if should_stop:
                        break
                    task = self.claim_next(name)
                    if task:
                        idle_started = time.time()
                        answer = runtime.run_turn(f"Complete task {task.id}: {task.subject}\n{task.description}")
                        self.mailbox.send(name, "lead", answer, "task_result", {"task_id": task.id})
                    with self._lock:
                        self.members[name]["status"] = "idle"
                    time.sleep(self.poll_seconds)
            finally:
                with self._lock:
                    self.members[name]["status"] = "stopped"

        thread = threading.Thread(target=run, name=f"teammate-{name}", daemon=True)
        self.members[name]["thread"] = thread
        thread.start()
        return f"Spawned teammate '{name}' as {role}"

    def send(self, target: str, content: str) -> str:
        self.mailbox.send("lead", target, content)
        return f"Sent message to {target}"

    def check_inbox(self) -> list[dict[str, Any]]:
        return self.mailbox.drain("lead")

    def request_shutdown(self, teammate: str) -> str:
        request = self.protocols.request("shutdown", "lead", teammate, "Graceful shutdown")
        self.mailbox.send("lead", teammate, request.payload, "shutdown_request", {"request_id": request.id})
        return request.id

    def request_plan(self, teammate: str, task: str) -> str:
        self.mailbox.send("lead", teammate, task, "request_plan")
        return f"Requested plan from {teammate}"

    def request_permission(self, teammate: str, call: ToolCall) -> str:
        payload = json.dumps(
            {"tool": call.name, "arguments": call.arguments}, ensure_ascii=False
        )
        request = self.protocols.request("permission", teammate, "lead", payload)
        self.mailbox.send(
            teammate,
            "lead",
            payload,
            "permission_request",
            {"request_id": request.id, "tool_call_id": call.id},
        )
        return request.id

    def await_permission(
        self, teammate: str, call: ToolCall, timeout: float = 60.0
    ) -> bool:
        request_id = self.request_permission(teammate, call)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.protocols.get(request_id).status
            if status == "approved":
                return True
            if status == "rejected":
                return False
            time.sleep(min(self.poll_seconds, 0.1))
        return False

    def review_permission(
        self, request_id: str, approve: bool, feedback: str = ""
    ) -> str:
        request = self.protocols.resolve(
            request_id, "permission_response", approve, feedback
        )
        self.mailbox.send(
            "lead",
            request.sender,
            feedback or request.status,
            "permission_response",
            {"request_id": request.id, "approve": approve},
        )
        return f"Permission {request.status}: {request.id}"

    def review_plan(self, request_id: str, approve: bool, feedback: str = "") -> str:
        request = self.protocols.resolve(request_id, "plan_approval_response", approve, feedback)
        self.mailbox.send("lead", request.sender, feedback or request.status, "plan_approval_response", {"request_id": request.id, "approve": approve})
        return f"Plan {request.status}: {request.id}"

    def status(self) -> str:
        with self._lock:
            if not self.members:
                return "No teammates."
            return "\n".join(f"{name} ({item['role']}): {item['status']}" for name, item in sorted(self.members.items()))

    def stop_all(self) -> None:
        with self._lock:
            for item in self.members.values():
                item["stop"].set()
