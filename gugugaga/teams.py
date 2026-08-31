from __future__ import annotations

import json
import random
import re
import threading
import time
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from . import config
from .hooks import trigger_hooks
from .models import ToolCall
from .observability import event_scope, notify, record_llm_call
from .permissions import PermissionPolicy
from .provider import is_context_length_error
from .tasks import can_start, claim_task, complete_task, list_tasks
from .workspace import run_bash, run_read, run_write

if TYPE_CHECKING:
    from .context_modes import SessionContextCoordinator


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
    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {}
_protocol_lock = threading.RLock()


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
        return True


def consume_lead_inbox(route_protocol: bool = True) -> list[dict]:
    msgs = BUS.read_inbox("lead")
    if route_protocol:
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
    return msgs


IDLE_POLL_INTERVAL = 5
IDLE_TIMEOUT = 60
PLAN_APPROVAL_TIMEOUT = 60


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
    deadline = time.monotonic() + max(0.0, float(IDLE_TIMEOUT))
    while time.monotonic() < deadline:
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
        remaining = max(0.0, deadline - time.monotonic())
        if stop_event.wait(min(float(IDLE_POLL_INTERVAL), remaining)):
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
        if rendered.startswith(("Error:", "Unknown:"))
        else "ok"
    )
    return rendered, status


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

    protocol_ctx = {"waiting_plan": None, "waiting_since": None}
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
            with _protocol_lock:
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
                burst_complete = False
                rounds_this_burst = 0
                while rounds_this_burst < _team_max_rounds_per_burst:
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
                            if _teammate_stop_event.wait(IDLE_POLL_INTERVAL):
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
                    if _teammate_stop_event.is_set():
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
                        burst_complete = True
                        break
                    results = []
                    plan_submitted = False
                    for block in tool_blocks:
                        if _teammate_stop_event.is_set():
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
                    _teammate_stop_event,
                )
                if idle_result in ("shutdown", "timeout"):
                    break

            if latest_summary and not _teammate_stop_event.is_set():
                BUS.send(name, "lead", latest_summary, "result")
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
        if state.type != "plan_approval" or state.target != "lead":
            return f"Request {request_id} is not a plan approval request"
        if state.status != "pending":
            return f"Request {request_id} is already {state.status}"
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
