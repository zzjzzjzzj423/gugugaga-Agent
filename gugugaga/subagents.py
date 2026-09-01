from __future__ import annotations

import inspect
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from . import config
from .hooks import trigger_hooks
from .models import ToolCall
from .observability import current_event_context, event_scope, notify, record_llm_call
from .permissions import PermissionDecision, PermissionPolicy
from .prompts import subagent_system_prompt
from .provider import is_context_length_error
from .web_search import run_web_search
from .workspace import run_bash, run_edit, run_glob, run_read, run_write

if TYPE_CHECKING:
    from .context_modes import SessionContextCoordinator


client = None
MODEL = config.MODEL
_permissions = PermissionPolicy()
_approval_callback: Callable[[ToolCall], bool] | None = None
_context_parent_resolver: Callable[[], SessionContextCoordinator] | None = None
_max_rounds = 30
_max_tokens = config.DEFAULT_MAX_TOKENS

MAX_CONCURRENT_SUBAGENTS = 4
SUBAGENT_TIMEOUT_SECONDS = 300.0
WRITE_FINISH_GRACE_SECONDS = 30.0
MAX_WAIT_SECONDS = 30.0
_PRIVILEGED_TOOLS = frozenset({"bash", "write_file", "edit_file"})


class SubagentState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_PERMISSION = "waiting_permission"
    WAITING_HUMAN_PERMISSION = "waiting_human_permission"
    WAITING_RESOURCE = "waiting_resource"
    WRITING = "writing"
    FINISHING_WRITE = "finishing_write"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    STUCK = "stuck"


TERMINAL_STATES = frozenset(
    {
        SubagentState.COMPLETED,
        SubagentState.FAILED,
        SubagentState.CANCELLED,
        SubagentState.TIMED_OUT,
    }
)


class SubagentCancelled(RuntimeError):
    pass


@dataclass
class PermissionRequest:
    request_id: str
    tool_call: ToolCall
    created_at: float = field(default_factory=time.monotonic)
    notified: bool = False
    decision: bool | None = None
    feedback: str = ""
    reviewing: bool = False


@dataclass
class SubagentJob:
    subagent_id: str
    parent_turn_id: str
    description: str
    task_id: str | None
    created_at: float
    deadline: float
    event_context: dict[str, Any]
    state: SubagentState = SubagentState.QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    result: str | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    permission_event: threading.Event = field(default_factory=threading.Event)
    pending_permission: PermissionRequest | None = None
    terminal_delivered: bool = False
    thread: threading.Thread | None = None
    timer: threading.Timer | None = None
    mutation_active: bool = False
    finish_after_write: bool = False
    effects_committed: bool = False
    cancel_reason: str | None = None
    runtime_approved_calls: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class SubagentShutdownOutcome:
    stopped: bool
    live_subagent_ids: tuple[str, ...]


def configure_subagent_runtime(
    provider,
    *,
    model: str | None = None,
    permissions: PermissionPolicy | None = None,
    approval_callback: Callable[[ToolCall], bool] | None = None,
    context_parent_resolver: Callable[[], SessionContextCoordinator] | None = None,
    max_rounds: int = 30,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
) -> None:
    global client, MODEL, _permissions, _approval_callback
    global _context_parent_resolver, _max_rounds, _max_tokens
    client = provider
    MODEL = model or config.MODEL
    _permissions = permissions or PermissionPolicy()
    _approval_callback = approval_callback
    _context_parent_resolver = context_parent_resolver
    _max_rounds = max(1, int(max_rounds))
    _max_tokens = max(1, int(max_tokens))
    initialize_subagent_manager()


def reset_subagent_runtime() -> None:
    configure_subagent_runtime(None, model=config.MODEL)


SUB_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command after lead approval.",
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
        "description": "Read file contents.",
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
        "description": "Write content after lead approval.",
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
        "name": "edit_file",
        "description": "Replace exact text once after lead approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "expected_sha256": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the current public web.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 400},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

SUB_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "web_search": run_web_search,
}


def extract_text(content) -> str:
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        getattr(block, "text", "")
        for block in content
        if getattr(block, "type", None) == "text"
    ).strip()


def has_tool_use(content) -> bool:
    return any(getattr(block, "type", None) == "tool_use" for block in content)


def _raise_if_cancelled(
    cancel_event: threading.Event | None,
    deadline: float | Callable[[], float] | None,
) -> None:
    value = deadline() if callable(deadline) else deadline
    if value is not None and time.monotonic() >= value:
        if cancel_event is not None:
            cancel_event.set()
        raise SubagentCancelled("Subagent timed out")
    if cancel_event is not None and cancel_event.is_set():
        raise SubagentCancelled("Subagent cancelled")


def _call_subagent_handler(
    handler,
    arguments: dict[str, Any],
    name: str,
    cancel_event: threading.Event | None,
    deadline: float | Callable[[], float] | None,
) -> str:
    from .tools import call_tool_handler

    if handler is not None and cancel_event is not None:
        names: set[str] = set()
        try:
            parameters = inspect.signature(handler).parameters.values()
            names = {parameter.name for parameter in parameters}
            supports_cancel = any(
                parameter.name == "cancel_event"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            supports_cancel = False
        if supports_cancel:
            arguments = {**(arguments or {}), "cancel_event": cancel_event}
        if "deadline" in names:
            arguments = {**(arguments or {}), "deadline": deadline}
    return str(call_tool_handler(handler, arguments, name))


def _request_runtime_approval(
    call: ToolCall,
    *,
    cancel_event: threading.Event | None,
    deadline: float | Callable[[], float] | None,
) -> bool:
    callback = _approval_callback
    if callback is None:
        return False
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if cancel_event is not None and (
        "cancel_event" in parameters or accepts_kwargs
    ):
        kwargs["cancel_event"] = cancel_event
    value = deadline() if callable(deadline) else deadline
    if value is not None and (
        "timeout_seconds" in parameters or accepts_kwargs
    ):
        kwargs["timeout_seconds"] = max(0.1, value - time.monotonic())
    return bool(callback(call, **kwargs))


def _execute_subagent(
    description: str,
    *,
    agent_id: str | None = None,
    cancel_event: threading.Event | None = None,
    deadline: float | Callable[[], float] | None = None,
    permission_gate: Callable[[ToolCall], tuple[bool, str]] | None = None,
    human_permission_state: Callable[[bool], None] | None = None,
    runtime_permission_preapproved: Callable[[ToolCall], bool] | None = None,
    should_finish_after_write: Callable[[], bool] | None = None,
) -> str:
    from .context_modes import (
        CompressionReason,
        ContextModeError,
        RequestContext,
        create_child_context_coordinator,
    )

    messages = [{"role": "user", "content": description}]
    if client is None:
        raise RuntimeError("Subagent provider is not configured")
    agent_id = agent_id or f"subagent_{uuid.uuid4().hex}"
    coordinator = None
    if _context_parent_resolver is not None:
        parent = _context_parent_resolver()
        coordinator = create_child_context_coordinator(
            parent,
            agent_type="subagent",
            agent_id=agent_id,
        )
    with event_scope(agent_type="subagent", agent_id=agent_id):
        notify("subagent_start", {"description": description})
        try:
            attempted_recovery = False
            system = subagent_system_prompt()
            request_context = RequestContext(system=system, tools=SUB_TOOLS)
            for iteration in range(1, _max_rounds + 1):
                _raise_if_cancelled(cancel_event, deadline)
                if coordinator is not None:
                    try:
                        provider_messages = coordinator.prepare_request(
                            messages, request_context
                        )
                    except ContextModeError:
                        _raise_if_cancelled(cancel_event, deadline)
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
                _raise_if_cancelled(cancel_event, deadline)
                with event_scope(iteration=iteration):
                    try:
                        response = record_llm_call(
                            client,
                            provider_messages,
                            system,
                            SUB_TOOLS,
                            _max_tokens,
                            model=MODEL,
                            call_type="subagent",
                        )
                    except Exception as error:
                        _raise_if_cancelled(cancel_event, deadline)
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
                        raise
                _raise_if_cancelled(cancel_event, deadline)
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason == "max_tokens":
                    raise RuntimeError(
                        "Subagent response reached the output-token limit"
                    )
                if not has_tool_use(response.content):
                    reply = extract_text(response.content)
                    if not reply:
                        raise RuntimeError("Subagent finished without a text summary")
                    notify("subagent_end", {"reply": reply})
                    return reply
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    _raise_if_cancelled(cancel_event, deadline)
                    started = time.monotonic()
                    blocked = trigger_hooks("PreToolUse", block)
                    if blocked:
                        output = str(blocked)
                        status = "blocked"
                    else:
                        call = ToolCall(block.id, block.name, block.input)
                        decision = _permissions.decide(call)
                        if decision is PermissionDecision.DENY:
                            approved, feedback = False, "Denied by permission policy."
                        elif permission_gate is not None and (
                            block.name in _PRIVILEGED_TOOLS
                            or decision is PermissionDecision.ASK
                        ):
                            approved, feedback = permission_gate(call)
                            if (
                                approved
                                and decision is PermissionDecision.ASK
                            ):
                                preapproved = bool(
                                    runtime_permission_preapproved is not None
                                    and runtime_permission_preapproved(call)
                                )
                                if preapproved:
                                    approved = True
                                else:
                                    if human_permission_state is not None:
                                        human_permission_state(True)
                                    try:
                                        approved = _request_runtime_approval(
                                            call,
                                            cancel_event=cancel_event,
                                            deadline=deadline,
                                        )
                                    finally:
                                        if human_permission_state is not None:
                                            human_permission_state(False)
                                if not approved:
                                    feedback = "Human/runtime approval was denied."
                        else:
                            approved = _permissions.approve(call, _approval_callback)
                            feedback = ""
                        _raise_if_cancelled(cancel_event, deadline)
                        if not approved:
                            detail = f" {feedback}" if feedback else ""
                            output = (
                                f"Permission denied for tool '{block.name}'."
                                f"{detail} Choose a safer approach."
                            )
                            status = "denied"
                        else:
                            handler = SUB_HANDLERS.get(block.name)
                            try:
                                output = _call_subagent_handler(
                                    handler,
                                    block.input,
                                    block.name,
                                    cancel_event,
                                    deadline,
                                )
                                _raise_if_cancelled(cancel_event, deadline)
                            except Exception as error:
                                notify(
                                    "tool",
                                    {
                                        "iteration": iteration,
                                        "tool": block.name,
                                        "args": block.input,
                                        "status": "error",
                                        "error_type": type(error).__name__,
                                        "error": str(error),
                                        "latency_ms": round(
                                            (time.monotonic() - started) * 1000
                                        ),
                                    },
                                )
                                raise
                            trigger_hooks("PostToolUse", block, output)
                            status = (
                                "error"
                                if str(output).startswith(
                                    ("Error:", "Unknown:", "Conflict:")
                                )
                                else "ok"
                            )
                            if (
                                should_finish_after_write is not None
                                and should_finish_after_write()
                            ):
                                reply = (
                                    "Subagent write finished during the timeout "
                                    f"grace period. Last tool result: {output}"
                                )
                                notify("subagent_end", {"reply": reply})
                                return reply
                    notify(
                        "tool",
                        {
                            "iteration": iteration,
                            "tool": block.name,
                            "args": block.input,
                            "output": output,
                            "status": status,
                            "latency_ms": round(
                                (time.monotonic() - started) * 1000
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
                messages.append({"role": "user", "content": results})
            raise RuntimeError(f"Subagent exceeded maximum rounds ({_max_rounds})")
        except Exception as error:
            notify(
                "subagent_error",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            raise
        finally:
            if coordinator is not None:
                coordinator.close()


def spawn_subagent(description: str) -> str:
    """Compatibility entry point for callers that explicitly need a blocking run."""

    return _execute_subagent(description)


class SubagentManager:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_SUBAGENTS):
        self.max_concurrent = max(1, int(max_concurrent))
        self.jobs: dict[str, SubagentJob] = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._accepting = True

    @staticmethod
    def parent_turn_id(turn_id: str | None = None) -> str:
        value = turn_id or current_event_context().get("turn_id")
        return str(value or "turn_unscoped")

    def initialize(self) -> None:
        with self._lock:
            live = [
                job.subagent_id
                for job in self.jobs.values()
                if job.thread is not None and job.thread.is_alive()
            ]
            if live:
                raise RuntimeError(
                    "cannot initialize while subagents are live: " + ", ".join(live)
                )
            self.jobs.clear()
            self._accepting = True

    def start(
        self,
        description: str,
        *,
        task_id: str | None = None,
        parent_turn_id: str | None = None,
        timeout_seconds: float = SUBAGENT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        if not str(description).strip():
            raise ValueError("description must not be empty")
        now = time.monotonic()
        timeout = min(max(float(timeout_seconds), 1.0), SUBAGENT_TIMEOUT_SECONDS)
        trace_context = current_event_context()
        resolved_turn_id = self.parent_turn_id(parent_turn_id)
        trace_context["turn_id"] = resolved_turn_id
        trace_context.pop("agent_id", None)
        trace_context.pop("iteration", None)
        trace_context["agent_type"] = "main"
        job = SubagentJob(
            subagent_id=f"subagent_{uuid.uuid4().hex}",
            parent_turn_id=resolved_turn_id,
            description=str(description),
            task_id=str(task_id) if task_id else None,
            created_at=now,
            deadline=now + timeout,
            event_context=trace_context,
        )
        with self._changed:
            if not self._accepting:
                raise RuntimeError("subagent manager is not accepting new jobs")
            self.jobs[job.subagent_id] = job
            notify("subagent_queued", self._snapshot_locked(job))
            self._dispatch_locked()
            snapshot = self._snapshot_locked(job)
        return snapshot

    def _live_count_locked(self) -> int:
        return sum(
            1
            for job in self.jobs.values()
            if job.state not in TERMINAL_STATES
            and job.state is not SubagentState.QUEUED
        )

    def _dispatch_locked(self) -> None:
        while self._live_count_locked() < self.max_concurrent:
            job = next(
                (item for item in self.jobs.values() if item.state is SubagentState.QUEUED),
                None,
            )
            if job is None:
                return
            if time.monotonic() >= job.deadline:
                self._finish_locked(job, SubagentState.TIMED_OUT, error="Timed out in queue")
                continue
            job.state = SubagentState.RUNNING
            job.started_at = time.monotonic()
            thread = threading.Thread(
                target=self._worker,
                args=(job.subagent_id,),
                name=f"gugugaga-{job.subagent_id}",
                daemon=True,
            )
            timer = threading.Timer(
                max(0.0, job.deadline - time.monotonic()),
                self._timeout,
                args=(job.subagent_id,),
            )
            timer.daemon = True
            job.thread = thread
            job.timer = timer
            timer.start()
            thread.start()

    def _worker(self, subagent_id: str) -> None:
        from .mutations import mutation_actor_scope

        with self._lock:
            job = self.jobs.get(subagent_id)
            if job is None:
                return
            description = job.description
            cancel_event = job.cancel_event
            trace_context = dict(job.event_context)
        try:
            with event_scope(**trace_context):
                with mutation_actor_scope(
                    observer=lambda state, details: self._mutation_event(
                        subagent_id, state, details
                    ),
                    require_hash=True,
                ):
                    result = _execute_subagent(
                        description,
                        agent_id=subagent_id,
                        cancel_event=cancel_event,
                        deadline=lambda: self._job_deadline(subagent_id),
                        permission_gate=lambda call: self._await_permission(
                            subagent_id, call
                        ),
                        human_permission_state=lambda waiting: (
                            self._human_permission_state(subagent_id, waiting)
                        ),
                        runtime_permission_preapproved=lambda call: (
                            self._consume_runtime_approval(subagent_id, call)
                        ),
                        should_finish_after_write=lambda: self._finish_after_write(
                            subagent_id
                        ),
                    )
            with self._changed:
                job = self.jobs.get(subagent_id)
                if job is not None and job.state not in TERMINAL_STATES:
                    if job.cancel_event.is_set():
                        state = (
                            SubagentState.TIMED_OUT
                            if job.cancel_reason == "timeout"
                            else SubagentState.CANCELLED
                        )
                        self._finish_locked(job, state, error=job.cancel_reason)
                    else:
                        self._finish_locked(job, SubagentState.COMPLETED, result=result)
        except SubagentCancelled as error:
            with self._changed:
                job = self.jobs.get(subagent_id)
                if job is not None and job.state not in TERMINAL_STATES:
                    state = (
                        SubagentState.TIMED_OUT
                        if job.cancel_reason == "timeout"
                        else SubagentState.CANCELLED
                    )
                    self._finish_locked(job, state, error=str(error))
        except Exception as error:
            with self._changed:
                job = self.jobs.get(subagent_id)
                if job is not None and job.state not in TERMINAL_STATES:
                    if job.cancel_event.is_set():
                        state = (
                            SubagentState.TIMED_OUT
                            if job.cancel_reason == "timeout"
                            else SubagentState.CANCELLED
                        )
                        self._finish_locked(job, state, error=str(error))
                    else:
                        self._finish_locked(
                            job,
                            SubagentState.FAILED,
                            error=f"{type(error).__name__}: {error}",
                        )
        finally:
            with self._changed:
                job = self.jobs.get(subagent_id)
                if job is not None and job.timer is not None:
                    job.timer.cancel()
                self._cleanup_locked()
                self._dispatch_locked()
                self._changed.notify_all()

    def _job_deadline(self, subagent_id: str) -> float:
        with self._lock:
            job = self.jobs.get(subagent_id)
            return job.deadline if job is not None else time.monotonic()

    def _finish_after_write(self, subagent_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(subagent_id)
            return bool(job and job.finish_after_write and not job.mutation_active)

    def _human_permission_state(self, subagent_id: str, waiting: bool) -> None:
        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None or job.state in TERMINAL_STATES:
                return
            if waiting:
                job.state = SubagentState.WAITING_HUMAN_PERMISSION
            elif job.state is SubagentState.WAITING_HUMAN_PERMISSION:
                job.state = SubagentState.RUNNING
            self._changed.notify_all()

    def _consume_runtime_approval(
        self, subagent_id: str, call: ToolCall
    ) -> bool:
        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None or call.id not in job.runtime_approved_calls:
                return False
            job.runtime_approved_calls.remove(call.id)
            return True

    def _mutation_event(
        self, subagent_id: str, state: str, details: dict[str, Any]
    ) -> None:
        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None or job.state in TERMINAL_STATES:
                return
            if state == "waiting_resource":
                job.state = SubagentState.WAITING_RESOURCE
            elif state == "writing":
                job.mutation_active = True
                job.state = SubagentState.WRITING
            elif state == "committed":
                job.effects_committed = True
            elif state == "released":
                job.mutation_active = False
                if job.state is SubagentState.FINISHING_WRITE:
                    job.finish_after_write = True
                if job.state not in {SubagentState.CANCELLING, SubagentState.STUCK}:
                    job.state = SubagentState.RUNNING
            self._changed.notify_all()
            notify(
                "subagent_status",
                {
                    "subagent_id": subagent_id,
                    "status": job.state.value,
                    "mutation": state,
                    **details,
                },
            )

    def _finish_locked(
        self,
        job: SubagentJob,
        state: SubagentState,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        job.state = state
        job.result = result
        job.error = error
        job.finished_at = time.monotonic()
        job.pending_permission = None
        job.permission_event.set()
        self._changed.notify_all()
        notify(
            "subagent_status",
            {
                "subagent_id": job.subagent_id,
                "parent_turn_id": job.parent_turn_id,
                "turn_id": job.parent_turn_id,
                "agent_type": "subagent",
                "agent_id": job.subagent_id,
                "status": state.value,
                "task_id": job.task_id,
                "error": error,
            },
        )

    def _timeout(self, subagent_id: str) -> None:
        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None or job.state in TERMINAL_STATES:
                return
            if job.state is SubagentState.QUEUED:
                job.cancel_reason = "timeout"
                self._finish_locked(
                    job, SubagentState.TIMED_OUT, error="Timed out in queue"
                )
                return
            if job.mutation_active and job.state is not SubagentState.FINISHING_WRITE:
                job.state = SubagentState.FINISHING_WRITE
                job.finish_after_write = True
                job.deadline = time.monotonic() + WRITE_FINISH_GRACE_SECONDS
                timer = threading.Timer(
                    WRITE_FINISH_GRACE_SECONDS,
                    self._timeout,
                    args=(subagent_id,),
                )
                timer.daemon = True
                job.timer = timer
                timer.start()
                self._changed.notify_all()
                return
            job.cancel_reason = "timeout"
            job.cancel_event.set()
            job.permission_event.set()
            job.state = SubagentState.CANCELLING
            self._changed.notify_all()

    def _await_permission(
        self, subagent_id: str, call: ToolCall
    ) -> tuple[bool, str]:
        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None:
                raise SubagentCancelled("Subagent no longer exists")
            _raise_if_cancelled(job.cancel_event, job.deadline)
            request = PermissionRequest(
                request_id=f"permission_{uuid.uuid4().hex}",
                tool_call=call,
            )
            job.pending_permission = request
            job.permission_event.clear()
            job.state = SubagentState.WAITING_PERMISSION
            self._changed.notify_all()
            notify(
                "subagent_permission_request",
                {
                    "subagent_id": subagent_id,
                    "request_id": request.request_id,
                    "tool": call.name,
                    "arguments": call.arguments,
                },
            )
        while True:
            remaining = job.deadline - time.monotonic()
            if remaining <= 0:
                job.cancel_event.set()
                raise SubagentCancelled("Subagent timed out awaiting permission")
            if job.cancel_event.is_set():
                raise SubagentCancelled("Subagent cancelled awaiting permission")
            if job.permission_event.wait(timeout=min(0.2, remaining)):
                break
        with self._changed:
            if job.state in TERMINAL_STATES or job.cancel_event.is_set():
                raise SubagentCancelled(job.error or "Subagent cancelled")
            current = job.pending_permission
            if current is None or current.request_id != request.request_id:
                raise SubagentCancelled("Permission request was withdrawn")
            approved = bool(current.decision)
            feedback = current.feedback
            job.pending_permission = None
            job.state = SubagentState.RUNNING
            self._changed.notify_all()
            return approved, feedback

    def review_permission(
        self,
        subagent_id: str,
        request_id: str,
        approve: bool,
        feedback: str = "",
    ) -> dict[str, Any]:
        runtime_required = False
        policy_denied = False
        runtime_error = ""
        cancel_event: threading.Event | None = None
        deadline: Callable[[], float] | None = None
        call: ToolCall | None = None
        parent_turn_id = ""
        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None:
                raise KeyError(f"unknown subagent: {subagent_id}")
            request = job.pending_permission
            if job.state is not SubagentState.WAITING_PERMISSION or request is None:
                raise ValueError("subagent is not waiting for permission")
            if request.request_id != request_id:
                raise ValueError("permission request does not match subagent")
            if request.decision is not None:
                raise ValueError("permission request is already resolved")
            if request.reviewing:
                raise ValueError("permission request is already being reviewed")
            request.reviewing = True
            call = request.tool_call
            policy_decision = _permissions.decide(call)
            policy_denied = policy_decision is PermissionDecision.DENY
            runtime_required = (
                bool(approve)
                and policy_decision is PermissionDecision.ASK
            )
            cancel_event = job.cancel_event
            deadline = lambda: self._job_deadline(subagent_id)
            parent_turn_id = job.parent_turn_id
            if runtime_required:
                job.state = SubagentState.WAITING_HUMAN_PERMISSION
            self._changed.notify_all()

        runtime_approved = bool(approve) and not policy_denied
        if runtime_required and call is not None:
            try:
                with event_scope(
                    agent_type="subagent",
                    agent_id=subagent_id,
                    turn_id=parent_turn_id,
                ):
                    runtime_approved = _request_runtime_approval(
                        call,
                        cancel_event=cancel_event,
                        deadline=deadline,
                    )
            except Exception as error:
                runtime_approved = False
                runtime_error = f"{type(error).__name__}: {error}"

        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None:
                raise KeyError(f"unknown subagent: {subagent_id}")
            current = job.pending_permission
            if current is None or current.request_id != request_id:
                raise ValueError("permission request was withdrawn")
            if job.state in TERMINAL_STATES or job.cancel_event.is_set():
                raise ValueError("subagent stopped while permission was reviewed")
            actual_approve = bool(approve) and runtime_approved
            current.decision = actual_approve
            current.feedback = str(feedback or "")
            if policy_denied:
                current.feedback = (
                    current.feedback + " Denied by permission policy."
                ).strip()
            elif bool(approve) and not runtime_approved:
                current.feedback = (
                    current.feedback
                    + " Human/runtime approval was denied."
                    + (f" {runtime_error}" if runtime_error else "")
                ).strip()
            if actual_approve and runtime_required and call is not None:
                job.runtime_approved_calls.add(call.id)
            job.state = SubagentState.RUNNING
            job.permission_event.set()
            self._changed.notify_all()
            return {
                "subagent_id": subagent_id,
                "request_id": request_id,
                "approved": actual_approve,
                "status": "resuming",
            }

    def cancel(self, subagent_id: str) -> dict[str, Any]:
        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None:
                raise KeyError(f"unknown subagent: {subagent_id}")
            if job.state in TERMINAL_STATES:
                return self._snapshot_locked(job)
            job.cancel_reason = "cancelled"
            job.cancel_event.set()
            job.permission_event.set()
            if job.state is SubagentState.QUEUED:
                self._finish_locked(
                    job, SubagentState.CANCELLED, error="Cancelled before start"
                )
                self._dispatch_locked()
            else:
                job.state = SubagentState.CANCELLING
                self._changed.notify_all()
            return self._snapshot_locked(job)

    def cancel_turn(self, turn_id: str | None) -> None:
        parent_turn_id = self.parent_turn_id(turn_id)
        with self._changed:
            ids = [
                job.subagent_id
                for job in self.jobs.values()
                if job.parent_turn_id == parent_turn_id
                and job.state not in TERMINAL_STATES
            ]
        for subagent_id in ids:
            try:
                self.cancel(subagent_id)
            except KeyError:
                continue

    def check(self, subagent_id: str, *, consume_terminal: bool = True) -> dict[str, Any]:
        with self._changed:
            job = self.jobs.get(subagent_id)
            if job is None:
                raise KeyError(f"unknown subagent: {subagent_id}")
            snapshot = self._snapshot_locked(job)
            if consume_terminal and job.state in TERMINAL_STATES:
                job.terminal_delivered = True
                self._cleanup_locked()
            return snapshot

    def snapshot(self, turn_id: str | None = None) -> list[dict[str, Any]]:
        """Return observable job state without consuming terminal delivery."""

        with self._changed:
            jobs = list(self.jobs.values())
            if turn_id:
                parent_turn_id = self.parent_turn_id(turn_id)
                jobs = [
                    job for job in jobs if job.parent_turn_id == parent_turn_id
                ]
            return [self._snapshot_locked(job, detailed=True) for job in jobs]

    def wait(
        self,
        subagent_ids: list[str],
        timeout_seconds: float = MAX_WAIT_SECONDS,
    ) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(str(value) for value in subagent_ids))
        if not ids:
            raise ValueError("subagent_ids must not be empty")
        timeout = min(max(float(timeout_seconds), 0.0), MAX_WAIT_SECONDS)
        deadline = time.monotonic() + timeout
        with self._changed:
            missing = [value for value in ids if value not in self.jobs]
            if missing:
                raise KeyError("unknown subagent: " + ", ".join(missing))
            while not any(
                self.jobs[value].state in TERMINAL_STATES
                or self.jobs[value].state
                in {
                    SubagentState.WAITING_PERMISSION,
                    SubagentState.WAITING_HUMAN_PERMISSION,
                }
                for value in ids
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(timeout=remaining)
            snapshots = [self._snapshot_locked(self.jobs[value]) for value in ids]
            for value in ids:
                job = self.jobs[value]
                if job.state in TERMINAL_STATES:
                    job.terminal_delivered = True
            self._cleanup_locked()
            return snapshots

    def collect_updates(self, turn_id: str | None) -> list[str]:
        parent_turn_id = self.parent_turn_id(turn_id)
        notes: list[str] = []
        with self._changed:
            for job in list(self.jobs.values()):
                if job.parent_turn_id != parent_turn_id:
                    continue
                request = job.pending_permission
                if (
                    job.state is SubagentState.WAITING_PERMISSION
                    and request is not None
                    and not request.notified
                ):
                    request.notified = True
                    notes.append(self._permission_note(job, request))
                if job.state in TERMINAL_STATES and not job.terminal_delivered:
                    job.terminal_delivered = True
                    notes.append(self._terminal_note(job))
            self._cleanup_locked()
        return notes

    def barrier(
        self, turn_id: str | None, timeout_seconds: float = MAX_WAIT_SECONDS
    ) -> tuple[list[str], bool]:
        parent_turn_id = self.parent_turn_id(turn_id)
        timeout = min(max(float(timeout_seconds), 0.0), MAX_WAIT_SECONDS)
        deadline = time.monotonic() + timeout
        while True:
            notes = self.collect_updates(parent_turn_id)
            with self._changed:
                active = [
                    job
                    for job in self.jobs.values()
                    if job.parent_turn_id == parent_turn_id
                    and job.state not in TERMINAL_STATES
                ]
                if notes:
                    return notes, False
                if not active:
                    return [], True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    statuses = ", ".join(
                        f"{job.subagent_id}={job.state.value}" for job in active
                    )
                    return [
                        "<subagent_barrier>Active Subagents prevent this Turn "
                        f"from ending: {statuses}. Continue useful work or call "
                        "wait_subagents/cancel_subagent.</subagent_barrier>"
                    ], False
                self._changed.wait(timeout=remaining)

    def _snapshot_locked(
        self, job: SubagentJob, *, detailed: bool = False
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "subagent_id": job.subagent_id,
            "status": job.state.value,
            "task_id": job.task_id,
            "effects_committed": job.effects_committed,
        }
        if detailed:
            value.update(
                {
                    "parent_turn_id": job.parent_turn_id,
                    "description": job.description,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                }
            )
        if job.state in TERMINAL_STATES:
            value["result"] = job.result
            value["error"] = job.error
        if job.state in {
            SubagentState.WAITING_PERMISSION,
            SubagentState.WAITING_HUMAN_PERMISSION,
        }:
            request = job.pending_permission
            if request is not None:
                value["permission_request"] = {
                    "request_id": request.request_id,
                    "tool": request.tool_call.name,
                    "arguments": request.tool_call.arguments,
                }
        return value

    @staticmethod
    def _permission_note(job: SubagentJob, request: PermissionRequest) -> str:
        payload = {
            "subagent_id": job.subagent_id,
            "status": job.state.value,
            "request_id": request.request_id,
            "tool": request.tool_call.name,
            "arguments": request.tool_call.arguments,
        }
        return "<subagent_permission>" + json.dumps(
            payload, ensure_ascii=False
        ) + "</subagent_permission>"

    @staticmethod
    def _terminal_note(job: SubagentJob) -> str:
        payload = {
            "subagent_id": job.subagent_id,
            "status": job.state.value,
            "task_id": job.task_id,
            "result": job.result,
            "error": job.error,
        }
        return "<subagent_result>" + json.dumps(
            payload, ensure_ascii=False
        ) + "</subagent_result>"

    def _cleanup_locked(self) -> None:
        removable = [
            job_id
            for job_id, job in self.jobs.items()
            if job.state in TERMINAL_STATES
            and job.terminal_delivered
            and (
                job.thread is None
                or job.thread is threading.current_thread()
                or not job.thread.is_alive()
            )
        ]
        for job_id in removable:
            job = self.jobs.pop(job_id)
            if job.timer is not None:
                job.timer.cancel()

    def shutdown(self, timeout: float = 5.0) -> SubagentShutdownOutcome:
        with self._changed:
            self._accepting = False
            jobs = list(self.jobs.values())
            for job in jobs:
                if job.state not in TERMINAL_STATES:
                    job.cancel_event.set()
                    job.permission_event.set()
                    if job.state is SubagentState.QUEUED:
                        self._finish_locked(
                            job,
                            SubagentState.CANCELLED,
                            error="Cancelled during shutdown",
                        )
                    else:
                        job.state = SubagentState.CANCELLING
            self._changed.notify_all()
        deadline = time.monotonic() + max(0.0, timeout)
        for job in jobs:
            thread = job.thread
            if (
                thread is None
                or thread is threading.current_thread()
                or not thread.is_alive()
            ):
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._changed:
            live = tuple(
                job.subagent_id
                for job in self.jobs.values()
                if job.thread is not None and job.thread.is_alive()
            )
            for subagent_id in live:
                job = self.jobs.get(subagent_id)
                if job is not None and job.state not in TERMINAL_STATES:
                    job.state = SubagentState.STUCK
        return SubagentShutdownOutcome(not live, live)


_manager = SubagentManager()


def initialize_subagent_manager() -> None:
    _manager.initialize()


def shutdown_subagents(timeout: float = 5.0) -> SubagentShutdownOutcome:
    return _manager.shutdown(timeout)


def run_spawn_subagent(description: str, task_id: str | None = None) -> str:
    try:
        if task_id:
            from .tasks import load_task

            load_task(task_id)
        value = _manager.start(description, task_id=task_id)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)
    return json.dumps({"ok": True, **value}, ensure_ascii=False)


def run_check_subagent(subagent_id: str) -> str:
    try:
        value = _manager.check(subagent_id)
    except (KeyError, ValueError) as error:
        return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)
    return json.dumps({"ok": True, **value}, ensure_ascii=False)


def run_wait_subagents(
    subagent_ids: list[str], timeout_seconds: float = MAX_WAIT_SECONDS
) -> str:
    try:
        values = _manager.wait(subagent_ids, timeout_seconds)
    except (KeyError, ValueError, TypeError) as error:
        return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)
    return json.dumps({"ok": True, "subagents": values}, ensure_ascii=False)


def run_cancel_subagent(subagent_id: str) -> str:
    try:
        value = _manager.cancel(subagent_id)
    except KeyError as error:
        return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)
    return json.dumps({"ok": True, **value}, ensure_ascii=False)


def run_review_subagent_permission(
    subagent_id: str,
    request_id: str,
    approve: bool,
    feedback: str = "",
) -> str:
    try:
        value = _manager.review_permission(
            subagent_id, request_id, approve, feedback
        )
    except (KeyError, ValueError) as error:
        return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)
    return json.dumps({"ok": True, **value}, ensure_ascii=False)


def collect_subagent_updates(turn_id: str | None) -> list[str]:
    return _manager.collect_updates(turn_id)


def snapshot_subagents(turn_id: str | None = None) -> list[dict[str, Any]]:
    return _manager.snapshot(turn_id)


def wait_for_subagent_barrier(
    turn_id: str | None, timeout_seconds: float = MAX_WAIT_SECONDS
) -> tuple[list[str], bool]:
    return _manager.barrier(turn_id, timeout_seconds)


def cancel_turn_subagents(turn_id: str | None) -> None:
    _manager.cancel_turn(turn_id)
