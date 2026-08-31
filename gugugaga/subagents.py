from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Callable

from . import config
from .hooks import trigger_hooks
from .models import ToolCall
from .observability import event_scope, notify, record_llm_call
from .permissions import PermissionPolicy
from .prompts import subagent_system_prompt
from .provider import is_context_length_error
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


def reset_subagent_runtime() -> None:
    configure_subagent_runtime(None, model=config.MODEL)

SUB_TOOLS = [
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
        "description": "Read file contents.",
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
        "description": "Write content to a file.",
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
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
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
]

SUB_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
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
    return any(
        getattr(block, "type", None) == "tool_use" for block in content
    )


def spawn_subagent(description: str) -> str:
    from .tools import call_tool_handler
    from .context_modes import (
        CompressionReason,
        ContextModeError,
        RequestContext,
        create_child_context_coordinator,
    )

    messages = [{"role": "user", "content": description}]
    if client is None:
        raise RuntimeError("Subagent provider is not configured")
    agent_id = f"subagent_{uuid.uuid4().hex}"
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
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason == "max_tokens":
                    raise RuntimeError(
                        "Subagent response reached the output-token limit"
                    )
                if not has_tool_use(response.content):
                    reply = extract_text(response.content)
                    if not reply:
                        raise RuntimeError(
                            "Subagent finished without a text summary"
                        )
                    notify("subagent_end", {"reply": reply})
                    return reply
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    started = time.monotonic()
                    blocked = trigger_hooks("PreToolUse", block)
                    if blocked:
                        output = str(blocked)
                        status = "blocked"
                    else:
                        call = ToolCall(block.id, block.name, block.input)
                        if not _permissions.approve(call, _approval_callback):
                            output = (
                                f"Permission denied for tool '{block.name}'. "
                                "Choose a safer approach."
                            )
                            status = "denied"
                        else:
                            handler = SUB_HANDLERS.get(block.name)
                            try:
                                output = call_tool_handler(
                                    handler, block.input, block.name
                                )
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
                                if str(output).startswith(("Error:", "Unknown:"))
                                else "ok"
                            )
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
            raise RuntimeError(
                f"Subagent exceeded maximum rounds ({_max_rounds})"
            )
        except Exception as error:
            notify(
                "subagent_error",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            raise
        finally:
            if coordinator is not None:
                coordinator.close()
