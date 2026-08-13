from __future__ import annotations

import contextlib
import time
import uuid

from . import config
from .hooks import trigger_hooks
from .prompts import subagent_system_prompt
from .workspace import run_bash, run_edit, run_glob, run_read, run_write
from .telemetry import model_call_scope
from .trace import bind_run_context, current_run_context


client = None
MODEL = config.MODEL

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

    messages = [{"role": "user", "content": description}]
    parent_run = current_run_context()
    subagent_id = f"subagent:{uuid.uuid4().hex[:8]}"
    if client is None:
        raise RuntimeError("Subagent provider is not configured")
    scope = (
        bind_run_context(parent_run.child(subagent_id))
        if parent_run is not None
        else contextlib.nullcontext()
    )
    with scope:
        for _ in range(30):
            with model_call_scope("subagent"):
                response = client.create(
                    messages,
                    subagent_system_prompt(),
                    SUB_TOOLS,
                    config.DEFAULT_MAX_TOKENS,
                    model=MODEL,
                )
            messages.append({"role": "assistant", "content": response.content})
            if not has_tool_use(response.content):
                break
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                run = current_run_context()
                span_id = f"tool_{uuid.uuid4().hex}"
                if run is not None:
                    run.recorder.record(
                        "tool_requested",
                        {
                            "tool_call_id": block.id,
                            "tool_name": block.name,
                            "arguments": block.input,
                        },
                        span_id=span_id,
                        agent_id=run.agent_id,
                    )
                    run.recorder.record(
                        "tool_started",
                        {"tool_call_id": block.id, "tool_name": block.name},
                        span_id=span_id,
                        agent_id=run.agent_id,
                    )
                started = time.monotonic()
                blocked = trigger_hooks("PreToolUse", block)
                try:
                    if blocked:
                        output = str(blocked)
                    else:
                        handler = SUB_HANDLERS.get(block.name)
                        output = call_tool_handler(handler, block.input, block.name)
                        trigger_hooks("PostToolUse", block, output)
                except Exception as error:
                    output = f"Error: {type(error).__name__}: {error}"
                    if run is not None:
                        run.recorder.record(
                            "tool_error",
                            {
                                "exception_class": type(error).__name__,
                                "message": str(error),
                                "latency_ms": round(
                                    (time.monotonic() - started) * 1000, 3
                                ),
                            },
                            span_id=span_id,
                            agent_id=run.agent_id,
                        )
                else:
                    if run is not None:
                        output_ref = run.recorder.store_artifact(
                            str(output),
                            media_type="text/plain",
                            source=f"subagent_tool:{block.name}",
                            suffix=".txt",
                        )
                        run.recorder.record(
                            "tool_result",
                            {
                                "success": not blocked
                                and not str(output).startswith("Error:"),
                                "latency_ms": round(
                                    (time.monotonic() - started) * 1000, 3
                                ),
                                "output_artifact": output_ref.as_dict(),
                            },
                            span_id=span_id,
                            agent_id=run.agent_id,
                        )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
            messages.append({"role": "user", "content": results})
    for message in reversed(messages):
        if message["role"] == "assistant":
            text = extract_text(message["content"])
            if text:
                return text
    return "Subagent finished without a text summary."
