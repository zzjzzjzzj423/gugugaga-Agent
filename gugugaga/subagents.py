from __future__ import annotations

import time
import uuid

from . import config
from .hooks import trigger_hooks
from .observability import event_scope, notify, record_llm_call
from .prompts import subagent_system_prompt
from .workspace import run_bash, run_edit, run_glob, run_read, run_write


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
    if client is None:
        raise RuntimeError("Subagent provider is not configured")
    agent_id = f"subagent_{uuid.uuid4().hex}"
    with event_scope(agent_type="subagent", agent_id=agent_id):
        notify("subagent_start", {"description": description})
        try:
            for iteration in range(1, 31):
                with event_scope(iteration=iteration):
                    response = record_llm_call(
                        client,
                        messages,
                        subagent_system_prompt(),
                        SUB_TOOLS,
                        config.DEFAULT_MAX_TOKENS,
                        model=MODEL,
                        call_type="subagent",
                    )
                messages.append({"role": "assistant", "content": response.content})
                if not has_tool_use(response.content):
                    break
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
                        status = "ok"
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
            for message in reversed(messages):
                if message["role"] == "assistant":
                    text = extract_text(message["content"])
                    if text:
                        notify("subagent_end", {"reply": text})
                        return text
            reply = "Subagent finished without a text summary."
            notify("subagent_end", {"reply": reply})
            return reply
        except Exception as error:
            notify(
                "subagent_error",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            raise
