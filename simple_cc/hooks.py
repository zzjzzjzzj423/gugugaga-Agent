from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any, Callable

from . import config
from .workspace import safe_path


HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def permission_hook(block):
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                return f"Permission denied: '{pattern}' is on the deny list"
        if any(token in command for token in DESTRUCTIVE):
            print("\n\033[33m[permission] destructive command\033[0m")
            print(f"  {command}")
            choice = input("  Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("path", "")
        try:
            safe_path(path)
        except Exception:
            return f"Permission denied: path escapes workspace: {path}"
    return None


def log_hook(block):
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None


def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(
            f"\033[33m[HOOK] large output from {block.name}: "
            f"{len(str(output))} chars\033[0m"
        )
    return None


def user_prompt_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: {config.WORKDIR}\033[0m")
    return None


def stop_hook(messages: list):
    tool_count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            tool_count += sum(
                1
                for item in content
                if isinstance(item, dict) and item.get("type") == "tool_result"
            )
    print(f"\033[90m[HOOK] Stop: {tool_count} tool result(s)\033[0m")
    return None


register_hook("UserPromptSubmit", user_prompt_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", stop_hook)


class HookEvent(str, Enum):
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"


class HookManager:
    """Compatibility adapter for the pre-migration runtime."""

    def __init__(self):
        self._hooks: dict[HookEvent, list[Callable[..., Any]]] = defaultdict(list)

    def register(self, event: HookEvent, callback: Callable[..., Any]) -> None:
        self._hooks[event].append(callback)

    def trigger(self, event: HookEvent, **payload: Any) -> list[Any]:
        return [callback(**payload) for callback in self._hooks[event]]


def install_audit_hooks(hooks: HookManager, log_path) -> None:
    def log(event: str, **payload):
        import json
        import time

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": event, "time": time.time(), **payload}, default=str
                )
                + "\n"
            )

    hooks.register(
        HookEvent.USER_PROMPT_SUBMIT,
        lambda query, **_: log("user_prompt", query=query),
    )
    hooks.register(
        HookEvent.PRE_TOOL_USE,
        lambda call, **_: log("pre_tool", tool=call.name),
    )
    hooks.register(
        HookEvent.POST_TOOL_USE,
        lambda call, output, **_: log(
            "post_tool", tool=call.name, size=len(str(output))
        ),
    )
    hooks.register(
        HookEvent.STOP,
        lambda messages, **_: log("stop", messages=len(messages)),
    )

