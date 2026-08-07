from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any, Callable


class HookEvent(str, Enum):
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"


class HookManager:
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
            handle.write(json.dumps({"event": event, "time": time.time(), **payload}, default=str) + "\n")

    hooks.register(HookEvent.USER_PROMPT_SUBMIT, lambda query, **_: log("user_prompt", query=query))
    hooks.register(HookEvent.PRE_TOOL_USE, lambda call, **_: log("pre_tool", tool=call.name))
    hooks.register(HookEvent.POST_TOOL_USE, lambda call, output, **_: log("post_tool", tool=call.name, size=len(str(output))))
    hooks.register(HookEvent.STOP, lambda messages, **_: log("stop", messages=len(messages)))

