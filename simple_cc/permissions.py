from __future__ import annotations

from enum import Enum
from typing import Callable

from .models import ToolCall


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionPolicy:
    def decide(self, call: ToolCall) -> PermissionDecision:
        if call.name not in {"bash", "background_run"}:
            return PermissionDecision.ALLOW
        command = str(call.arguments.get("command", ""))
        if "\x00" in command:
            return PermissionDecision.DENY
        # A shell can invoke arbitrary interpreters and redirect to absolute paths.
        # Regex classification cannot prove workspace containment, so every shell
        # command requires an explicit decision from the lead.
        return PermissionDecision.ASK

    def approve(self, call: ToolCall, callback: Callable[[ToolCall], bool] | None) -> bool:
        decision = self.decide(call)
        if decision is PermissionDecision.ALLOW:
            return True
        if decision is PermissionDecision.DENY or callback is None:
            return False
        return bool(callback(call))
