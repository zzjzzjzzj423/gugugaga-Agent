from __future__ import annotations

import re
from enum import Enum
from typing import Callable

from .models import ToolCall


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionPolicy:
    _dangerous = (
        r"\brm\s+-[^\n]*r[^\n]*f\b",
        r"\bgit\s+(reset\s+--hard|clean\s+-|checkout\s+--|restore\s+--source)",
        r"\b(sudo|shutdown|reboot|format|diskpart)\b",
        r"\b(del|rmdir|remove-item)\b[^\n]*(/s|-recurse)",
        r">\s*(/dev/|[A-Za-z]:\\Windows)",
    )

    def decide(self, call: ToolCall) -> PermissionDecision:
        if call.name != "bash":
            return PermissionDecision.ALLOW
        command = str(call.arguments.get("command", ""))
        if "\x00" in command:
            return PermissionDecision.DENY
        if any(re.search(pattern, command, re.IGNORECASE) for pattern in self._dangerous):
            return PermissionDecision.ASK
        return PermissionDecision.ALLOW

    def approve(self, call: ToolCall, callback: Callable[[ToolCall], bool] | None) -> bool:
        decision = self.decide(call)
        if decision is PermissionDecision.ALLOW:
            return True
        if decision is PermissionDecision.DENY or callback is None:
            return False
        return bool(callback(call))

