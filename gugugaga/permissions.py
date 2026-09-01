from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from typing import Callable

from .models import ToolCall
from .observability import current_event_context, notify, sanitize


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


@dataclass
class PermissionRequest:
    permission_id: str
    call: ToolCall
    source: dict[str, Any]
    created_at: float
    expires_at: float
    status: str = "pending"
    approve: bool | None = None
    feedback: str = ""
    event: threading.Event = field(default_factory=threading.Event, repr=False)


class PermissionBroker:
    """Single-use human approval broker shared by Web and runtime workers."""

    def __init__(self, timeout_seconds: float = 120.0):
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._lock = threading.RLock()
        self._requests: dict[str, PermissionRequest] = {}
        self._accepting = True

    def callback(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        return self.request(
            call,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )

    def request(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        now = time.monotonic()
        requested_timeout = (
            self.timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        timeout = min(self.timeout_seconds, max(0.1, requested_timeout))
        request = PermissionRequest(
            permission_id=f"permission_{uuid.uuid4().hex}",
            call=ToolCall(
                str(call.id), str(call.name), copy.deepcopy(call.arguments or {})
            ),
            source=current_event_context(),
            created_at=now,
            expires_at=now + timeout,
        )
        with self._lock:
            if not self._accepting:
                return False
            self._requests[request.permission_id] = request
        notify("permission_requested", self._public(request))
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self._resolve_internal(request, False, "cancelled", "request cancelled")
                break
            remaining = request.expires_at - time.monotonic()
            if remaining <= 0:
                self._resolve_internal(request, False, "expired", "approval timed out")
                break
            if request.event.wait(timeout=min(0.2, remaining)):
                break
        with self._lock:
            return request.status == "approved" and request.approve is True

    def review(
        self, permission_id: str, approve: bool, feedback: str = ""
    ) -> dict[str, Any]:
        with self._lock:
            request = self._requests.get(str(permission_id))
            if request is None:
                raise KeyError(f"unknown permission request: {permission_id}")
            if request.status != "pending":
                raise ValueError(f"permission request is already {request.status}")
            if time.monotonic() >= request.expires_at:
                self._resolve_locked(request, False, "expired", "approval timed out")
                raise ValueError("permission request has expired")
            status = "approved" if bool(approve) else "rejected"
            self._resolve_locked(request, bool(approve), status, feedback)
            return self._public(request)

    def _resolve_internal(
        self, request: PermissionRequest, approve: bool, status: str, feedback: str
    ) -> None:
        with self._lock:
            if request.status == "pending":
                self._resolve_locked(request, approve, status, feedback)

    def _resolve_locked(
        self, request: PermissionRequest, approve: bool, status: str, feedback: str
    ) -> None:
        request.approve = bool(approve)
        request.status = status
        request.feedback = str(feedback or "")
        request.event.set()
        notify("permission_resolved", self._public(request))

    def pending(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        expired: list[PermissionRequest] = []
        with self._lock:
            for request in self._requests.values():
                if request.status == "pending" and now >= request.expires_at:
                    expired.append(request)
            for request in expired:
                self._resolve_locked(request, False, "expired", "approval timed out")
            return [
                self._public(request)
                for request in self._requests.values()
                if request.status == "pending"
            ]

    def close(self) -> None:
        with self._lock:
            self._accepting = False
            for request in self._requests.values():
                if request.status == "pending":
                    self._resolve_locked(
                        request, False, "cancelled", "permission broker closed"
                    )

    @staticmethod
    def _public(request: PermissionRequest) -> dict[str, Any]:
        remaining = max(0.0, request.expires_at - time.monotonic())
        return sanitize(
            {
                "permission_id": request.permission_id,
                "tool_call_id": request.call.id,
                "tool": request.call.name,
                "arguments": request.call.arguments,
                "source": request.source,
                "status": request.status,
                "approve": request.approve,
                "feedback": request.feedback,
                "remaining_seconds": round(remaining, 1),
            }
        )
