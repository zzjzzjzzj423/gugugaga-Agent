from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .models import ToolCall
from .planning import Task, TaskStore


class Mailbox:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in "_-.")
        if not safe or safe != name:
            raise ValueError(f"invalid agent name: {name}")
        return self.directory / f"{safe}.jsonl"

    def send(
        self,
        sender: str,
        target: str,
        content: str,
        message_type: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        message = {
            "from": sender,
            "to": target,
            "content": content,
            "type": message_type,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        with self._lock, self._path(target).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")

    def drain(self, name: str) -> list[dict[str, Any]]:
        path = self._path(name)
        with self._lock:
            if not path.exists():
                return []
            items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            path.write_text("", encoding="utf-8")
            return items

    def peek(self, name: str) -> bool:
        path = self._path(name)
        with self._lock:
            return path.exists() and bool(path.read_text(encoding="utf-8").strip())


class ProtocolError(RuntimeError):
    pass


@dataclass
class ProtocolRequest:
    id: str
    type: str
    sender: str
    target: str
    payload: str
    status: str = "pending"
    feedback: str = ""


class ProtocolStore:
    RESPONSE_TYPES = {
        "plan_approval": "plan_approval_response",
        "shutdown": "shutdown_response",
        "permission": "permission_response",
    }

    def __init__(self):
        self.requests: dict[str, ProtocolRequest] = {}
        self._lock = threading.Lock()

    def request(self, request_type: str, sender: str, target: str, payload: str) -> ProtocolRequest:
        if request_type not in self.RESPONSE_TYPES:
            raise ProtocolError(f"unknown protocol type: {request_type}")
        item = ProtocolRequest(f"req_{uuid.uuid4().hex[:10]}", request_type, sender, target, payload)
        with self._lock:
            self.requests[item.id] = item
        return item

    def resolve(self, request_id: str, response_type: str, approve: bool, feedback: str = "") -> ProtocolRequest:
        with self._lock:
            item = self.requests.get(request_id)
            if item is None:
                raise ProtocolError(f"unknown request: {request_id}")
            expected = self.RESPONSE_TYPES[item.type]
            if response_type != expected:
                raise ProtocolError(f"expected {expected}, got {response_type}")
            item.status = "approved" if approve else "rejected"
            item.feedback = feedback
            return item

    def get(self, request_id: str) -> ProtocolRequest:
        return self.requests[request_id]


class TeamManager:
    def __init__(
        self,
        mailbox: Mailbox,
        tasks: TaskStore,
        protocols: ProtocolStore,
        runtime_factory: Callable[[str, str], Any],
        poll_seconds: float = 1.0,
        idle_timeout: float = 30.0,
    ):
        self.mailbox = mailbox
        self.tasks = tasks
        self.protocols = protocols
        self.runtime_factory = runtime_factory
        self.poll_seconds = poll_seconds
        self.idle_timeout = idle_timeout
        self.members: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def claim_next(self, name: str) -> Task | None:
        for task in self.tasks.list():
            if task.status == "pending" and not task.owner and self.tasks.can_start(task):
                if self.tasks.claim(task.id, name).startswith("Claimed"):
                    return self.tasks.get(task.id)
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        with self._lock:
            if name in self.members and self.members[name]["status"] != "stopped":
                return f"Error: teammate '{name}' already exists"
            self.members[name] = {"role": role, "status": "working", "stop": threading.Event()}

        def run():
            runtime = self.runtime_factory(name, role)
            try:
                result = runtime.run_turn(prompt)
                self.mailbox.send(name, "lead", result, "result")
                idle_started = time.time()
                while not self.members[name]["stop"].is_set() and time.time() - idle_started < self.idle_timeout:
                    inbox = self.mailbox.drain(name)
                    if inbox:
                        idle_started = time.time()
                    should_stop = False
                    for message in inbox:
                        kind = message.get("type", "message")
                        if kind == "shutdown_request":
                            request_id = message.get("metadata", {}).get("request_id", "")
                            self.protocols.resolve(request_id, "shutdown_response", True)
                            self.mailbox.send(name, "lead", "Shutdown approved", "shutdown_response", {"request_id": request_id, "approve": True})
                            should_stop = True
                            break
                        if kind == "request_plan":
                            plan = runtime.run_turn(f"Create a concise plan for: {message['content']}")
                            request = self.protocols.request("plan_approval", name, "lead", plan)
                            self.mailbox.send(name, "lead", plan, "plan_approval_request", {"request_id": request.id})
                        elif kind == "permission_response":
                            continue
                        else:
                            answer = runtime.run_turn(message["content"])
                            self.mailbox.send(name, "lead", answer, "message")
                    if should_stop:
                        break
                    task = self.claim_next(name)
                    if task:
                        idle_started = time.time()
                        answer = runtime.run_turn(f"Complete task {task.id}: {task.subject}\n{task.description}")
                        self.mailbox.send(name, "lead", answer, "task_result", {"task_id": task.id})
                    with self._lock:
                        self.members[name]["status"] = "idle"
                    time.sleep(self.poll_seconds)
            finally:
                with self._lock:
                    self.members[name]["status"] = "stopped"

        thread = threading.Thread(target=run, name=f"teammate-{name}", daemon=True)
        self.members[name]["thread"] = thread
        thread.start()
        return f"Spawned teammate '{name}' as {role}"

    def send(self, target: str, content: str) -> str:
        self.mailbox.send("lead", target, content)
        return f"Sent message to {target}"

    def check_inbox(self) -> list[dict[str, Any]]:
        return self.mailbox.drain("lead")

    def request_shutdown(self, teammate: str) -> str:
        request = self.protocols.request("shutdown", "lead", teammate, "Graceful shutdown")
        self.mailbox.send("lead", teammate, request.payload, "shutdown_request", {"request_id": request.id})
        return request.id

    def request_plan(self, teammate: str, task: str) -> str:
        self.mailbox.send("lead", teammate, task, "request_plan")
        return f"Requested plan from {teammate}"

    def request_permission(self, teammate: str, call: ToolCall) -> str:
        payload = json.dumps(
            {"tool": call.name, "arguments": call.arguments}, ensure_ascii=False
        )
        request = self.protocols.request("permission", teammate, "lead", payload)
        self.mailbox.send(
            teammate,
            "lead",
            payload,
            "permission_request",
            {"request_id": request.id, "tool_call_id": call.id},
        )
        return request.id

    def await_permission(
        self, teammate: str, call: ToolCall, timeout: float = 60.0
    ) -> bool:
        request_id = self.request_permission(teammate, call)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.protocols.get(request_id).status
            if status == "approved":
                return True
            if status == "rejected":
                return False
            time.sleep(min(self.poll_seconds, 0.1))
        return False

    def review_permission(
        self, request_id: str, approve: bool, feedback: str = ""
    ) -> str:
        request = self.protocols.resolve(
            request_id, "permission_response", approve, feedback
        )
        self.mailbox.send(
            "lead",
            request.sender,
            feedback or request.status,
            "permission_response",
            {"request_id": request.id, "approve": approve},
        )
        return f"Permission {request.status}: {request.id}"

    def review_plan(self, request_id: str, approve: bool, feedback: str = "") -> str:
        request = self.protocols.resolve(request_id, "plan_approval_response", approve, feedback)
        self.mailbox.send("lead", request.sender, feedback or request.status, "plan_approval_response", {"request_id": request.id, "approve": approve})
        return f"Plan {request.status}: {request.id}"

    def status(self) -> str:
        with self._lock:
            if not self.members:
                return "No teammates."
            return "\n".join(f"{name} ({item['role']}): {item['status']}" for name, item in sorted(self.members.items()))

    def stop_all(self) -> None:
        with self._lock:
            for item in self.members.values():
                item["stop"].set()
