from __future__ import annotations

import threading
import uuid
from typing import Callable

from .hooks import trigger_hooks
from .models import ToolCall
from .tools import call_tool_handler


_bg_counter = 0
background_tasks: dict[str, dict] = {}
background_results: dict[str, str] = {}
background_lock = threading.Lock()


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    if tool_name != "bash":
        return False
    command = tool_input.get("command", "").lower()
    slow_keywords = [
        "install",
        "build",
        "test",
        "deploy",
        "compile",
        "docker build",
        "pip install",
        "npm install",
        "cargo build",
        "pytest",
        "make",
    ]
    return any(keyword in command for keyword in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    if tool_name != "bash":
        return False
    return bool(tool_input.get("run_in_background")) or is_slow_operation(
        tool_name, tool_input
    )


def start_background_task(block, handlers: dict) -> str:
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    command = block.input.get("command", block.name)

    def worker():
        handler = handlers.get(block.name)
        result = call_tool_handler(handler, block.input, block.name)
        trigger_hooks("PostToolUse", block, result)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = str(result)

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": command,
            "status": "running",
        }
    threading.Thread(target=worker, daemon=True).start()
    print(f"  \033[33m[background] {bg_id}: {str(command)[:60]}\033[0m")
    return bg_id


def dispatch_background_task(block, handlers: dict) -> str | None:
    if not should_run_background(block.name, block.input):
        return None
    bg_id = start_background_task(block, handlers)
    return (
        f"[Background task {bg_id} started] "
        "Result will arrive as a task_notification."
    )


def collect_background_results() -> list[str]:
    with background_lock:
        ready = [
            bg_id
            for bg_id, task in background_tasks.items()
            if task["status"] == "completed"
        ]
    notifications = []
    for bg_id in ready:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>"
        )
    return notifications


class BackgroundManager:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self._notifications: list[str] = []
        self._lock = threading.Lock()

    def start(self, call: ToolCall, function: Callable[[], str]) -> str:
        job_id = f"bg_{uuid.uuid4().hex[:8]}"
        self.jobs[job_id] = {"status": "running", "tool": call.name}

        def run():
            try:
                output = str(function())
                status = "completed"
            except Exception as error:
                output, status = f"Error: {type(error).__name__}: {error}", "failed"
            with self._lock:
                self.jobs[job_id].update(status=status, result=output)
                self._notifications.append(f"<task_notification id='{job_id}' status='{status}'>{output}</task_notification>")

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._notifications)

    def drain(self) -> list[str]:
        with self._lock:
            items, self._notifications = self._notifications, []
            return items


from .cron import CronJob, CronScheduler, cron_matches, validate_cron
