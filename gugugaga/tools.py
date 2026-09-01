from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from .cron import run_cancel_cron, run_list_crons, run_schedule_cron
from .models import ToolSpec
from .observability import current_event_context
from .skills import load_skill
from .tasks import (
    run_create_task,
    run_get_task,
    run_list_tasks,
    run_todo_write,
)
from .teams import (
    run_check_inbox,
    run_request_plan,
    run_request_shutdown,
    run_review_plan,
    run_send_message,
    run_spawn_teammate,
    run_restart_teammate,
    run_stop_teammate,
)
from .workspace import run_bash, run_edit, run_glob, run_read, run_write
from .web_search import run_web_search


def call_tool_handler(handler, args: dict, name: str) -> str:
    if not handler:
        return f"Unknown: {name}"
    try:
        return handler(**(args or {}))
    except TypeError as error:
        return f"Error: {error}"


def run_compact(focus: str = "") -> str:
    """Signal the special message-mutating branch in the agent loop."""
    return "[Compaction requested.]"


def run_lead_create_task(
    subject: str,
    description: str = "",
    blockedBy: list[str] | None = None,
) -> str:
    context = current_event_context()
    if context.get("source") == "team_inbox":
        return (
            "Error: an automatic Lead inbox Turn cannot create tasks; "
            "wait for an explicit user Turn"
        )
    return run_create_task(subject, description, blockedBy)


def run_lead_claim_task(task_id: str) -> str:
    del task_id
    return (
        "Error: Lead cannot claim Task System work. Assign the task to an "
        "online idle Team Agent from the Task UI."
    )


def run_lead_complete_task(task_id: str) -> str:
    del task_id
    return (
        "Error: Lead cannot complete Task System work. Only the Team Agent "
        "that owns the task may complete it."
    )


TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "run_in_background": {"type": "boolean"},
                "write_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
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
                "include_hash": {"type": "boolean"},
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
                "expected_sha256": {"type": "string"},
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
                "expected_sha256": {"type": "string"},
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
    {
        "name": "todo_write",
        "description": "Create and manage a task list for the current session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                ],
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
    {
        "name": "spawn_subagent",
        "description": (
            "Start a focused Subagent asynchronously inside the current Turn. "
            "Returns its subagent_id immediately. The Turn cannot finish until "
            "every started Subagent reaches a terminal state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "task_id": {"type": "string"},
            },
            "required": ["description"],
        },
    },
    {
        "name": "check_subagent",
        "description": "Read one Subagent's current status or consume its result.",
        "input_schema": {
            "type": "object",
            "properties": {"subagent_id": {"type": "string"}},
            "required": ["subagent_id"],
        },
    },
    {
        "name": "wait_subagents",
        "description": (
            "Wait up to 30 seconds for any listed Subagent to finish or request "
            "permission, then return current structured states."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subagent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 30,
                },
            },
            "required": ["subagent_ids"],
        },
    },
    {
        "name": "cancel_subagent",
        "description": "Cooperatively cancel one active Subagent.",
        "input_schema": {
            "type": "object",
            "properties": {"subagent_id": {"type": "string"}},
            "required": ["subagent_id"],
        },
    },
    {
        "name": "review_subagent_permission",
        "description": (
            "Approve or reject the exact pending privileged tool request from a "
            "Subagent. Validate its tool and arguments before approving."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string"},
                "request_id": {"type": "string"},
                "approve": {"type": "boolean"},
                "feedback": {"type": "string"},
            },
            "required": ["subagent_id", "request_id", "approve"],
        },
    },
    {
        "name": "load_skill",
        "description": "Load the full content of a skill by name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "compact",
        "description": (
            "Summarize earlier conversation and continue with compacted context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"focus": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the current public web with Tavily. Use for recent, changing, "
            "or externally verifiable information and cite returned source URLs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 400},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_task",
        "description": "Create a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "blockedBy": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["subject"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List all tasks.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_task",
        "description": "Get full task details.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "claim_task",
        "description": (
            "Task ownership is reserved for Team Agents. Lead cannot claim "
            "tasks; use the Task UI to assign one to an online idle teammate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "complete_task",
        "description": (
            "Task completion is reserved for the owning Team Agent. Lead "
            "cannot complete Task System work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "schedule_cron",
        "description": (
            "Schedule a cron job. cron is 5-field: min hour dom month dow. "
            "For one-shot reminders, compute the target minute and set "
            "recurring=false."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cron": {"type": "string"},
                "prompt": {"type": "string"},
                "recurring": {"type": "boolean"},
                "durable": {"type": "boolean"},
            },
            "required": ["cron", "prompt"],
        },
    },
    {
        "name": "list_crons",
        "description": "List registered cron jobs.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "cancel_cron",
        "description": "Cancel a cron job by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "spawn_teammate",
        "description": (
            "Register an autonomous teammate and leave it idle. Work begins only "
            "after a Task System assignment, or an auto-claim when workspace "
            "Team auto-claim is enabled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["name", "role", "prompt"],
        },
    },
    {
        "name": "send_message",
        "description": (
            "Send a message from Lead to an active teammate. Lead is the current "
            "main Agent and cannot message itself with this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["to", "content"],
        },
    },
    {
        "name": "stop_teammate",
        "description": "Stop one active Team Agent at the user's request.",
        "input_schema": {
            "type": "object",
            "properties": {"teammate": {"type": "string"}},
            "required": ["teammate"],
        },
    },
    {
        "name": "restart_teammate",
        "description": (
            "Restart one stopped Team Agent from its persisted workspace profile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"teammate": {"type": "string"}},
            "required": ["teammate"],
        },
    },
    {
        "name": "check_inbox",
        "description": "Check inbox for messages and protocol responses.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "request_shutdown",
        "description": "Request a teammate to shut down.",
        "input_schema": {
            "type": "object",
            "properties": {"teammate": {"type": "string"}},
            "required": ["teammate"],
        },
    },
    {
        "name": "request_plan",
        "description": "Ask a teammate to submit a plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "teammate": {"type": "string"},
                "task": {"type": "string"},
            },
            "required": ["teammate", "task"],
        },
    },
    {
        "name": "review_plan",
        "description": "Approve or reject a submitted plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "approve": {"type": "boolean"},
                "feedback": {"type": "string"},
            },
            "required": ["request_id", "approve"],
        },
    },
]


from .subagents import (
    run_cancel_subagent,
    run_check_subagent,
    run_review_subagent_permission,
    run_spawn_subagent,
    run_wait_subagents,
)


TOOL_HANDLERS: dict[str, Callable] = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "web_search": run_web_search,
    "todo_write": run_todo_write,
    "spawn_subagent": run_spawn_subagent,
    "check_subagent": run_check_subagent,
    "wait_subagents": run_wait_subagents,
    "cancel_subagent": run_cancel_subagent,
    "review_subagent_permission": run_review_subagent_permission,
    "load_skill": load_skill,
    "compact": run_compact,
    "create_task": run_lead_create_task,
    "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_lead_claim_task,
    "complete_task": run_lead_complete_task,
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
    "spawn_teammate": run_spawn_teammate,
    "send_message": run_send_message,
    "stop_teammate": run_stop_teammate,
    "restart_teammate": run_restart_teammate,
    "check_inbox": run_check_inbox,
    "request_shutdown": run_request_shutdown,
    "request_plan": run_request_plan,
    "review_plan": run_review_plan,
}

BUILTIN_TOOLS = TOOL_DEFINITIONS
BUILTIN_HANDLERS = TOOL_HANDLERS


class ToolRegistry:
    """Compatibility adapter for the pre-migration runtime."""

    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[..., Any],
    ) -> None:
        self._specs[name] = ToolSpec(name, description, parameters)
        self._handlers[name] = handler

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return str(handler(**arguments))
        except Exception as error:
            return f"Error: {type(error).__name__}: {error}"


class WorkspaceTools:
    """Compatibility adapter for the pre-migration runtime."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def _safe(self, value: str) -> Path:
        path = (self.workspace / value).resolve()
        try:
            path.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError(f"path escapes workspace: {value}") from error
        return path

    def bash(self, command: str, timeout: int = 120) -> str:
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=min(max(timeout, 1), 300),
        )
        output = (result.stdout + result.stderr).strip()
        return output[:100_000] or "(no output)"

    def read_file(self, path: str, limit: int | None = None) -> str:
        try:
            lines = self._safe(path).read_text(encoding="utf-8").splitlines()
            if limit is not None and limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
            return "\n".join(lines)
        except Exception as error:
            return f"Error: {error}"

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._safe(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} characters to {path}"
        except Exception as error:
            return f"Error: {error}"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            target = self._safe(path)
            content = target.read_text(encoding="utf-8")
            if old_text not in content:
                return f"Error: text not found in {path}"
            target.write_text(
                content.replace(old_text, new_text, 1), encoding="utf-8"
            )
            return f"Edited {path}"
        except Exception as error:
            return f"Error: {error}"

    def glob(self, pattern: str) -> str:
        try:
            matches = [
                path.relative_to(self.workspace).as_posix()
                for path in self.workspace.glob(pattern)
                if path.is_file()
            ]
            return "\n".join(sorted(matches)[:1000]) or "No matches"
        except Exception as error:
            return f"Error: {error}"

    def register_into(self, registry: ToolRegistry) -> None:
        def obj(props, required=()):
            return {
                "type": "object",
                "properties": props,
                "required": list(required),
            }

        registry.register(
            "bash",
            "Run a shell command in the workspace",
            obj(
                {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                ["command"],
            ),
            self.bash,
        )
        registry.register(
            "read_file",
            "Read a UTF-8 file",
            obj(
                {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                ["path"],
            ),
            self.read_file,
        )
        registry.register(
            "write_file",
            "Write a UTF-8 file",
            obj(
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            ),
            self.write_file,
        )
        registry.register(
            "edit_file",
            "Replace exact text once",
            obj(
                {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                ["path", "old_text", "new_text"],
            ),
            self.edit_file,
        )
        registry.register(
            "glob",
            "List files matching a glob",
            obj({"pattern": {"type": "string"}}, ["pattern"]),
            self.glob,
        )
