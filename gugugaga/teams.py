from __future__ import annotations

import contextlib
import json
import os
import random
import re
import threading
import time
import uuid
from collections import deque
from contextvars import copy_context
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from . import config
from .hooks import trigger_hooks
from .interactions import AgentInteraction, interaction_broker
from .models import ToolCall
from .observability import (
    current_event_context,
    event_scope,
    notify,
    record_llm_call,
)
from .permissions import PermissionPolicy
from .mutations import mutation_actor_scope
from .provider import is_context_length_error
from .stateio import atomic_write_text, interprocess_lock
from .skills import load_skill
from .tasks import (
    append_task_intervention,
    assign_task,
    can_start,
    claim_task,
    complete_task,
    create_queued_task,
    get_task_json,
    interrupt_task,
    list_tasks,
    load_task,
)
from .web_search import run_web_search
from .workspace import run_bash, run_edit, run_glob, run_read, run_write

if TYPE_CHECKING:
    from .context_modes import SessionContextCoordinator


# S15-S17 source-compatible team communication. Paths are resolved from
# config at call time so every teammate sees the selected shared workspace.
_mailbox_lock = threading.RLock()
_AGENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_LEAD_AGENT_NAME = "lead"
_LEAD_AGENT_ALIASES = frozenset({"lead", "leader", "main"})
_RESERVED_TEAMMATE_NAMES = _LEAD_AGENT_ALIASES
_lead_inbox_event = threading.Event()
_LEAD_WAKE_MESSAGE_TYPES = frozenset(
    {"result", "error", "plan_approval_request"}
)


def _team_communications_path() -> Path:
    return config.WORKDIR / ".gugugaga" / "team-communications.jsonl"


def _record_team_communication(message: dict[str, Any]) -> None:
    """Persist routing metadata for the Team graph without storing message text."""

    path = _team_communications_path()
    value = {
        "id": message["id"],
        "from": message["from"],
        "to": message["to"],
        "type": message["type"],
        "ts": message["ts"],
        "task_id": message.get("metadata", {}).get("task_id"),
        "interaction_id": message.get("metadata", {}).get("interaction_id"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with interprocess_lock(path.with_suffix(".lock")), path.open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    except OSError:
        # The mailbox is authoritative. A visualization log failure must not
        # change message delivery semantics.
        return


def list_team_communications(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent Team message routes for dashboard animation."""

    limit = max(1, min(int(limit), 500))
    path = _team_communications_path()
    if not path.exists():
        return []
    items: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with interprocess_lock(path.with_suffix(".lock")):
            lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            items.append(value)
    return list(items)


def _validate_agent_name(name: str) -> str:
    if not isinstance(name, str) or not _AGENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid agent name: {name}")
    return name


def _canonical_agent_name(name: str) -> str:
    value = _validate_agent_name(name)
    if value.casefold() in _LEAD_AGENT_ALIASES:
        return _LEAD_AGENT_NAME
    return value


def _validate_teammate_name(name: str) -> str:
    value = _validate_agent_name(name)
    if value.casefold() in _RESERVED_TEAMMATE_NAMES:
        raise ValueError(f"reserved teammate name: {name}")
    return value


@dataclass(frozen=True)
class InboxBatch:
    batch_id: str
    agent: str
    messages: tuple[dict, ...]
    path: Path | None = field(default=None, repr=False, compare=False)


class MessageBus:
    def __init__(self):
        self._claimed: dict[str, Path] = {}

    @staticmethod
    def _lock_path(agent: str) -> Path:
        agent = _canonical_agent_name(agent)
        return config.MAILBOX_DIR / f".{agent}.lock"

    def _recover_lead_alias_mailboxes_locked(self) -> None:
        """Move mail addressed to Lead aliases into the canonical mailbox."""

        canonical = config.MAILBOX_DIR / f"{_LEAD_AGENT_NAME}.jsonl"
        recovered: list[str] = []
        claimed = set(self._claimed.values())
        for path in sorted(config.MAILBOX_DIR.iterdir()):
            if not path.is_file() or path in claimed:
                continue
            alias: str | None = None
            if path.suffix == ".jsonl" and not path.name.startswith("."):
                alias = path.stem
            else:
                match = re.fullmatch(
                    r"\.([A-Za-z0-9_-]+)\.batch_[A-Za-z0-9]+\.inflight\.jsonl",
                    path.name,
                )
                if match:
                    alias = match.group(1)
            if alias is None or alias.casefold() not in _LEAD_AGENT_ALIASES:
                continue
            if canonical.exists():
                try:
                    if path.samefile(canonical):
                        continue
                except OSError:
                    pass
            try:
                recovered.append(path.read_text(encoding="utf-8"))
                path.unlink()
            except FileNotFoundError:
                continue
        if recovered:
            existing = canonical.read_text(encoding="utf-8") if canonical.exists() else ""
            atomic_write_text(canonical, existing + "".join(recovered))

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: dict | None = None,
    ) -> str:
        from_agent = _canonical_agent_name(from_agent)
        to_agent = _canonical_agent_name(to_agent)
        msg = {
            "id": f"msg_{uuid.uuid4().hex}",
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "ts": time.time(),
            "metadata": metadata or {},
        }
        config.MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
        inbox = config.MAILBOX_DIR / f"{to_agent}.jsonl"
        with _mailbox_lock, interprocess_lock(self._lock_path(to_agent)), inbox.open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(msg, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _record_team_communication(msg)
        notify(
            "team_message",
            {
                "message_id": msg["id"],
                "from_agent": from_agent,
                "to_agent": to_agent,
                "message_type": msg_type,
                "task_id": msg["metadata"].get("task_id"),
                "sent_at": msg["ts"],
                "status": "sent",
            },
        )
        if to_agent == _LEAD_AGENT_NAME and msg_type in _LEAD_WAKE_MESSAGE_TYPES:
            _lead_inbox_event.set()
            notify(
                "team_inbox_unread",
                {
                    "message_id": msg["id"],
                    "from_agent": from_agent,
                    "message_type": msg_type,
                    "task_id": msg["metadata"].get("task_id"),
                    "status": "unread",
                },
            )
        return msg["id"]

    def claim_inbox(self, agent: str) -> InboxBatch:
        agent = _canonical_agent_name(agent)
        config.MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
        with _mailbox_lock, interprocess_lock(self._lock_path(agent)):
            if agent == _LEAD_AGENT_NAME:
                self._recover_lead_alias_mailboxes_locked()
            claimed_paths = set(self._claimed.values())
            recoverable = [
                path
                for path in sorted(config.MAILBOX_DIR.glob(f".{agent}.*.inflight.jsonl"))
                if path not in claimed_paths
            ]
            if recoverable:
                processing = recoverable[0]
                batch_id = processing.name.split(".")[2]
            else:
                inbox = config.MAILBOX_DIR / f"{agent}.jsonl"
                if not inbox.exists():
                    return InboxBatch("", agent, ())
                batch_id = f"batch_{uuid.uuid4().hex}"
                processing = config.MAILBOX_DIR / f".{agent}.{batch_id}.inflight.jsonl"
                os.replace(inbox, processing)

            messages: list[dict] = []
            malformed: list[str] = []
            for line in processing.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("mailbox entry is not an object")
                    for field_name in ("from", "to"):
                        field_value = value.get(field_name)
                        if (
                            isinstance(field_value, str)
                            and field_value.casefold() in _LEAD_AGENT_ALIASES
                        ):
                            value[field_name] = _LEAD_AGENT_NAME
                    messages.append(value)
                except (json.JSONDecodeError, ValueError):
                    malformed.append(line)
            if malformed:
                dead_letter = config.MAILBOX_DIR / f"{agent}.dead-letter.jsonl"
                with dead_letter.open("a", encoding="utf-8") as handle:
                    for line in malformed:
                        handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                atomic_write_text(
                    processing,
                    "".join(
                        json.dumps(message, ensure_ascii=False) + "\n"
                        for message in messages
                    ),
                )
            self._claimed[batch_id] = processing
            return InboxBatch(batch_id, agent, tuple(messages), processing)

    def ack_inbox(self, batch: InboxBatch) -> None:
        if not batch.batch_id or batch.path is None:
            return
        with _mailbox_lock, interprocess_lock(self._lock_path(batch.agent)):
            path = self._claimed.pop(batch.batch_id, batch.path)
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def nack_inbox(self, batch: InboxBatch) -> None:
        if not batch.batch_id or batch.path is None:
            return
        with _mailbox_lock, interprocess_lock(self._lock_path(batch.agent)):
            path = self._claimed.pop(batch.batch_id, batch.path)
            if not path.exists():
                return
            inbox = config.MAILBOX_DIR / f"{batch.agent}.jsonl"
            older = path.read_text(encoding="utf-8")
            newer = inbox.read_text(encoding="utf-8") if inbox.exists() else ""
            atomic_write_text(inbox, older + newer)
            path.unlink()

    @contextlib.contextmanager
    def consume(self, agent: str):
        batch = self.claim_inbox(agent)
        try:
            yield list(batch.messages)
        except BaseException:
            self.nack_inbox(batch)
            raise
        else:
            self.ack_inbox(batch)

    def read_inbox(self, agent: str) -> list[dict]:
        with self.consume(agent) as messages:
            return messages


BUS = MessageBus()
active_teammates: dict[str, bool] = {}
_teammate_lock = threading.RLock()
_teammate_threads: dict[str, threading.Thread] = {}
_teammate_stop_events: dict[str, threading.Event] = {}
_teammate_states: dict[str, dict[str, Any]] = {}
_teammate_lifecycle_lock = threading.RLock()
_teammate_stop_event = threading.Event()
_team_accepting = False
_teammate_profile_restart_pending: set[str] = set()


TEAM_CORE_TOOLS = (
    "send_message",
    "list_tasks",
    "get_task",
    "claim_task",
    "complete_task",
)
TEAM_OPTIONAL_TOOLS = (
    "read_file",
    "glob",
    "todo_write",
    "submit_plan",
    "load_skill",
    "web_search",
    "write_file",
    "edit_file",
    "bash",
)
TEAM_DEFAULT_ALLOWED_TOOLS = (
    *TEAM_CORE_TOOLS,
    "read_file",
    "submit_plan",
    "write_file",
    "bash",
)
_TEAM_TOOL_ORDER = (*TEAM_CORE_TOOLS, *TEAM_OPTIONAL_TOOLS)
_TEAM_TOOL_LABELS = {
    "send_message": ("发送消息", "协作"),
    "list_tasks": ("查看任务列表", "任务系统"),
    "get_task": ("查看任务详情", "任务系统"),
    "claim_task": ("领取任务", "任务系统"),
    "complete_task": ("完成任务", "任务系统"),
    "read_file": ("读取文件", "工作区"),
    "glob": ("查找文件", "工作区"),
    "todo_write": ("维护执行计划", "规划"),
    "submit_plan": ("提交方案审批", "规划"),
    "load_skill": ("加载技能", "能力"),
    "web_search": ("搜索互联网", "能力"),
    "write_file": ("写入文件", "工作区"),
    "edit_file": ("编辑文件", "工作区"),
    "bash": ("执行命令", "工作区"),
}


def _normalize_teammate_tools(value: Any = None) -> list[str]:
    if value is None:
        requested = set(TEAM_DEFAULT_ALLOWED_TOOLS)
    else:
        if not isinstance(value, list):
            raise ValueError("allowed_tools must be an array")
        if any(not isinstance(item, str) for item in value):
            raise ValueError("allowed_tools must contain tool names")
        requested = {item.strip() for item in value if item.strip()}
        unknown = requested.difference(_TEAM_TOOL_ORDER)
        if unknown:
            raise ValueError(
                "unknown Team Agent tools: " + ", ".join(sorted(unknown))
            )
    requested.update(TEAM_CORE_TOOLS)
    return [name for name in _TEAM_TOOL_ORDER if name in requested]


def teammate_tool_catalog() -> list[dict[str, Any]]:
    defaults = set(TEAM_DEFAULT_ALLOWED_TOOLS)
    core = set(TEAM_CORE_TOOLS)
    return [
        {
            "name": name,
            "label": _TEAM_TOOL_LABELS[name][0],
            "group": _TEAM_TOOL_LABELS[name][1],
            "required": name in core,
            "default_enabled": name in defaults,
        }
        for name in _TEAM_TOOL_ORDER
    ]


_TEAM_TOOL_DEFINITIONS = {
    "send_message": {
        "name": "send_message",
        "description": (
            "Send a message to another agent. To report to the workspace Lead, "
            "set to='lead'; Lead, Leader, and main are normalized to that recipient."
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
    "list_tasks": {
        "name": "list_tasks",
        "description": "List all tasks.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "get_task": {
        "name": "get_task",
        "description": "Read the full task, including description and owner.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    "claim_task": {
        "name": "claim_task",
        "description": "Claim a pending task.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    "complete_task": {
        "name": "complete_task",
        "description": "Mark an in-progress task as completed.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    "read_file": {
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
    "glob": {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    "todo_write": {
        "name": "todo_write",
        "description": "Create and manage a task list for this Agent.",
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
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
    "submit_plan": {
        "name": "submit_plan",
        "description": "Submit a plan for Lead approval.",
        "input_schema": {
            "type": "object",
            "properties": {"plan": {"type": "string"}},
            "required": ["plan"],
        },
    },
    "load_skill": {
        "name": "load_skill",
        "description": "Load the full content of a skill by name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the current public web with Tavily.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 400},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                "topic": {"type": "string", "enum": ["general", "news", "finance"]},
                "search_depth": {"type": "string", "enum": ["basic", "advanced"]},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "write_file": {
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
    "edit_file": {
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
    "bash": {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "write_paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["command"],
        },
    },
}


def _team_settings_path() -> Path:
    return config.WORKDIR / ".gugugaga" / "team-settings.json"


def _team_profiles_path() -> Path:
    return config.WORKDIR / ".gugugaga" / "team-agents.json"


def _load_teammate_profiles() -> dict[str, dict[str, Any]]:
    path = _team_profiles_path()
    with interprocess_lock(path.with_suffix(".lock")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            payload = {}
    raw_agents = payload.get("agents", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_agents, dict):
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for name, raw in raw_agents.items():
        if not isinstance(raw, dict):
            continue
        try:
            _validate_teammate_name(name)
        except ValueError:
            continue
        role = str(raw.get("role", "")).strip()
        prompt = str(raw.get("prompt", "")).strip()
        if not role or not prompt:
            continue
        try:
            allowed_tools = _normalize_teammate_tools(raw.get("allowed_tools"))
        except ValueError:
            allowed_tools = list(TEAM_DEFAULT_ALLOWED_TOOLS)
        profiles[name] = {
            "name": name,
            "role": role,
            "initial_role": str(raw.get("initial_role") or role).strip(),
            "prompt": prompt,
            "initial_prompt": str(raw.get("initial_prompt") or prompt).strip(),
            "allowed_tools": allowed_tools,
            "created_at": raw.get("created_at"),
            "updated_at": raw.get("updated_at"),
        }
    return profiles


def _persist_teammate_profile(
    name: str,
    role: str,
    prompt: str,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    path = _team_profiles_path()
    with interprocess_lock(path.with_suffix(".lock")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        agents = payload.get("agents")
        if not isinstance(agents, dict):
            agents = {}
        now = time.time()
        previous = agents.get(name, {})
        previous = previous if isinstance(previous, dict) else {}
        normalized_tools = _normalize_teammate_tools(
            allowed_tools
            if allowed_tools is not None
            else previous.get("allowed_tools")
        )
        agents[name] = {
            "name": name,
            "role": role,
            "initial_role": str(previous.get("initial_role") or role),
            "prompt": prompt,
            "initial_prompt": str(previous.get("initial_prompt") or prompt),
            "allowed_tools": normalized_tools,
            "created_at": previous.get("created_at", now),
            "updated_at": now,
        }
        atomic_write_text(
            path,
            json.dumps(
                {"version": 2, "agents": agents},
                ensure_ascii=False,
                indent=2,
            ),
        )
        return dict(agents[name])


def _delete_teammate_profile(name: str) -> bool:
    path = _team_profiles_path()
    with interprocess_lock(path.with_suffix(".lock")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        agents = payload.get("agents")
        if not isinstance(agents, dict) or name not in agents:
            return False
        agents.pop(name, None)
        atomic_write_text(
            path,
            json.dumps(
                {"version": 2, "agents": agents},
                ensure_ascii=False,
                indent=2,
            ),
        )
        return True


def teammate_profile(name: str) -> dict[str, Any]:
    _validate_teammate_name(name)
    profile = _load_teammate_profiles().get(name)
    if profile is None:
        raise KeyError(f"teammate '{name}' was not found")
    return profile


def update_teammate_profile(
    name: str,
    *,
    role: str | None = None,
    prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    """Persist a Team Agent configuration and schedule a safe runtime reload."""

    with _teammate_lifecycle_lock:
        current = teammate_profile(name)
        if reset:
            next_role = current["initial_role"]
            next_prompt = current["initial_prompt"]
            next_tools = list(TEAM_DEFAULT_ALLOWED_TOOLS)
        else:
            next_role = current["role"] if role is None else str(role).strip()
            next_prompt = current["prompt"] if prompt is None else str(prompt).strip()
            next_tools = (
                current["allowed_tools"]
                if allowed_tools is None
                else allowed_tools
            )
        if not next_role:
            raise ValueError("teammate role is required")
        if len(next_role) > 200:
            raise ValueError("teammate role is too long")
        if not next_prompt:
            raise ValueError("teammate prompt is required")
        if len(next_prompt) > 20_000:
            raise ValueError("teammate prompt is too long")
        normalized_tools = _normalize_teammate_tools(next_tools)
        saved = _persist_teammate_profile(
            name,
            next_role,
            next_prompt,
            normalized_tools,
        )
        with _teammate_lock:
            active = name in active_teammates
            state = _teammate_states.get(name) or {}
            status = str(state.get("status") or "stopped")
            has_active_task = bool(state.get("current_task_id"))
            if active:
                _teammate_profile_restart_pending.add(name)
                apply_state = "pending"
                if status == "idle" and not has_active_task:
                    stop_event = _teammate_stop_events.get(name)
                    if stop_event is not None:
                        state["status"] = "restarting"
                        stop_event.set()
                        apply_state = "restarting"
            else:
                apply_state = "next_start"
        return {
            **saved,
            "apply_state": apply_state,
            "restart_required": active,
        }


def get_team_settings() -> dict[str, Any]:
    path = _team_settings_path()
    with interprocess_lock(path.with_suffix(".lock")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            value = {}
    return {
        "auto_claim_enabled": bool(value.get("auto_claim_enabled", False)),
        "updated_at": value.get("updated_at"),
    }


def update_team_settings(auto_claim_enabled: bool) -> dict[str, Any]:
    if not isinstance(auto_claim_enabled, bool):
        raise ValueError("auto_claim_enabled must be boolean")
    value = {
        "auto_claim_enabled": auto_claim_enabled,
        "updated_at": time.time(),
    }
    path = _team_settings_path()
    with interprocess_lock(path.with_suffix(".lock")):
        atomic_write_text(
            path, json.dumps(value, ensure_ascii=False, indent=2)
        )
    return value


def _set_teammate_state(name: str, **updates: Any) -> None:
    with _teammate_lock:
        state = _teammate_states.get(name)
        if state is None:
            return
        state.update(updates)
        state["last_active_at"] = time.time()


def list_teammate_states() -> list[dict[str, Any]]:
    tasks = list_tasks()
    active_by_owner = {
        task.owner: task.id
        for task in tasks
        if task.status == "in_progress" and task.owner
    }
    reserved_by_assignee: dict[str, str] = {}
    for task in sorted(
        tasks,
        key=lambda value: (
            value.queue_position is None,
            value.queue_position or 0,
            value.id,
        ),
    ):
        if task.status == "pending" and task.assignee:
            reserved_by_assignee.setdefault(task.assignee, task.id)
    profiles = _load_teammate_profiles()
    with _teammate_lock:
        values = []
        names = set(profiles) | set(_teammate_states)
        for name in names:
            state = _teammate_states.get(name) or {
                "name": name,
                "role": profiles.get(name, {}).get("role", "teammate"),
                "status": "stopped",
                "online": False,
                "current_task_id": None,
                "started_at": None,
                "last_active_at": profiles.get(name, {}).get("updated_at"),
            }
            current_task_id = active_by_owner.get(name) or reserved_by_assignee.get(name)
            status = str(state.get("status", "idle"))
            online = name in active_teammates
            if online and active_by_owner.get(name):
                status = "running"
            values.append(
                {
                    **state,
                    "name": name,
                    "role": state.get("role") or profiles.get(name, {}).get("role"),
                    "online": online,
                    "status": status,
                    "current_task_id": current_task_id,
                }
            )
    return sorted(values, key=lambda item: item["name"].casefold())


def assign_task_to_teammate(task_id: str, teammate: str) -> dict[str, Any]:
    _validate_teammate_name(teammate)
    with _teammate_lock:
        state = _teammate_states.get(teammate)
        if teammate not in active_teammates or state is None:
            raise ValueError(f"teammate '{teammate}' is offline")
        if state.get("status") != "idle":
            raise ValueError(f"teammate '{teammate}' is not idle")
        task = assign_task(task_id, teammate)
        try:
            BUS.send(
                "lead",
                teammate,
                f"Task {task.id} has been assigned to you.",
                "task_assignment",
                {"task_id": task.id},
            )
        except Exception:
            from .tasks import unassign_task

            unassign_task(task.id)
            raise
        _set_teammate_state(teammate, current_task_id=task.id)
    return asdict(task)


def _submit_team_interaction_unlocked(
    name: str,
    action: str,
    content: str = "",
) -> dict[str, Any]:
    """Apply one explicit user interaction to a selected Team Agent."""

    _validate_teammate_name(name)
    action = str(action or "").strip().lower()
    profiles = _load_teammate_profiles()
    with _teammate_lock:
        state = dict(_teammate_states.get(name) or {})
        active = name in active_teammates
    if name not in profiles and not state:
        raise ValueError(f"teammate '{name}' was not found")
    broker = interaction_broker()
    target = f"team:{name}"
    current_task_id = state.get("current_task_id")
    if action in {"steer", "redirect", "stop"} and not active:
        raise ValueError(f"teammate '{name}' is not active")
    if action == "queue":
        item = broker.submit(target, action, content)
        title = next(
            (line.strip() for line in content.splitlines() if line.strip()),
            "Queued Team Agent task",
        )[:80]
        try:
            task = create_queued_task(
                title,
                content,
                name,
                interaction_id=item.id,
            )
            broker.update(item.id, "task_created", task_id=task.id)
            if active and state.get("status") == "idle":
                BUS.send(
                    "lead",
                    name,
                    f"Queued task {task.id} is ready.",
                    "task_assignment",
                    {"task_id": task.id, "interaction_id": item.id},
                )
            return {**asdict(item), "status": "task_created", "task_id": task.id}
        except Exception as error:
            broker.update(item.id, "failed", metadata={"error": str(error)})
            raise
    item = broker.submit(
        target,
        action,
        content,
        task_id=current_task_id,
    )
    if current_task_id and action in {"steer", "redirect"}:
        try:
            append_task_intervention(
                current_task_id,
                interaction_id=item.id,
                action=action,
                content=content,
                status="pending",
            )
        except (FileNotFoundError, ValueError):
            pass
    if action == "stop":
        result = stop_teammate(name)
        if result.startswith("Error:"):
            broker.update(item.id, "failed", metadata={"error": result})
            raise ValueError(result.removeprefix("Error:").strip())
    return asdict(item)


def submit_team_interaction(
    name: str,
    action: str,
    content: str = "",
) -> dict[str, Any]:
    with _teammate_lifecycle_lock:
        return _submit_team_interaction_unlocked(name, action, content)


@dataclass(frozen=True)
class TeammateShutdownOutcome:
    stopped: bool
    live_names: tuple[str, ...]


@dataclass
class ProtocolState:
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: str
    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {}
_protocol_lock = threading.RLock()
_protocol_workspace: Path | None = None
_PROTOCOL_RETENTION_SECONDS = 24 * 60 * 60


def _protocol_path() -> Path:
    return config.MAILBOX_DIR / "protocol-requests.json"


def _prune_protocol_requests_locked(now: float | None = None) -> None:
    cutoff = (now or time.time()) - _PROTOCOL_RETENTION_SECONDS
    for request_id, state in list(pending_requests.items()):
        if state.status != "pending" and state.created_at < cutoff:
            pending_requests.pop(request_id, None)


def _ensure_protocol_state_loaded_locked() -> None:
    global _protocol_workspace
    workspace = config.MAILBOX_DIR.resolve()
    if _protocol_workspace == workspace:
        return
    pending_requests.clear()
    path = _protocol_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw.get("requests", []):
                state = ProtocolState(**item)
                pending_requests[state.request_id] = state
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            corrupt = path.with_name(
                f"{path.stem}.corrupt-{int(time.time())}{path.suffix}"
            )
            try:
                os.replace(path, corrupt)
            except OSError:
                pass
    _protocol_workspace = workspace
    _prune_protocol_requests_locked()


def _persist_protocol_requests_locked() -> None:
    config.MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
    _prune_protocol_requests_locked()
    content = json.dumps(
        {
            "version": 1,
            "requests": [
                asdict(state)
                for state in sorted(
                    pending_requests.values(), key=lambda value: value.created_at
                )
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    with interprocess_lock(config.MAILBOX_DIR / ".protocol.lock"):
        atomic_write_text(_protocol_path(), content)


def _ensure_protocol_state_loaded() -> None:
    with _protocol_lock:
        _ensure_protocol_state_loaded_locked()


def _new_request_id_locked() -> str:
    _ensure_protocol_state_loaded_locked()
    while True:
        request_id = f"req_{random.randint(0, 999999):06d}"
        if request_id not in pending_requests:
            return request_id


def new_request_id() -> str:
    with _protocol_lock:
        return _new_request_id_locked()


def _create_protocol_request(
    request_type: str,
    sender: str,
    target: str,
    payload: str,
) -> ProtocolState:
    # ID selection and reservation are one critical section. Otherwise two
    # concurrent creators can both observe the same random ID as available.
    with _protocol_lock:
        request_id = _new_request_id_locked()
        state = ProtocolState(
            request_id=request_id,
            type=request_type,
            sender=sender,
            target=target,
            status="pending",
            payload=payload,
        )
        pending_requests[request_id] = state
        _persist_protocol_requests_locked()
        return state


def match_response(
    response_type: str,
    request_id: str,
    approve: bool,
    *,
    sender: str = "",
    target: str = "",
) -> bool:
    with _protocol_lock:
        _ensure_protocol_state_loaded_locked()
        state = pending_requests.get(request_id)
        if not state or state.status != "pending":
            return False
        expected_type = {
            "shutdown": "shutdown_response",
            "plan_approval": "plan_approval_response",
        }.get(state.type)
        if response_type != expected_type:
            return False
        if sender != state.target or target != state.sender:
            return False
        state.status = "approved" if approve else "rejected"
        _persist_protocol_requests_locked()
        return True


def _route_protocol_messages(msgs: list[dict]) -> None:
    for msg in msgs:
        meta = msg.get("metadata", {})
        request_id = meta.get("request_id", "")
        msg_type = msg.get("type", "")
        if request_id and msg_type.endswith("_response"):
            match_response(
                msg_type,
                request_id,
                meta.get("approve", False),
                sender=msg.get("from", ""),
                target=msg.get("to", ""),
            )


def _lead_inbox_has_pending() -> bool:
    if (config.MAILBOX_DIR / "lead.jsonl").exists():
        return True
    claimed = set(getattr(BUS, "_claimed", {}).values())
    if not config.MAILBOX_DIR.exists():
        return False
    for path in config.MAILBOX_DIR.iterdir():
        if path in claimed or not path.is_file():
            continue
        if path.name.startswith("."):
            match = re.fullmatch(
                r"\.([A-Za-z0-9_-]+)\.batch_[A-Za-z0-9]+\.inflight\.jsonl",
                path.name,
            )
            if match and match.group(1).casefold() in _LEAD_AGENT_ALIASES:
                return True
        elif path.suffix == ".jsonl" and path.stem.casefold() in _LEAD_AGENT_ALIASES:
            return True
    return False


def signal_pending_lead_inbox() -> bool:
    pending = _lead_inbox_has_pending()
    if pending:
        _lead_inbox_event.set()
    return pending


def wait_for_lead_inbox(
    stop_event: threading.Event | None = None,
    timeout: float = 1.0,
) -> bool:
    stop_event = stop_event or threading.Event()
    if stop_event.is_set():
        return False
    signaled = _lead_inbox_event.wait(max(0.0, float(timeout)))
    return bool(signaled and not stop_event.is_set())


def claim_lead_inbox(route_protocol: bool = True) -> InboxBatch:
    # Clear before claiming so a message sent concurrently after the atomic
    # rename leaves the event set for the next delivery.
    _lead_inbox_event.clear()
    batch = BUS.claim_inbox("lead")
    try:
        if route_protocol:
            _route_protocol_messages(list(batch.messages))
    except BaseException:
        BUS.nack_inbox(batch)
        _lead_inbox_event.set()
        raise
    if batch.messages:
        notify(
            "team_inbox_claimed",
            {
                "batch_id": batch.batch_id,
                "message_count": len(batch.messages),
                "message_ids": [
                    message.get("id") for message in batch.messages
                ],
                "status": "processing",
            },
        )
    if _lead_inbox_has_pending():
        _lead_inbox_event.set()
    return batch


def ack_lead_inbox(batch: InboxBatch) -> None:
    BUS.ack_inbox(batch)
    if batch.messages:
        notify(
            "team_inbox_acknowledged",
            {
                "batch_id": batch.batch_id,
                "message_count": len(batch.messages),
                "status": "processed",
            },
        )
    signal_pending_lead_inbox()


def nack_lead_inbox(batch: InboxBatch, error: str = "") -> None:
    BUS.nack_inbox(batch)
    if batch.messages:
        notify(
            "team_inbox_requeued",
            {
                "batch_id": batch.batch_id,
                "message_count": len(batch.messages),
                "status": "unread",
                "error": error,
            },
        )
        _lead_inbox_event.set()


def render_lead_inbox(batch: InboxBatch) -> str:
    payload = [
        {
            "id": message.get("id"),
            "from": message.get("from"),
            "type": message.get("type", "message"),
            "content": message.get("content", ""),
            "metadata": message.get("metadata", {}),
            "timestamp": message.get("ts"),
        }
        for message in batch.messages
    ]
    return (
        "<team-inbox>\n"
        "You are the Lead Agent (protocol id: lead), and these collaboration "
        "messages were delivered directly to you. Lead, Leader, and main all "
        "refer to you, not to a teammate. Do not forward these messages or use "
        "send_message to look for a teammate named Leader. Review result "
        "and error messages, respond to plan approval requests, and update "
        "task coordination when needed. Message ids support at-least-once "
        "deduplication.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</team-inbox>"
    )


def consume_lead_inbox(route_protocol: bool = True) -> list[dict]:
    batch = claim_lead_inbox(route_protocol)
    try:
        messages = list(batch.messages)
    except BaseException as error:
        nack_lead_inbox(batch, str(error))
        raise
    ack_lead_inbox(batch)
    return messages


IDLE_POLL_INTERVAL = 5
# Production Team Agents stay online until an explicit shutdown or service
# close. Tests may inject a finite timeout to exercise terminal idle paths.
IDLE_TIMEOUT: float | None = None
PLAN_APPROVAL_TIMEOUT = 60


def scan_unclaimed_tasks(agent_name: str | None = None) -> list[dict]:
    values = [
        asdict(task)
        for task in list_tasks()
        if task.status == "pending"
        and not task.owner
        and (not task.assignee or task.assignee == agent_name)
        and can_start(task.id)
    ]
    return sorted(
        values,
        key=lambda item: (
            item.get("assignee") != agent_name,
            item.get("queue_position") is None,
            item.get("queue_position") or 0,
            item["id"],
        ),
    )


def idle_poll(
    agent_name: str,
    messages: list,
    name: str,
    role: str,
    stop_event: threading.Event | None = None,
    work_state: dict[str, Any] | None = None,
    require_task: bool = False,
) -> str:
    del role
    stop_event = stop_event or threading.Event()
    deadline = (
        None
        if IDLE_TIMEOUT is None
        else time.monotonic() + max(0.0, float(IDLE_TIMEOUT))
    )
    _set_teammate_state(agent_name, status="idle")
    while deadline is None or time.monotonic() < deadline:
        if stop_event.is_set():
            return "shutdown"
        with BUS.consume(agent_name) as inbox:
            if inbox:
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        request_id = msg.get("metadata", {}).get(
                            "request_id", ""
                        )
                        BUS.send(
                            name,
                            "lead",
                            "Shutting down.",
                            "shutdown_response",
                            {"request_id": request_id, "approve": True},
                        )
                        return "shutdown"
                    if msg.get("type") == "task_assignment":
                        task_id = str(msg.get("metadata", {}).get("task_id", ""))
                        try:
                            assigned = load_task(task_id)
                        except (FileNotFoundError, ValueError):
                            assigned = None
                        result = (
                            claim_task(task_id, agent_name)
                            if assigned is not None
                            and assigned.assignee == agent_name
                            else "assignment was cancelled or replaced"
                        )
                        if not result.startswith("Claimed"):
                            BUS.send(
                                agent_name,
                                "lead",
                                f"Could not accept assigned task {task_id}: {result}",
                                "error",
                                {"task_id": task_id},
                            )
                            continue
                        claimed = load_task(task_id)
                        if work_state is not None:
                            work_state["task_id"] = task_id
                        _set_teammate_state(
                            agent_name,
                            status="running",
                            current_task_id=task_id,
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "<assigned-task>"
                                    + json.dumps(asdict(claimed), ensure_ascii=False)
                                    + "</assigned-task>"
                                ),
                            }
                        )
                        return "work"
                messages.append(
                    {
                        "role": "user",
                        "content": "<inbox>" + json.dumps(inbox) + "</inbox>",
                    }
                )
                ordinary_message = any(
                    msg.get("type", "message") == "message" for msg in inbox
                )
                if ordinary_message or not require_task or (
                    work_state is not None and work_state.get("task_id")
                ):
                    return "work"
        # Explicitly queued work is user-owned dispatch and must remain usable
        # even while autonomous claiming is disabled.
        assigned = [
            item
            for item in scan_unclaimed_tasks(agent_name)
            if item.get("assignee") == agent_name
        ]
        if assigned:
            task_data = assigned[0]
            result = claim_task(task_data["id"], agent_name)
            if result.startswith("Claimed"):
                claimed = load_task(task_data["id"])
                if work_state is not None:
                    work_state["task_id"] = claimed.id
                _set_teammate_state(
                    agent_name,
                    status="running",
                    current_task_id=claimed.id,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "<queued-task>"
                            + json.dumps(asdict(claimed), ensure_ascii=False)
                            + "</queued-task>"
                        ),
                    }
                )
                return "work"
        if not get_team_settings()["auto_claim_enabled"]:
            wait_seconds = float(IDLE_POLL_INTERVAL)
            if deadline is not None:
                wait_seconds = min(
                    wait_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            if stop_event.wait(wait_seconds):
                return "shutdown"
            continue
        unclaimed = scan_unclaimed_tasks(agent_name)
        if unclaimed:
            task_data = unclaimed[0]
            result = claim_task(task_data["id"], agent_name)
            if "Claimed" in result:
                claimed = load_task(task_data["id"])
                if work_state is not None:
                    work_state["task_id"] = claimed.id
                _set_teammate_state(
                    agent_name,
                    status="running",
                    current_task_id=claimed.id,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "<auto-claimed>"
                            + json.dumps(asdict(claimed), ensure_ascii=False)
                            + "</auto-claimed>"
                        ),
                    }
                )
                return "work"
        wait_seconds = float(IDLE_POLL_INTERVAL)
        if deadline is not None:
            wait_seconds = min(
                wait_seconds,
                max(0.0, deadline - time.monotonic()),
            )
        if stop_event.wait(wait_seconds):
            return "shutdown"
    return "timeout"


_team_provider: Any | None = None
_team_permissions = PermissionPolicy()
_team_approval_callback: Callable[[ToolCall], bool] | None = None
_team_context_parent_resolver: Callable[[], SessionContextCoordinator] | None = None
_team_max_tokens = config.DEFAULT_MAX_TOKENS
_team_max_rounds_per_burst = 10


def set_team_provider(
    provider: Any | None,
    permissions: PermissionPolicy | None = None,
    approval_callback: Callable[[ToolCall], bool] | None = None,
    *,
    context_parent_resolver: Callable[[], SessionContextCoordinator] | None = None,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    max_rounds_per_burst: int = 10,
) -> None:
    global _team_provider, _team_permissions, _team_approval_callback
    global _team_accepting, _team_context_parent_resolver
    global _team_max_tokens, _team_max_rounds_per_burst
    _team_provider = provider
    _team_permissions = permissions or PermissionPolicy()
    _team_approval_callback = approval_callback
    _team_context_parent_resolver = context_parent_resolver
    _team_max_tokens = max(1, int(max_tokens))
    _team_max_rounds_per_burst = max(1, int(max_rounds_per_burst))
    _ensure_protocol_state_loaded()
    with _teammate_lock:
        _team_accepting = provider is not None
        if provider is not None and not _teammate_threads:
            _teammate_stop_event.clear()


def _dispatch_teammate_tool(
    block, handlers: dict[str, Callable]
) -> tuple[str, str]:
    from .tools import call_tool_handler

    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked), "blocked"
    call = ToolCall(block.id, block.name, block.input)
    if not _team_permissions.approve(call, _team_approval_callback):
        return (
            f"Permission denied for tool '{block.name}'. "
            "Choose a safer approach."
        ), "denied"
    output = call_tool_handler(
        handlers.get(block.name), block.input, block.name
    )
    trigger_hooks("PostToolUse", block, output)
    rendered = str(output)
    status = (
        "error"
        if rendered.startswith(("Error:", "Unknown:", "Conflict:"))
        else "ok"
    )
    return rendered, status


def _spawn_teammate_thread_unlocked(
    name: str,
    role: str,
    prompt: str,
    *,
    allowed_tools: list[str] | None = None,
    await_assignment: bool = False,
    persist_profile: bool = True,
) -> str:
    try:
        _validate_teammate_name(name)
    except ValueError as error:
        return f"Error: {error}"
    if _team_provider is None:
        return "Error: teammate provider is not configured"
    role = str(role or "").strip()
    prompt = str(prompt or "").strip()
    if not role:
        return "Error: teammate role is required"
    if len(role) > 200:
        return "Error: teammate role is too long"
    if not prompt:
        return "Error: teammate prompt is required"
    if len(prompt) > 20_000:
        return "Error: teammate prompt is too long"
    try:
        normalized_tools = _normalize_teammate_tools(allowed_tools)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        if not _team_accepting:
            return "Error: teammate manager is not accepting new teammates"
        if name in active_teammates:
            return f"Teammate '{name}' already exists"
        if persist_profile:
            profile = _persist_teammate_profile(
                name, role, prompt, normalized_tools
            )
        else:
            profile = _load_teammate_profiles().get(name, {})
        if not _teammate_threads:
            _teammate_stop_event.clear()
        stop_event = threading.Event()
        active_teammates[name] = True
        _teammate_stop_events[name] = stop_event
        now = time.time()
        _teammate_states[name] = {
            "name": name,
            "role": role,
            "status": "running",
            "online": True,
            "current_task_id": None,
            "started_at": now,
            "last_active_at": now,
            "configuration_updated_at": profile.get("updated_at"),
            "active_allowed_tools": normalized_tools,
        }

    protocol_ctx = {"waiting_plan": None, "waiting_since": None}
    system = (
        f"You are '{name}', a {role}. Use tools to complete tasks in the "
        f"shared selected workspace at {config.WORKDIR}. Before overwriting an "
        "existing file, read it with include_hash=true and pass the returned "
        "SHA-256 as expected_sha256. Declare every file a Bash command may "
        "modify in write_paths. The Task System is the authority for work "
        "dispatch. When workspace auto-claim is disabled, you may claim and "
        "execute only a task explicitly assigned to you. Do not start workspace "
        "mutations from your role description or an ordinary inbox message. "
        "An ordinary inbox message may wake you for a conversational response; "
        "answer it without workspace mutations unless you also hold an active task."
    )

    def handle_inbox_message(agent_name: str, msg: dict, messages: list):
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        request_id = meta.get("request_id", "")
        if msg_type == "shutdown_request":
            BUS.send(
                agent_name,
                "lead",
                "Shutting down.",
                "shutdown_response",
                {"request_id": request_id, "approve": True},
            )
            return True
        if msg_type == "plan_approval_response":
            approve = meta.get("approve", False)
            with _protocol_lock:
                _ensure_protocol_state_loaded_locked()
                state = pending_requests.get(request_id)
                valid = (
                    request_id == protocol_ctx["waiting_plan"]
                    and state is not None
                    and state.type == "plan_approval"
                    and state.sender == agent_name
                    and state.target == "lead"
                    and state.status
                    == ("approved" if approve else "rejected")
                    and msg.get("from") == "lead"
                    and msg.get("to") == agent_name
                )
            if not valid:
                notify(
                    "teammate_protocol_error",
                    {
                        "name": agent_name,
                        "request_id": request_id,
                        "message_type": msg_type,
                    },
                )
                return False
            protocol_ctx["waiting_plan"] = None
            protocol_ctx["waiting_since"] = None
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[Plan approved]"
                        if approve
                        else f"[Plan rejected] {msg['content']}"
                    ),
                }
            )
        return False

    def run():
        from .context_modes import (
            CompressionReason,
            ContextModeError,
            RequestContext,
            create_child_context_coordinator,
        )

        scope = event_scope(agent_type="teammate", agent_id=name)
        scope.__enter__()
        notify(
            "teammate_start",
            {"name": name, "role": role, "prompt_chars": len(prompt)},
        )
        messages = [{"role": "user", "content": prompt}]
        sub_tools = [
            _TEAM_TOOL_DEFINITIONS[tool_name]
            for tool_name in normalized_tools
        ]

        work_state: dict[str, Any] = {
            "task_id": None,
            "report_task_id": None,
        }

        def send_message(to: str, content: str) -> str:
            try:
                canonical_to = _canonical_agent_name(to)
                if canonical_to != _LEAD_AGENT_NAME:
                    _validate_teammate_name(canonical_to)
                    with _teammate_lock:
                        if canonical_to not in active_teammates:
                            return f"Error: teammate '{canonical_to}' is not active"
            except (AttributeError, ValueError) as error:
                return f"Error: {error}"
            BUS.send(name, canonical_to, content)
            return "Sent"

        def list_task_lines() -> str:
            current = list_tasks()
            if not current:
                return "No tasks."
            return "\n".join(
                (
                    f"  {task.id}: {task.subject} [{task.status}] "
                    f"owner={task.owner or '-'} blockedBy={task.blockedBy}\n"
                    f"    {task.description or '(no description)'}"
                )
                for task in current
            )

        def claim_owned_task(task_id: str) -> str:
            try:
                candidate = load_task(task_id)
            except (FileNotFoundError, ValueError) as error:
                return f"Error: {error}"
            if (
                not get_team_settings()["auto_claim_enabled"]
                and candidate.assignee != name
            ):
                return (
                    "Error: manual assignment required while Team auto-claim "
                    f"is disabled; task {task_id} is not assigned to {name}"
                )
            result = claim_task(task_id, owner=name)
            if result.startswith("Claimed"):
                work_state["task_id"] = task_id
            return result

        def run_task_mutation(handler: Callable, **arguments) -> str:
            if await_assignment:
                task_id = work_state.get("task_id")
                try:
                    current = load_task(task_id) if task_id else None
                except (FileNotFoundError, ValueError):
                    current = None
                if (
                    current is None
                    or current.status != "in_progress"
                    or current.owner != name
                ):
                    return (
                        "Error: no active assigned task; workspace mutations "
                        "require a Task System assignment"
                    )
            if handler in {run_bash, run_write, run_edit}:
                arguments.setdefault("cancel_event", stop_event)
            return handler(**arguments)

        def complete_owned_task(task_id: str) -> str:
            result = complete_task(task_id, owner=name)
            if result.startswith("Completed"):
                try:
                    completed = load_task(task_id)
                    for intervention in completed.interventions:
                        if intervention.get("action") == "queue" and intervention.get("id"):
                            broker.update(
                                str(intervention["id"]),
                                "completed",
                                task_id=task_id,
                            )
                except (FileNotFoundError, KeyError, ValueError):
                    pass
                work_state["report_task_id"] = task_id
                if work_state["task_id"] == task_id:
                    work_state["task_id"] = None
            return result

        teammate_todos: list[dict[str, str]] = []

        def update_todos(todos: list[dict[str, str]]) -> str:
            if not isinstance(todos, list):
                return "Error: todos must be an array"
            normalized: list[dict[str, str]] = []
            for index, item in enumerate(todos):
                if not isinstance(item, dict):
                    return f"Error: todos[{index}] must be an object"
                content = str(item.get("content") or "").strip()
                status = str(item.get("status") or "")
                if not content:
                    return f"Error: todos[{index}].content is required"
                if status not in {"pending", "in_progress", "completed"}:
                    return f"Error: todos[{index}].status is invalid"
                normalized.append({"content": content, "status": status})
            teammate_todos[:] = normalized
            return f"Updated {len(normalized)} todos"

        sub_handlers = {
            "bash": lambda **arguments: run_task_mutation(
                run_bash, **arguments
            ),
            "read_file": run_read,
            "write_file": lambda **arguments: run_task_mutation(
                run_write, **arguments
            ),
            "edit_file": lambda **arguments: run_task_mutation(
                run_edit, **arguments
            ),
            "glob": run_glob,
            "todo_write": update_todos,
            "load_skill": load_skill,
            "web_search": run_web_search,
            "send_message": send_message,
            "list_tasks": list_task_lines,
            "get_task": get_task_json,
            "claim_task": claim_owned_task,
            "complete_task": complete_owned_task,
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
        }

        coordinator = None
        if _team_context_parent_resolver is not None:
            coordinator = create_child_context_coordinator(
                _team_context_parent_resolver(),
                agent_type="teammate",
                agent_id=name,
            )
        request_context = RequestContext(system=system, tools=sub_tools)
        latest_summary = ""
        failed = False
        broker = interaction_broker()
        interaction_target = f"team:{name}"

        def set_activity(phase: str, summary: str) -> None:
            task_id = work_state.get("task_id")
            _set_teammate_state(
                name,
                activity_phase=phase,
                activity_summary=summary,
            )
            broker.set_phase(
                interaction_target,
                phase,
                task_id=task_id,
                summary=summary,
            )
            notify(
                "teammate_activity",
                {
                    "name": name,
                    "phase": phase,
                    "summary": summary,
                    "task_id": task_id,
                },
            )

        def inject_interactions(actions: set[str]) -> list[AgentInteraction]:
            values = broker.consume(interaction_target, actions)
            for item in values:
                label = "追加要求" if item.action == "steer" else "方向修正"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<user-{item.action} id=\"{item.id}\">"
                            f"{label}：{item.content}"
                            f"</user-{item.action}>"
                        ),
                    }
                )
                task_id = work_state.get("task_id")
                if task_id:
                    try:
                        append_task_intervention(
                            task_id,
                            interaction_id=item.id,
                            action=item.action,
                            content=item.content,
                            status="injected",
                        )
                    except (FileNotFoundError, ValueError):
                        pass
                broker.update(item.id, "injected", task_id=task_id)
                notify(
                    "teammate_activity",
                    {
                        "name": name,
                        "phase": "interaction",
                        "summary": f"已应用用户 {item.action}：{item.content}",
                        "interaction_id": item.id,
                        "task_id": task_id,
                    },
                )
            return values

        try:
            iteration = 0
            attempted_recovery = False
            broker.clear_cancel(interaction_target)
            if await_assignment:
                set_activity("idle", "等待用户分配、队列任务或普通消息")
                initial_dispatch = idle_poll(
                    name,
                    messages,
                    name,
                    role,
                    stop_event,
                    work_state,
                    require_task=True,
                )
                if initial_dispatch in ("shutdown", "timeout"):
                    return
            while not stop_event.is_set():
                burst_has_task = bool(work_state.get("task_id"))
                _set_teammate_state(name, status="running")
                if len(messages) <= 3:
                    messages.insert(
                        0,
                        {
                            "role": "user",
                            "content": (
                                f"<identity>You are '{name}', role: {role}. "
                                "Continue your work.</identity>"
                            ),
                        },
                    )
                should_shutdown = False
                burst_complete = False
                rounds_this_burst = 0
                while rounds_this_burst < _team_max_rounds_per_burst:
                    if stop_event.is_set():
                        should_shutdown = True
                        break
                    with BUS.consume(name) as inbox:
                        for msg in inbox:
                            if handle_inbox_message(name, msg, messages):
                                should_shutdown = True
                                break
                    if should_shutdown:
                        break
                    non_protocol = [
                        msg for msg in inbox if msg.get("type") == "message"
                    ]
                    if non_protocol:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "<inbox>"
                                    + json.dumps(non_protocol)
                                    + "</inbox>"
                                ),
                            }
                        )
                    if protocol_ctx["waiting_plan"]:
                        waited = time.monotonic() - protocol_ctx["waiting_since"]
                        if waited >= PLAN_APPROVAL_TIMEOUT:
                            request_id = protocol_ctx["waiting_plan"]
                            with _protocol_lock:
                                state = pending_requests.get(request_id)
                                if state is not None and state.status == "pending":
                                    state.status = "expired"
                                    _persist_protocol_requests_locked()
                            protocol_ctx["waiting_plan"] = None
                            protocol_ctx["waiting_since"] = None
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"[Plan approval timed out: {request_id}] "
                                        "Revise the plan or stop safely."
                                    ),
                                }
                            )
                        else:
                            if stop_event.wait(IDLE_POLL_INTERVAL):
                                should_shutdown = True
                                break
                            continue

                    inject_interactions({"redirect", "steer"})

                    if coordinator is not None:
                        try:
                            provider_messages = coordinator.prepare_request(
                                messages, request_context
                            )
                        except ContextModeError:
                            if attempted_recovery:
                                raise
                            provider_messages = coordinator.reactive_recover(
                                messages,
                                request_context,
                                reason=CompressionReason.STRATEGY_FAILURE_RECOVERY,
                            )
                            attempted_recovery = True
                    else:
                        provider_messages = messages
                    iteration += 1
                    rounds_this_burst += 1
                    set_activity(
                        "llm_running",
                        "正在分析当前任务并规划下一步"
                        if burst_has_task
                        else "正在处理收到的普通消息",
                    )
                    try:
                        response = record_llm_call(
                            _team_provider,
                            model=config.MODEL or None,
                            system=system,
                            messages=provider_messages,
                            tools=sub_tools,
                            max_tokens=_team_max_tokens,
                            call_type="teammate",
                        )
                    except Exception as error:
                        if (
                            coordinator is not None
                            and is_context_length_error(error)
                            and not attempted_recovery
                        ):
                            coordinator.reactive_recover(
                                messages,
                                request_context,
                                reason=CompressionReason.PROVIDER_OVERFLOW,
                            )
                            attempted_recovery = True
                            continue
                        raise RuntimeError(
                            f"Teammate provider failed: {type(error).__name__}: {error}"
                        ) from error
                    if stop_event.is_set():
                        should_shutdown = True
                        break
                    if broker.pending(interaction_target, "redirect"):
                        notify(
                            "teammate_activity",
                            {
                                "name": name,
                                "phase": "redirecting",
                                "summary": "已丢弃过时模型响应，正在应用用户修正",
                                "task_id": work_state.get("task_id"),
                            },
                        )
                        inject_interactions({"redirect"})
                        continue
                    if response.stop_reason == "max_tokens":
                        raise RuntimeError(
                            "Teammate response reached the output-token limit"
                        )
                    messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                    tool_blocks = [
                        block
                        for block in response.content
                        if getattr(block, "type", None) == "tool_use"
                    ]
                    if not tool_blocks:
                        latest_summary = "".join(
                            getattr(block, "text", "")
                            for block in response.content
                            if getattr(block, "type", None) == "text"
                        ).strip()
                        if not latest_summary:
                            raise RuntimeError(
                                "Teammate finished without a text summary"
                            )
                        if broker.pending(interaction_target, "steer", "redirect"):
                            set_activity(
                                "finalizing",
                                "已形成阶段性结论，正在接收用户追加要求",
                            )
                            inject_interactions({"redirect", "steer"})
                            continue
                        set_activity(
                            "finalizing",
                            f"形成任务结论：{latest_summary[:500]}",
                        )
                        result_metadata = {}
                        report_task_id = (
                            work_state["report_task_id"] or work_state["task_id"]
                        )
                        if report_task_id:
                            result_metadata["task_id"] = report_task_id
                        BUS.send(
                            name,
                            "lead",
                            latest_summary,
                            "result",
                            result_metadata,
                        )
                        work_state["report_task_id"] = None
                        burst_complete = True
                        break
                    results = []
                    plan_submitted = False
                    for block_index, block in enumerate(tool_blocks):
                        if stop_event.is_set():
                            should_shutdown = True
                            break
                        set_activity(
                            "tool_running",
                            f"正在执行工具 {block.name}",
                        )
                        tool_started = time.monotonic()
                        if plan_submitted:
                            output = (
                                "Tool not executed because this tool group already "
                                "submitted a plan for approval."
                            )
                            tool_status = "blocked"
                        else:
                            with mutation_actor_scope(require_hash=True):
                                output, tool_status = _dispatch_teammate_tool(
                                    block, sub_handlers
                                )
                        if block.name == "submit_plan" and output.startswith(
                            "Plan submitted ("
                        ):
                            match = re.search(r"\((req_\d+)\)", output)
                            protocol_ctx["waiting_plan"] = (
                                match.group(1) if match else output
                            )
                            protocol_ctx["waiting_since"] = time.monotonic()
                            plan_submitted = True
                        notify(
                            "tool",
                            {
                                "iteration": iteration,
                                "tool": block.name,
                                "args": block.input,
                                "output": output,
                                "status": tool_status,
                                "latency_ms": round(
                                    (time.monotonic() - tool_started) * 1000
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
                        if broker.pending(interaction_target, "redirect"):
                            for skipped in tool_blocks[block_index + 1 :]:
                                results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": skipped.id,
                                        "content": (
                                            "Tool not executed because the user "
                                            "redirected the active task."
                                        ),
                                    }
                                )
                            break
                    if should_shutdown:
                        break
                    messages.append({"role": "user", "content": results})
                    if broker.pending(interaction_target, "redirect", "steer"):
                        inject_interactions({"redirect", "steer"})
                    if protocol_ctx["waiting_plan"]:
                        burst_complete = True
                        break
                if should_shutdown:
                    break
                if protocol_ctx["waiting_plan"]:
                    continue
                if not burst_complete:
                    raise RuntimeError(
                        "Teammate exceeded maximum rounds "
                        f"({_team_max_rounds_per_burst})"
                    )
                set_activity(
                    "idle",
                    "任务阶段结束，等待下一项队列任务"
                    if burst_has_task
                    else "消息处理完毕，等待用户分配、队列任务或新消息",
                )
                with _teammate_lock:
                    reload_requested = name in _teammate_profile_restart_pending
                if reload_requested and not work_state.get("task_id"):
                    stop_event.set()
                    break
                idle_result = idle_poll(
                    name,
                    messages,
                    name,
                    role,
                    stop_event,
                    work_state,
                    require_task=await_assignment,
                )
                if idle_result in ("shutdown", "timeout"):
                    break

        except Exception as error:
            failed = True
            set_activity("error", f"执行失败：{type(error).__name__}: {error}")
            notify(
                "teammate_error",
                {
                    "name": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            try:
                BUS.send(
                    name,
                    "lead",
                    f"Teammate '{name}' failed: {type(error).__name__}: {error}",
                    "error",
                )
            except Exception:
                pass
        finally:
            restart_with_saved_profile = False
            task_id = work_state.get("task_id")
            if task_id:
                try:
                    interrupt_task(task_id, name)
                except (FileNotFoundError, ValueError):
                    pass
            for item in broker.consume(interaction_target, {"stop"}):
                try:
                    broker.update(
                        item.id,
                        "completed",
                        task_id=task_id,
                        metadata={"agent_stopped": True},
                    )
                except KeyError:
                    pass
            set_activity(
                "error" if failed else "stopped",
                "执行失败，等待用户处理" if failed else "Agent 已停止",
            )
            if coordinator is not None:
                coordinator.close()
            notify("teammate_end", {"name": name})
            scope.__exit__(None, None, None)
            with _teammate_lock:
                restart_with_saved_profile = (
                    name in _teammate_profile_restart_pending
                    and not failed
                    and _team_accepting
                )
                _teammate_profile_restart_pending.discard(name)
                active_teammates.pop(name, None)
                _teammate_threads.pop(name, None)
                _teammate_stop_events.pop(name, None)
                state = _teammate_states.get(name)
                if state is not None:
                    state.update(
                        {
                            "status": "error" if failed else "stopped",
                            "online": False,
                            "current_task_id": None,
                            "last_active_at": time.time(),
                        }
                    )
            if restart_with_saved_profile:
                result = restart_teammate(name)
                if result.startswith("Error:"):
                    notify(
                        "teammate_error",
                        {
                            "name": name,
                            "error_type": "ProfileReloadError",
                            "error": result,
                        },
                    )

    parent_context = copy_context()
    thread = threading.Thread(
        target=lambda: parent_context.run(run),
        name=f"teammate-{name}",
        daemon=True,
    )
    with _teammate_lock:
        _teammate_threads[name] = thread
    thread.start()
    return f"Teammate '{name}' spawned as {role}"


def spawn_teammate_thread(
    name: str,
    role: str,
    prompt: str,
    *,
    allowed_tools: list[str] | None = None,
    await_assignment: bool = False,
    persist_profile: bool = True,
) -> str:
    with _teammate_lifecycle_lock:
        return _spawn_teammate_thread_unlocked(
            name,
            role,
            prompt,
            allowed_tools=allowed_tools,
            await_assignment=await_assignment,
            persist_profile=persist_profile,
        )


def stop_teammate(name: str) -> str:
    """Cooperatively stop one teammate without affecting the rest of the team."""

    try:
        _validate_teammate_name(name)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        _teammate_profile_restart_pending.discard(name)
        thread = _teammate_threads.get(name)
        stop_event = _teammate_stop_events.get(name)
        if (
            name not in active_teammates
            or thread is None
            or stop_event is None
        ):
            return f"Error: teammate '{name}' is not active"
        _set_teammate_state(name, status="stopping")
        stop_event.set()
    return f"Stop requested for {name}"


def restart_teammate(name: str) -> str:
    """Restart one stopped teammate from its persisted workspace profile."""

    with _teammate_lifecycle_lock:
        try:
            _validate_teammate_name(name)
        except ValueError as error:
            return f"Error: {error}"
        with _teammate_lock:
            if name in active_teammates:
                return f"Error: teammate '{name}' is already active"
        profile = _load_teammate_profiles().get(name)
        if profile is None:
            return f"Error: persisted profile for teammate '{name}' was not found"
        return _spawn_teammate_thread_unlocked(
            name,
            profile["role"],
            profile["prompt"],
            allowed_tools=profile["allowed_tools"],
            await_assignment=True,
            persist_profile=False,
        )


def delete_teammate(name: str) -> dict[str, Any]:
    """Delete one stopped teammate profile while preserving audit history."""

    with _teammate_lifecycle_lock:
        _validate_teammate_name(name)
        profiles = _load_teammate_profiles()
        with _teammate_lock:
            state = dict(_teammate_states.get(name) or {})
            thread = _teammate_threads.get(name)
            running = (
                name in active_teammates
                or bool(thread and thread.is_alive())
                or bool(state.get("online"))
                or state.get("status") in {"running", "stopping"}
            )
        if name not in profiles and not state:
            raise ValueError(f"teammate '{name}' was not found")
        if running:
            raise ValueError(f"teammate '{name}' is running and cannot be deleted")
        bound_tasks = [
            task.id
            for task in list_tasks()
            if task.status in {"pending", "in_progress"}
            and (task.assignee == name or task.owner == name)
        ]
        if bound_tasks:
            raise ValueError(
                f"teammate '{name}' still has unfinished tasks: "
                + ", ".join(bound_tasks)
            )
        profile_deleted = _delete_teammate_profile(name)
        with _teammate_lock:
            _teammate_profile_restart_pending.discard(name)
            _teammate_states.pop(name, None)
            _teammate_threads.pop(name, None)
            _teammate_stop_events.pop(name, None)
            active_teammates.pop(name, None)
        return {
            "name": name,
            "deleted": True,
            "profile_deleted": profile_deleted,
            "audit_preserved": True,
        }


def stop_all_teammates(timeout: float = 5.0) -> TeammateShutdownOutcome:
    """Cancel and join all source-faithful teammate threads."""
    global _team_accepting
    with _teammate_lock:
        _team_accepting = False
        _teammate_profile_restart_pending.clear()
    _teammate_stop_event.set()
    with _teammate_lock:
        for stop_event in _teammate_stop_events.values():
            stop_event.set()
        threads = list(_teammate_threads.items())
    deadline = time.monotonic() + timeout
    for _, thread in threads:
        if thread is threading.current_thread() or not thread.is_alive():
            continue
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    with _teammate_lock:
        for name, thread in list(_teammate_threads.items()):
            if not thread.is_alive():
                _teammate_threads.pop(name, None)
                active_teammates.pop(name, None)
                _teammate_stop_events.pop(name, None)
                state = _teammate_states.get(name)
                if state is not None:
                    state.update(
                        {
                            "status": "stopped",
                            "online": False,
                            "current_task_id": None,
                            "last_active_at": time.time(),
                        }
                    )
        live = tuple(
            name
            for name, thread in _teammate_threads.items()
            if thread.is_alive()
        )
    return TeammateShutdownOutcome(not live, live)


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    _validate_teammate_name(from_name)
    state = _create_protocol_request(
        "plan_approval", from_name, "lead", plan
    )
    try:
        BUS.send(
            from_name,
            "lead",
            plan,
            "plan_approval_request",
            {"request_id": state.request_id},
        )
    except Exception:
        with _protocol_lock:
            pending_requests.pop(state.request_id, None)
            _persist_protocol_requests_locked()
        raise
    return f"Plan submitted ({state.request_id})"


def run_request_shutdown(teammate: str) -> str:
    try:
        _validate_teammate_name(teammate)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        if teammate not in active_teammates:
            return f"Error: teammate '{teammate}' is not active"
    state = _create_protocol_request("shutdown", "lead", teammate, "")
    try:
        BUS.send(
            "lead",
            teammate,
            "Shut down.",
            "shutdown_request",
            {"request_id": state.request_id},
        )
    except Exception:
        with _protocol_lock:
            pending_requests.pop(state.request_id, None)
            _persist_protocol_requests_locked()
        raise
    return f"Shutdown request sent to {teammate}"


def run_request_plan(teammate: str, task: str) -> str:
    try:
        _validate_teammate_name(teammate)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        if teammate not in active_teammates:
            return f"Error: teammate '{teammate}' is not active"
    BUS.send("lead", teammate, f"Submit plan for: {task}", "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(
    request_id: str, approve: bool, feedback: str = ""
) -> str:
    with _protocol_lock:
        _ensure_protocol_state_loaded_locked()
        state = pending_requests.get(request_id)
        if not state:
            return f"Request {request_id} not found"
        if state.type != "plan_approval" or state.target != "lead":
            return f"Request {request_id} is not a plan approval request"
        if state.status != "pending":
            return f"Request {request_id} is already {state.status}"
        state.status = "approved" if approve else "rejected"
        try:
            BUS.send(
                "lead",
                state.sender,
                feedback or ("Approved" if approve else "Rejected"),
                "plan_approval_response",
                {"request_id": request_id, "approve": approve},
            )
        except Exception:
            state.status = "pending"
            _persist_protocol_requests_locked()
            raise
        _persist_protocol_requests_locked()
    return f"Plan {'approved' if approve else 'rejected'}"


def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    context = current_event_context()
    if (
        context.get("agent_type") == "main"
        and context.get("source") == "team_inbox"
    ):
        return (
            "Error: Team Agents cannot be created by an automatic Lead inbox "
            "Turn; wait for an explicit user Turn"
        )
    return spawn_teammate_thread(
        name,
        role,
        prompt,
        await_assignment=True,
    )


def _automatic_inbox_lifecycle_error(action: str) -> str | None:
    context = current_event_context()
    if (
        context.get("agent_type") == "main"
        and context.get("source") == "team_inbox"
    ):
        return (
            f"Error: an automatic Lead inbox Turn cannot {action} Team Agents; "
            "wait for an explicit user Turn"
        )
    return None


def run_stop_teammate(teammate: str) -> str:
    blocked = _automatic_inbox_lifecycle_error("stop")
    return blocked or stop_teammate(teammate)


def run_restart_teammate(teammate: str) -> str:
    blocked = _automatic_inbox_lifecycle_error("restart")
    return blocked or restart_teammate(teammate)


def run_send_message(to: str, content: str) -> str:
    try:
        canonical_to = _canonical_agent_name(to)
        if canonical_to == _LEAD_AGENT_NAME:
            return (
                "Error: you are the Lead Agent; send_message cannot be used "
                "to send a message to yourself"
            )
        _validate_teammate_name(canonical_to)
    except ValueError as error:
        return f"Error: {error}"
    with _teammate_lock:
        if canonical_to not in active_teammates:
            return f"Error: teammate '{canonical_to}' is not active"
    BUS.send(_LEAD_AGENT_NAME, canonical_to, content)
    return f"Sent to {canonical_to}"


def run_check_inbox() -> str:
    batch = claim_lead_inbox(route_protocol=True)
    try:
        msgs = list(batch.messages)
        if not msgs:
            rendered = "(inbox empty)"
        else:
            lines = []
            for message in msgs:
                meta = message.get("metadata", {})
                request_id = meta.get("request_id", "")
                tag = (
                    f" [{message['type']} req:{request_id}]"
                    if request_id
                    else f" [{message['type']}]"
                )
                lines.append(
                    f"  [{message['from']}]{tag} {message['content']}"
                )
            rendered = "\n".join(lines)
    except BaseException as error:
        nack_lead_inbox(batch, str(error))
        raise
    ack_lead_inbox(batch)
    return rendered
