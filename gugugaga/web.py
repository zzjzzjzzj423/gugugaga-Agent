from __future__ import annotations

import argparse
import ipaddress
import json
import mimetypes
import os
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from . import config
from .__main__ import GugugagaApp, build_runtime
from .config import Settings
from .context_modes import ContextModeError
from .memory import memory_hit_kinds
from .memory.repository import MemoryRepository
from .observability import sanitize
from .permissions import PermissionBroker
from .subagents import snapshot_subagents
from .tasks import load_task, release_task, unassign_task
from .teams import (
    assign_task_to_teammate,
    get_team_settings,
    list_teammate_states,
    restart_teammate,
    stop_teammate,
    update_team_settings,
)
from .web_config import WebConfiguration


WEB_ASSETS = Path(__file__).with_name("web_assets")
MAX_REQUEST_BYTES = 1_000_000
CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class EventHub:
    """Small long-poll event buffer for the local dashboard."""

    def __init__(self, max_events: int = 500):
        self._condition = threading.Condition()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._next_id = 1

    def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            value = {**sanitize(event), "event_id": self._next_id}
            self._next_id += 1
            self._events.append(value)
            self._condition.notify_all()
            return value

    @property
    def latest_id(self) -> int:
        with self._condition:
            return self._next_id - 1

    def wait_after(self, event_id: int, timeout: float = 20.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not any(item["event_id"] > event_id for item in self._events):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return [dict(item) for item in self._events if item["event_id"] > event_id]


class DashboardStore:
    """Read-only dashboard queries over gugugaga's real local state."""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace).expanduser().resolve()
        self.state_dir = self.workspace / ".gugugaga"
        self.database = self.state_dir / "state.db"
        self.repository = MemoryRepository(self.database)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _last_usage(self, session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return {}
        path = self.state_dir / "usage.jsonl"
        if not path.exists():
            return {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                value = json.loads(line)
                if (
                    value.get("session_id") == session_id
                    and value.get("agent_type", "main") == "main"
                    and value.get("call_type", "agent") == "agent"
                ):
                    return value
            return {}
        except (OSError, ValueError, TypeError):
            return {}

    def overview(
        self,
        runtime: Any | None = None,
        memory_hits: int = 0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        coordinator = getattr(runtime, "context_coordinator", None)
        current_session_id = getattr(coordinator, "session_id", None)
        selected_session_id = session_id or current_session_id
        turn_count = 0
        last_assistant = None
        if selected_session_id:
            with self._connect() as connection:
                turn_count = int(
                    connection.execute(
                        "SELECT COUNT(DISTINCT turn_id) FROM chat_log WHERE session_id=?",
                        (selected_session_id,),
                    ).fetchone()[0]
                )
                last_assistant = connection.execute(
                    "SELECT meta, created_at FROM chat_log "
                    "WHERE session_id=? AND role='assistant' ORDER BY id DESC LIMIT 1",
                    (selected_session_id,),
                ).fetchone()
        meta: dict[str, Any] = {}
        if last_assistant:
            try:
                meta = json.loads(str(last_assistant["meta"] or "{}"))
            except (TypeError, ValueError):
                meta = {}
        memory = self.repository.status()
        stored_context = meta.get("context")
        if runtime is not None and selected_session_id == current_session_id:
            context = runtime.context_status()
        elif isinstance(stored_context, dict):
            context = stored_context
        else:
            mode = os.getenv("GUGUGAGA_CONTEXT_MODE", "cc")
            context = {
                "mode": mode,
                "display_name": mode.upper(),
                "lifecycle": "idle",
                "context_window_tokens": int(
                    os.getenv("GUGUGAGA_CONTEXT_WINDOW_TOKENS", "131072")
                ),
                "successful_compactions": 0,
                "locked": bool(turn_count),
            }
        usage = self._last_usage(selected_session_id)
        input_tokens = usage.get("input_tokens")
        window = int(context.get("context_window_tokens") or 131_072)
        if isinstance(input_tokens, int):
            ratio = round((int(input_tokens) / window) * 100, 1)
        else:
            ratio = 0.0 if turn_count == 0 else None
        return {
            "session_id": selected_session_id,
            "turn_count": turn_count,
            "last_latency_ms": meta.get("latency_ms"),
            "memory_hits": memory_hits if selected_session_id == current_session_id else 0,
            "context_ratio": ratio,
            "context_tokens": input_tokens,
            "context_window_tokens": window,
            "model": meta.get("model") or os.getenv("SILICONFLOW_MODEL") or "not configured",
            "provider": meta.get("provider") or "siliconflow",
            "memory": memory,
            "context": context,
            "last_turn_at": last_assistant["created_at"] if last_assistant else None,
        }

    @staticmethod
    def _excerpt(value: Any, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    def sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.session_id,
                       MIN(c.created_at) AS created_at,
                       MAX(c.created_at) AS updated_at,
                       COUNT(DISTINCT c.turn_id) AS turn_count,
                       (SELECT first.content FROM chat_log AS first
                        WHERE first.session_id=c.session_id AND first.role='user'
                        ORDER BY first.id ASC LIMIT 1) AS title,
                       (SELECT recent.content FROM chat_log AS recent
                        WHERE recent.session_id=c.session_id
                        ORDER BY recent.id DESC LIMIT 1) AS preview,
                       (SELECT recent_assistant.meta FROM chat_log AS recent_assistant
                        WHERE recent_assistant.session_id=c.session_id
                          AND recent_assistant.role='assistant'
                        ORDER BY recent_assistant.id DESC LIMIT 1) AS context_meta,
                       MAX(c.id) AS latest_id
                FROM chat_log AS c
                GROUP BY c.session_id
                ORDER BY latest_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": str(row["session_id"]),
                "title": self._excerpt(row["title"], 42) or "未命名对话",
                "preview": self._excerpt(row["preview"], 72),
                "turn_count": int(row["turn_count"] or 0),
                "context_mode": self._context_mode_from_meta(row["context_meta"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _context_mode_from_meta(value: Any) -> str:
        try:
            meta = json.loads(str(value or "{}"))
            mode = str(meta.get("context", {}).get("mode", "cc")).lower()
            return mode if mode in {"cc", "hermes", "pi"} else "cc"
        except (TypeError, ValueError):
            return "cc"

    def session_context_mode(self, session_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT meta FROM chat_log WHERE session_id=? AND role='assistant' "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return self._context_mode_from_meta(row["meta"] if row else None)

    def chat_history(self, session_id: str | None, limit: int = 100) -> list[dict[str, Any]]:
        if not session_id:
            return []
        limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, turn_id, role, content, created_at FROM chat_log "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def runtime_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM chat_log WHERE session_id=? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [
            {
                "role": str(row["role"]),
                "content": str(row["content"])
                if row["role"] == "user"
                else [{"type": "text", "text": str(row["content"])}],
            }
            for row in rows
        ]

    def _procedural_memories(self, query: str | None = None) -> list[dict[str, Any]]:
        values = [
            {
                "id": "runtime_tool_permission",
                "kind": "procedural",
                "subject": "Tool Permission Policy",
                "text": "Shell 和后台命令必须经过显式批准；其他固定工具按策略直接执行。",
                "status": "active",
                "source": "runtime_rule",
                "occurred_at": None,
            },
            {
                "id": "runtime_memory_validation",
                "kind": "procedural",
                "subject": "Memory Write Validation",
                "text": "记忆写入前执行格式校验、凭据拦截与活动事实去重。",
                "status": "active",
                "source": "runtime_rule",
                "occurred_at": None,
            },
            {
                "id": "runtime_context_compression",
                "kind": "procedural",
                "subject": "Context Compression",
                "text": "按会话配置使用 CC、Hermes 或 Pi 模式压缩上下文，并保留工具协议边界。",
                "status": "active",
                "source": "runtime_rule",
                "occurred_at": None,
            },
        ]
        skills_dir = self.state_dir / "skills"
        if skills_dir.exists():
            for manifest in sorted(skills_dir.glob("*/SKILL.md")):
                try:
                    raw = manifest.read_text(encoding="utf-8")[:16_000]
                except OSError:
                    continue
                name_match = re.search(r"(?m)^name:\s*[\"']?([^\n\"']+)", raw)
                description_match = re.search(r"(?m)^description:\s*[\"']?([^\n\"']+)", raw)
                values.append(
                    {
                        "id": f"skill_{manifest.parent.name}",
                        "kind": "procedural",
                        "subject": (name_match.group(1).strip() if name_match else manifest.parent.name),
                        "text": (
                            description_match.group(1).strip()
                            if description_match
                            else "Workspace skill"
                        ),
                        "status": "active",
                        "source": "skill",
                        "occurred_at": None,
                    }
                )
        if query:
            folded = query.casefold()
            values = [
                item
                for item in values
                if folded in f"{item['subject']} {item['text']}".casefold()
            ]
        return values

    def memories(self, kind: str, query: str | None = None, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(limit, 100))
        if kind == "procedural":
            items = self._procedural_memories(query)[:limit]
        else:
            pattern = f"%{query}%" if query else None
            with self._connect() as connection:
                if kind == "semantic":
                    where = "WHERE subject LIKE ? OR content LIKE ?" if pattern else ""
                    params: tuple[Any, ...] = (pattern, pattern, limit) if pattern else (limit,)
                    rows = connection.execute(
                        "SELECT id, 'semantic' AS kind, subject, content AS text, status, "
                        "source, updated_at AS occurred_at, seen_count FROM facts "
                        f"{where} ORDER BY updated_at DESC LIMIT ?",
                        params,
                    ).fetchall()
                elif kind == "episodic":
                    where = "WHERE summary LIKE ?" if pattern else ""
                    params = (pattern, limit) if pattern else (limit,)
                    rows = connection.execute(
                        "SELECT id, 'episodic' AS kind, '' AS subject, summary AS text, "
                        "status, 'implicit' AS source, period_end AS occurred_at, "
                        "period_start, period_end FROM episodes "
                        f"{where} ORDER BY period_end DESC LIMIT ?",
                        params,
                    ).fetchall()
                else:
                    raise ValueError("kind must be procedural, semantic, or episodic")
            items = [dict(row) for row in rows]
        status = self.repository.status()
        return {
            "kind": kind,
            "items": items,
            "counts": {
                "procedural": len(self._procedural_memories()),
                "semantic": int(status.get("facts", 0)),
                "episodic": int(status.get("episodes", 0)),
            },
        }

    def tables(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            names = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            values = []
            for name in names:
                quoted = _quote_identifier(name)
                count = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
                columns = len(connection.execute(f"PRAGMA table_info({quoted})").fetchall())
                values.append({"name": name, "row_count": count, "column_count": columns})
        return values

    def table_view(self, name: str, view: str = "rows", limit: int = 50) -> dict[str, Any]:
        available = {item["name"] for item in self.tables()}
        if name not in available:
            raise KeyError("unknown table")
        if view not in {"rows", "schema", "indexes"}:
            raise ValueError("view must be rows, schema, or indexes")
        limit = max(1, min(limit, 100))
        quoted = _quote_identifier(name)
        with self._connect() as connection:
            if view == "schema":
                rows = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
                columns = ["cid", "name", "type", "not_null", "default", "primary_key"]
                data = [
                    {
                        "cid": row[0],
                        "name": row[1],
                        "type": row[2],
                        "not_null": bool(row[3]),
                        "default": row[4],
                        "primary_key": bool(row[5]),
                    }
                    for row in rows
                ]
            elif view == "indexes":
                columns = ["name", "unique", "origin", "partial", "columns"]
                data = []
                for row in connection.execute(f"PRAGMA index_list({quoted})").fetchall():
                    index_name = str(row[1])
                    index_quoted = _quote_identifier(index_name)
                    index_columns = [
                        str(item[2])
                        for item in connection.execute(f"PRAGMA index_info({index_quoted})").fetchall()
                    ]
                    data.append(
                        {
                            "name": index_name,
                            "unique": bool(row[2]),
                            "origin": row[3],
                            "partial": bool(row[4]),
                            "columns": index_columns,
                        }
                    )
            else:
                columns = [
                    str(row[1])
                    for row in connection.execute(f"PRAGMA table_info({quoted})").fetchall()
                ]
                try:
                    rows = connection.execute(
                        f"SELECT * FROM {quoted} ORDER BY rowid DESC LIMIT ?", (limit,)
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = connection.execute(f"SELECT * FROM {quoted} LIMIT ?", (limit,)).fetchall()
                data = [
                    {column: _json_safe(row[column]) for column in columns}
                    for row in rows
                ]
        return {"table": name, "view": view, "columns": columns, "rows": data}

    @staticmethod
    def _next_cron_run(expression: str) -> str | None:
        """Return the next local-time match supported by gugugaga's cron parser."""
        from .cron import cron_matches

        candidate = datetime.now().astimezone().replace(second=0, microsecond=0)
        candidate += timedelta(minutes=1)
        for _ in range(366 * 24 * 60):
            try:
                if cron_matches(expression, candidate):
                    return candidate.isoformat()
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            candidate += timedelta(minutes=1)
        return None

    def _scheduled_tasks(self) -> list[dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        durable_path = self.workspace / ".scheduled_tasks.json"
        if durable_path.exists():
            try:
                raw_jobs = json.loads(durable_path.read_text(encoding="utf-8"))
                if isinstance(raw_jobs, list):
                    for raw in raw_jobs:
                        if isinstance(raw, dict) and raw.get("id") and raw.get("cron"):
                            values[str(raw["id"])] = dict(raw)
            except (OSError, TypeError, ValueError):
                pass

        # Once the runtime is active, include non-durable jobs that exist only
        # in memory. Guard against exposing another configured workspace in tests.
        from . import config as runtime_config
        from .cron import cron_lock, scheduled_jobs

        try:
            same_workspace = runtime_config.WORKDIR.resolve() == self.workspace
        except OSError:
            same_workspace = False
        if same_workspace:
            with cron_lock:
                for job_id, job in scheduled_jobs.items():
                    values[job_id] = {
                        "id": job.id,
                        "cron": job.cron,
                        "prompt": job.prompt,
                        "recurring": job.recurring,
                        "durable": job.durable,
                    }

        items = []
        for raw in values.values():
            expression = str(raw.get("cron", ""))
            items.append(
                {
                    "id": str(raw.get("id", "")),
                    "cron": expression,
                    "prompt": self._excerpt(raw.get("prompt"), 500),
                    "recurring": bool(raw.get("recurring", True)),
                    "durable": bool(raw.get("durable", True)),
                    "next_run": self._next_cron_run(expression),
                }
            )
        return sorted(
            items,
            key=lambda item: (
                item["next_run"] is None,
                item["next_run"] or "",
                item["id"],
            ),
        )

    def task_system(self) -> dict[str, Any]:
        tasks_dir = self.workspace / ".tasks"
        raw_tasks: list[dict[str, Any]] = []
        if tasks_dir.exists():
            for path in sorted(tasks_dir.glob("task_*.json")):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(raw, dict) or not raw.get("id"):
                        continue
                    raw["_path"] = path
                    raw_tasks.append(raw)
                except (OSError, TypeError, ValueError):
                    continue

        status_by_id = {
            str(raw["id"]): str(raw.get("status", "pending")) for raw in raw_tasks
        }
        tasks = []
        for raw in raw_tasks:
            task_id = str(raw["id"])
            dependencies = [str(value) for value in (raw.get("blockedBy") or [])]
            dependency_details = [
                {"id": dependency, "status": status_by_id.get(dependency, "missing")}
                for dependency in dependencies
            ]
            blocked = any(item["status"] != "completed" for item in dependency_details)
            path = raw.pop("_path")
            timestamp_match = re.match(r"task_(\d+)_", task_id)
            created_at = None
            if timestamp_match:
                try:
                    created_at = datetime.fromtimestamp(
                        int(timestamp_match.group(1)), timezone.utc
                    ).isoformat()
                except (OverflowError, OSError, ValueError):
                    pass
            try:
                updated_at = datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat()
            except OSError:
                updated_at = None
            tasks.append(
                {
                    "id": task_id,
                    "subject": self._excerpt(raw.get("subject"), 200) or "未命名任务",
                    "description": self._excerpt(raw.get("description"), 1000),
                    "status": status_by_id[task_id],
                    "owner": raw.get("owner"),
                    "assignee": raw.get("assignee"),
                    "blocked_by": dependencies,
                    "dependencies": dependency_details,
                    "blocked": blocked and status_by_id[task_id] == "pending",
                    "ready": status_by_id[task_id] == "pending" and not blocked,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )

        order = {"in_progress": 0, "pending": 1, "completed": 2}
        tasks.sort(
            key=lambda item: (
                order.get(item["status"], 3),
                item["created_at"] or "",
                item["id"],
            )
        )
        scheduled = self._scheduled_tasks()
        counts = {
            "total": len(tasks),
            "pending": sum(item["status"] == "pending" for item in tasks),
            "in_progress": sum(item["status"] == "in_progress" for item in tasks),
            "completed": sum(item["status"] == "completed" for item in tasks),
            "blocked": sum(item["blocked"] for item in tasks),
            "scheduled": len(scheduled),
        }
        counts["progress"] = round(
            (counts["completed"] / counts["total"] * 100) if counts["total"] else 0
        )
        return {"counts": counts, "tasks": tasks, "scheduled_tasks": scheduled}

    def _subagent_trace_events(
        self,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        directory = self.state_dir / "traces"
        if not directory.exists():
            return []
        events: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.jsonl"), reverse=True)[:7]:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if event.get("agent_type") != "subagent":
                    continue
                if session_id and event.get("session_id") != session_id:
                    continue
                if turn_id and event.get("turn_id") != turn_id:
                    continue
                events.append(event)
                if len(events) >= limit:
                    return list(reversed(events))
        return list(reversed(events))

    def subagent_current(self, turn_id: str | None = None) -> dict[str, Any]:
        jobs = snapshot_subagents(turn_id)
        selected_turn = turn_id or next(
            (
                str(job.get("parent_turn_id"))
                for job in reversed(jobs)
                if job.get("parent_turn_id")
            ),
            None,
        )
        events = (
            self._subagent_trace_events(turn_id=selected_turn, limit=500)
            if selected_turn
            else []
        )
        return {
            "turn_id": selected_turn,
            "items": jobs,
            "events": events,
        }

    def subagent_history(
        self, session_id: str | None = None, *, current_turn_id: str | None = None
    ) -> dict[str, Any]:
        events = self._subagent_trace_events(session_id=session_id, limit=5000)
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for event in events:
            turn_id = str(event.get("turn_id") or "")
            agent_id = str(event.get("agent_id") or "")
            if not turn_id or not agent_id or turn_id == current_turn_id:
                continue
            key = (turn_id, agent_id)
            item = grouped.setdefault(
                key,
                {
                    "turn_id": turn_id,
                    "subagent_id": agent_id,
                    "session_id": event.get("session_id"),
                    "status": "running",
                    "description": "",
                    "tool_count": 0,
                    "started_at": event.get("timestamp"),
                    "finished_at": None,
                    "summary": "",
                },
            )
            event_type = event.get("type")
            if event_type == "subagent_start":
                item["description"] = self._excerpt(event.get("description"), 500)
            elif event_type == "tool":
                item["tool_count"] += 1
            elif event_type == "subagent_end":
                item["status"] = "completed"
                item["finished_at"] = event.get("timestamp")
                item["summary"] = self._excerpt(
                    event.get("reply")
                    or event.get("result")
                    or event.get("summary"),
                    500,
                )
            elif event_type == "subagent_status" and event.get("status") in {
                "completed",
                "failed",
                "cancelled",
                "timed_out",
            }:
                item["status"] = event.get("status")
                item["finished_at"] = event.get("timestamp")
                if event.get("error"):
                    item["summary"] = self._excerpt(event.get("error"), 500)
            elif event_type in {"subagent_error", "turn_error"}:
                item["status"] = "failed"
                item["finished_at"] = event.get("timestamp")
                item["summary"] = self._excerpt(event.get("error"), 500)
        items = sorted(
            grouped.values(),
            key=lambda item: item.get("started_at") or "",
            reverse=True,
        )[:100]
        return {"items": items}


@dataclass
class ChatResult:
    reply: str
    memory_hits: int
    session_id: str


class DashboardApplication:
    def __init__(
        self,
        workspace: Path | str,
        *,
        model: str | None = None,
        runtime_factory: Callable[[], GugugagaApp] | None = None,
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        config.configure_workspace(self.workspace)
        self.model = model
        self.configuration = WebConfiguration(self.workspace)
        self.configuration.apply_environment()
        self.store = DashboardStore(self.workspace)
        self.events = EventHub()
        self.permissions = PermissionBroker(timeout_seconds=120)
        self._runtime_factory = runtime_factory
        self._app: GugugagaApp | None = None
        self._runtime_error: str | None = None
        self._runtime_lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._unsubscribe: Callable[[], None] | None = None
        self._active_turn_id: str | None = None
        self.last_memory_hits = 0

    def _build_runtime(self) -> GugugagaApp:
        if self._runtime_factory is not None:
            return self._runtime_factory()
        effective = self.configuration.effective(self.model)
        settings = Settings.from_env(self.workspace, effective["model"] or None)
        return build_runtime(
            settings, approval_callback=self.permissions.callback
        )

    def runtime(self) -> Any:
        if self._app is not None:
            return self._app.runtime
        with self._runtime_lock:
            if self._app is None:
                try:
                    self._app = self._build_runtime()
                    self._unsubscribe = self._app.runtime.recording.observer.subscribe(
                        "*", self._forward_runtime_event
                    )
                    starter = getattr(self._app, "start_lead_inbox_loop", None)
                    if callable(starter):
                        starter(self._on_lead_inbox_reply)
                    self._runtime_error = None
                except Exception as error:
                    self._runtime_error = str(error)
                    raise
        return self._app.runtime

    def _forward_runtime_event(self, event: dict[str, Any]) -> None:
        """Translate context decisions into explicit dashboard graph transitions."""
        if event.get("type") == "turn_start":
            self._active_turn_id = str(event.get("turn_id") or "") or None
        elif event.get("type") in {"turn_end", "turn_error"}:
            if event.get("turn_id") == self._active_turn_id:
                self._active_turn_id = None
        if (
            event.get("type") == "tool"
            and event.get("tool") == "load_skill"
            and event.get("status") == "ok"
        ):
            self.events.publish(
                {
                    "type": "memory",
                    "action": "recall",
                    "status": "hit",
                    "kinds": ["procedural"],
                }
            )
        if event.get("type") == "context":
            self.events.publish(
                {"type": "runtime", "stage": "compression_gate", "status": "active"}
            )
        self.events.publish(event)
        if event.get("type") == "context" and event.get("status") != "failed":
            self.events.publish(
                {"type": "runtime", "stage": "agent", "status": "active"}
            )

    def _on_lead_inbox_reply(self, reply: str) -> None:
        runtime = self._app.runtime if self._app is not None else None
        coordinator = getattr(runtime, "context_coordinator", None)
        self.events.publish(
            {
                "type": "lead_inbox_reply",
                "status": "complete",
                "reply": reply,
                "session_id": getattr(coordinator, "session_id", None),
            }
        )

    def status(self) -> dict[str, Any]:
        runtime = self._app.runtime if self._app is not None else None
        coordinator = getattr(runtime, "context_coordinator", None)
        context_status = coordinator.status() if coordinator is not None else None
        configuration = self.configuration.public(self.model)
        return {
            "ok": True,
            "workspace": str(self.workspace),
            "runtime_ready": self._app is not None,
            "chat_configured": bool(
                configuration["siliconflow_api_key_configured"]
                and configuration["model"]
            ) or self._runtime_factory is not None,
            "web_search_configured": configuration["tavily_api_key_configured"],
            "runtime_error": self._runtime_error,
            "event_id": self.events.latest_id,
            "session_id": getattr(coordinator, "session_id", None),
            "context_mode": context_status.get("mode") if context_status else None,
            "context_mode_locked": bool(context_status and context_status.get("locked")),
        }

    def configuration_status(self) -> dict[str, Any]:
        return self.configuration.public(self.model)

    def update_configuration(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("another turn is already running")
        try:
            value = self.configuration.update(payload, self.model)
            previous = self._app
            session_state = None
            if previous is not None:
                runtime = previous.runtime
                coordinator = getattr(runtime, "context_coordinator", None)
                session_state = (
                    getattr(coordinator, "session_id", None),
                    list(getattr(runtime, "messages", [])),
                    getattr(getattr(coordinator, "mode", None), "value", None),
                )
                if self._unsubscribe is not None:
                    self._unsubscribe()
                    self._unsubscribe = None
                self._app = None
                previous.close()

            reloaded = False
            reload_error = None
            if previous is not None:
                try:
                    runtime = self.runtime()
                    if session_state and session_state[0] and hasattr(
                        runtime, "restore_session_state"
                    ):
                        runtime.restore_session_state(*session_state)
                    reloaded = True
                except Exception as error:
                    reload_error = str(error)
            self.events.publish(
                {
                    "type": "configuration",
                    "action": "updated",
                    "status": "complete" if reload_error is None else "failed",
                    "runtime_reloaded": reloaded,
                }
            )
            return {
                **value,
                "runtime_reloaded": reloaded,
                "reload_error": reload_error,
            }
        finally:
            self._turn_lock.release()

    def sessions(self) -> dict[str, Any]:
        current_session_id = self.status()["session_id"]
        items = self.store.sessions()
        if current_session_id and not any(
            item["session_id"] == current_session_id for item in items
        ):
            items.insert(
                0,
                {
                    "session_id": current_session_id,
                    "title": "新对话",
                    "preview": "长期记忆仍然可用",
                    "turn_count": 0,
                    "context_mode": self.status()["context_mode"] or "cc",
                    "created_at": None,
                    "updated_at": None,
                },
            )
        return {"current_session_id": current_session_id, "items": items}

    def new_session(self, context_mode: str | None = None) -> str:
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("another turn is already running")
        try:
            runtime = self.runtime()
            session_id = runtime.start_new_session(context_mode)
            self.last_memory_hits = 0
            self.events.publish(
                {
                    "type": "session",
                    "action": "created",
                    "status": "complete",
                    "session_id": session_id,
                    "context_mode": runtime.context_coordinator.mode.value,
                }
            )
            return session_id
        finally:
            self._turn_lock.release()

    def resume_session(self, session_id: str) -> tuple[str, int]:
        value = session_id.strip()
        if not value or len(value) > 200:
            raise ValueError("valid session_id is required")
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("another turn is already running")
        try:
            messages = self.store.runtime_history(value)
            if not messages:
                raise KeyError("conversation not found")
            runtime = self.runtime()
            context_mode = self.store.session_context_mode(value)
            resumed = runtime.resume_session(value, messages, context_mode)
            self.last_memory_hits = 0
            self.events.publish(
                {
                    "type": "session",
                    "action": "resumed",
                    "status": "complete",
                    "session_id": resumed,
                    "message_count": len(messages),
                    "context_mode": context_mode,
                }
            )
            return resumed, len(messages)
        finally:
            self._turn_lock.release()

    def set_context_mode(self, context_mode: str) -> dict[str, Any]:
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("another turn is already running")
        try:
            runtime = self.runtime()
            runtime.context_coordinator.set_mode(context_mode)
            value = runtime.context_coordinator.status()
            self.events.publish(
                {
                    "type": "session",
                    "action": "context_mode_selected",
                    "status": "complete",
                    "session_id": runtime.context_coordinator.session_id,
                    "context_mode": value["mode"],
                }
            )
            return value
        finally:
            self._turn_lock.release()

    def chat(self, message: str) -> ChatResult:
        value = message.strip()
        if not value:
            raise ValueError("message is required")
        if len(value) > 20_000:
            raise ValueError("message is too long")
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("another turn is already running")
        try:
            runtime = self.runtime()
            self.events.publish({"type": "runtime", "stage": "input", "status": "active"})
            self.events.publish({"type": "runtime", "stage": "retrieval_gate", "status": "active"})
            recalled = runtime.memory_service.recall(value)
            hits = len(re.findall(r"(?m)^- \[", recalled))
            kinds = list(memory_hit_kinds(recalled))
            self.last_memory_hits = hits
            self.events.publish(
                {
                    "type": "runtime",
                    "stage": "memory_injection" if hits else "working_context",
                    "status": "active",
                    "memory_hits": hits,
                    "memory_kinds": kinds,
                }
            )
            self.events.publish({"type": "runtime", "stage": "working_context", "status": "active"})
            self.events.publish({"type": "runtime", "stage": "compression_gate", "status": "active"})
            reply = runtime.run_turn(value, source="web")
            self.events.publish({"type": "runtime", "stage": "reply", "status": "complete"})
            return ChatResult(
                reply=reply,
                memory_hits=hits,
                session_id=runtime.context_coordinator.session_id,
            )
        except Exception as error:
            self.events.publish(
                {"type": "runtime", "stage": "error", "status": "error", "error": str(error)}
            )
            raise
        finally:
            self._turn_lock.release()

    def overview(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id is not None and (not session_id.strip() or len(session_id) > 200):
            raise ValueError("valid session_id is required")
        runtime = self._app.runtime if self._app is not None else None
        return self.store.overview(runtime, self.last_memory_hits, session_id)

    def pending_permissions(self) -> dict[str, Any]:
        return {"items": self.permissions.pending()}

    def team_settings(self) -> dict[str, Any]:
        return get_team_settings()

    def set_team_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        value = payload.get("auto_claim_enabled")
        if not isinstance(value, bool):
            raise ValueError("auto_claim_enabled must be boolean")
        settings = update_team_settings(value)
        self.events.publish(
            {
                "type": "team_settings",
                "action": "updated",
                **settings,
            }
        )
        return settings

    def team_agents(self) -> dict[str, Any]:
        return {"items": list_teammate_states()}

    def stop_team_agent(self, name: str) -> dict[str, Any]:
        result = stop_teammate(name)
        if result.startswith("Error:"):
            raise ValueError(result.removeprefix("Error:").strip())
        self.events.publish(
            {"type": "team_agent", "action": "stop_requested", "name": name}
        )
        return {"name": name, "status": "stopping", "message": result}

    def restart_team_agent(self, name: str) -> dict[str, Any]:
        # A stopped profile can outlive the Web process. Rebuild the shared
        # runtime first so its provider and permission policy are installed
        # before the teammate thread is recreated.
        self.runtime()
        result = restart_teammate(name)
        if result.startswith("Error:"):
            raise ValueError(result.removeprefix("Error:").strip())
        self.events.publish(
            {"type": "team_agent", "action": "restarted", "name": name}
        )
        return {"name": name, "status": "running", "message": result}

    def assign_task(self, task_id: str, teammate: str) -> dict[str, Any]:
        value = assign_task_to_teammate(task_id, teammate)
        self.events.publish(
            {
                "type": "task_assignment",
                "action": "assigned",
                "task_id": task_id,
                "teammate": teammate,
            }
        )
        return value

    def unassign_task(self, task_id: str) -> dict[str, Any]:
        value = asdict(unassign_task(task_id))
        self.events.publish(
            {
                "type": "task_assignment",
                "action": "unassigned",
                "task_id": task_id,
            }
        )
        return value

    def release_task(self, task_id: str) -> dict[str, Any]:
        task = load_task(task_id)
        active_owner = next(
            (
                agent
                for agent in list_teammate_states()
                if agent.get("online")
                and agent.get("name") == task.owner
                and agent.get("current_task_id") == task.id
            ),
            None,
        )
        if active_owner is not None:
            raise ValueError(
                f"task {task_id} is still running on teammate {task.owner}"
            )
        value = asdict(release_task(task_id))
        self.events.publish(
            {
                "type": "task_assignment",
                "action": "released",
                "task_id": task_id,
            }
        )
        return value

    def subagent_current(self, turn_id: str | None = None) -> dict[str, Any]:
        return self.store.subagent_current(turn_id or self._active_turn_id)

    def subagent_history(self, session_id: str | None = None) -> dict[str, Any]:
        return self.store.subagent_history(
            session_id, current_turn_id=self._active_turn_id
        )

    def review_permission(
        self, permission_id: str, approve: bool, feedback: str = ""
    ) -> dict[str, Any]:
        return self.permissions.review(permission_id, approve, feedback)

    def close(self) -> None:
        self.permissions.close()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._app is not None:
            self._app.close()
            self._app = None


def _handler_factory(application: DashboardApplication):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "GugugagaWeb/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _headers(self, content_type: str, length: int, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'")
            self.end_headers()

        def _json(self, value: Any, status: int = 200) -> None:
            payload = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            self._write_payload(
                payload,
                "application/json; charset=utf-8",
                status,
            )

        def _write_payload(
            self,
            payload: bytes,
            content_type: str,
            status: int = HTTPStatus.OK,
        ) -> bool:
            try:
                self._headers(content_type, len(payload), status)
                self.wfile.write(payload)
            except CLIENT_DISCONNECT_ERRORS:
                # Browsers routinely cancel an outstanding long-poll request
                # while refreshing, navigating, or closing the page. There is
                # no client left to receive an error response, so end the
                # request quietly instead of attempting a second write.
                self.close_connection = True
                return False
            return True

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _read_json_body(self, *, allow_empty: bool = False) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if allow_empty and length == 0:
                return {}
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            return payload

        def _is_loopback(self) -> bool:
            try:
                return ipaddress.ip_address(self.client_address[0]).is_loopback
            except ValueError:
                return False

        def _asset(self, path: Path) -> None:
            try:
                payload = path.read_bytes()
            except OSError:
                self._error(HTTPStatus.NOT_FOUND, "asset not found")
                return
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._write_payload(payload, f"{media_type}; charset=utf-8")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._asset(WEB_ASSETS / "index.html")
                elif parsed.path == "/assets/styles.css":
                    self._asset(WEB_ASSETS / "styles.css")
                elif parsed.path == "/assets/app.js":
                    self._asset(WEB_ASSETS / "app.js")
                elif parsed.path == "/assets/gugugaga-avatar.png":
                    self._asset(WEB_ASSETS / "gugugaga-avatar.png")
                elif parsed.path == "/api/status":
                    self._json(application.status())
                elif parsed.path == "/api/config":
                    self._json(application.configuration_status())
                elif parsed.path == "/api/overview":
                    session_id = query.get("session_id", [None])[0]
                    self._json(application.overview(session_id))
                elif parsed.path == "/api/sessions":
                    self._json(application.sessions())
                elif parsed.path == "/api/chat/history":
                    session_id = query.get("session_id", [application.status()["session_id"]])[0]
                    self._json(
                        {
                            "session_id": session_id,
                            "items": application.store.chat_history(session_id),
                        }
                    )
                elif parsed.path == "/api/memories":
                    kind = query.get("kind", ["semantic"])[0]
                    search = query.get("q", [None])[0]
                    self._json(application.store.memories(kind, search))
                elif parsed.path == "/api/tasks":
                    self._json(application.store.task_system())
                elif parsed.path == "/api/team/settings":
                    self._json(application.team_settings())
                elif parsed.path == "/api/team/agents":
                    self._json(application.team_agents())
                elif parsed.path == "/api/subagents":
                    turn_id = query.get("turn_id", [None])[0]
                    self._json(application.subagent_current(turn_id))
                elif parsed.path == "/api/subagents/history":
                    session_id = query.get("session_id", [None])[0]
                    self._json(application.subagent_history(session_id))
                elif parsed.path == "/api/database/tables":
                    self._json({"items": application.store.tables()})
                elif parsed.path == "/api/database/table":
                    name = unquote(query.get("name", [""])[0])
                    view = query.get("view", ["rows"])[0]
                    limit = int(query.get("limit", ["50"])[0])
                    self._json(application.store.table_view(name, view, limit))
                elif parsed.path == "/api/events":
                    after = int(query.get("after", ["0"])[0])
                    timeout = min(float(query.get("timeout", ["20"])[0]), 25.0)
                    items = application.events.wait_after(after, timeout)
                    self._json({"items": items})
                elif parsed.path == "/api/permissions":
                    if not self._is_loopback():
                        self._error(
                            HTTPStatus.FORBIDDEN,
                            "permission review requires a local connection",
                        )
                    else:
                        self._json(application.pending_permissions())
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
            except KeyError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
            except (TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            task_action = re.fullmatch(
                r"/api/tasks/(task_[0-9]+_[0-9]{4})/(assign|unassign|release)",
                parsed.path,
            )
            agent_action = re.fullmatch(
                r"/api/team/agents/([A-Za-z0-9][A-Za-z0-9_-]{0,63})/(stop|restart)",
                parsed.path,
            )
            if parsed.path not in {
                "/api/chat",
                "/api/config",
                "/api/session/new",
                "/api/session/resume",
                "/api/session/mode",
                "/api/permissions/review",
            } and task_action is None and agent_action is None:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                if (
                    parsed.path in {"/api/config", "/api/permissions/review"}
                    or task_action
                    or agent_action
                ):
                    if not self._is_loopback():
                        self._error(
                            HTTPStatus.FORBIDDEN,
                            "configuration and permission changes require a local connection",
                        )
                        return
                if task_action is not None:
                    payload = self._read_json_body(allow_empty=True)
                    task_id, action = task_action.groups()
                    try:
                        if action == "assign":
                            teammate = str(payload.get("teammate", "")).strip()
                            if not teammate:
                                raise ValueError("teammate is required")
                            self._json(application.assign_task(task_id, teammate))
                        elif action == "unassign":
                            self._json(application.unassign_task(task_id))
                        else:
                            self._json(application.release_task(task_id))
                    except FileNotFoundError:
                        self._error(HTTPStatus.NOT_FOUND, "task not found")
                    except ValueError as error:
                        self._error(HTTPStatus.CONFLICT, str(error))
                    return
                if agent_action is not None:
                    self._read_json_body(allow_empty=True)
                    name, action = agent_action.groups()
                    try:
                        if action == "stop":
                            self._json(application.stop_team_agent(name))
                        else:
                            self._json(application.restart_team_agent(name))
                    except ValueError as error:
                        self._error(HTTPStatus.CONFLICT, str(error))
                    return
                if parsed.path == "/api/session/new":
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = {}
                    if length:
                        if length > MAX_REQUEST_BYTES:
                            raise ValueError("invalid request size")
                        payload = json.loads(self.rfile.read(length).decode("utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("JSON object required")
                    session_id = application.new_session(
                        str(payload.get("context_mode", "") or "") or None
                    )
                    self._json(
                        {
                            "created": True,
                            "session_id": session_id,
                            "context_mode": application.status()["context_mode"],
                        },
                        HTTPStatus.CREATED,
                    )
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
                if parsed.path == "/api/config":
                    self._json(application.update_configuration(payload))
                    return
                if parsed.path == "/api/permissions/review":
                    permission_id = str(payload.get("permission_id", ""))
                    if not permission_id:
                        raise ValueError("permission_id is required")
                    approve = payload.get("approve")
                    if not isinstance(approve, bool):
                        raise ValueError("approve must be boolean")
                    self._json(
                        application.review_permission(
                            permission_id,
                            approve,
                            str(payload.get("feedback", "")),
                        )
                    )
                    return
                if parsed.path == "/api/session/resume":
                    session_id, message_count = application.resume_session(
                        str(payload.get("session_id", ""))
                    )
                    self._json(
                        {
                            "resumed": True,
                            "session_id": session_id,
                            "message_count": message_count,
                            "context_mode": application.status()["context_mode"],
                        }
                    )
                    return
                if parsed.path == "/api/session/mode":
                    context = application.set_context_mode(
                        str(payload.get("context_mode", ""))
                    )
                    self._json(
                        {
                            "session_id": application.status()["session_id"],
                            "context_mode": context["mode"],
                            "locked": context["locked"],
                        }
                    )
                    return
                result = application.chat(str(payload.get("message", "")))
                self._json(
                    {
                        "reply": result.reply,
                        "memory_hits": result.memory_hits,
                        "session_id": result.session_id,
                    },
                    HTTPStatus.OK,
                )
            except ContextModeError as error:
                self._error(HTTPStatus.CONFLICT, error.safe_message)
            except RuntimeError as error:
                self._error(HTTPStatus.CONFLICT, str(error))
            except KeyError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            except Exception as error:
                status = HTTPStatus.SERVICE_UNAVAILABLE if "SILICONFLOW" in str(error) else HTTPStatus.INTERNAL_SERVER_ERROR
                self._error(status, str(error))

        def do_PUT(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/team/settings":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            if not self._is_loopback():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "team setting changes require a local connection",
                )
                return
            try:
                self._json(application.set_team_settings(self._read_json_body()))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    return DashboardHandler


def create_server(
    application: DashboardApplication,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler_factory(application))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gugugaga local web console")
    parser.add_argument("--workspace", default=".", help="Workspace used by the Agent")
    parser.add_argument("--model", help="Override SILICONFLOW_MODEL")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    application = DashboardApplication(args.workspace, model=args.model)
    server = create_server(application, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"gugugaga Web | {url} | workspace={application.workspace}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        application.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
