from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .agent import SourceRuntime
from .background import initialize_background_tasks, shutdown_background_tasks
from .cron import shutdown_cron
from .permissions import PermissionPolicy
from .teams import stop_all_teammates
from .tools import TOOL_DEFINITIONS, TOOL_HANDLERS
from .trace import TraceRecorder


DEFAULT_FORBIDDEN_TOOLS = {
    "task",
    "schedule_cron",
    "list_crons",
    "cancel_cron",
    "spawn_teammate",
    "send_message",
    "check_inbox",
    "request_shutdown",
    "request_plan",
    "review_plan",
}


@dataclass(frozen=True)
class BenchmarkOptions:
    memory_enabled: bool = False
    cron_enabled: bool = False
    team_enabled: bool = False
    subagent_enabled: bool = False
    max_rounds: int = 40


@dataclass(frozen=True)
class BenchmarkCloseOutcome:
    stopped: bool
    live_resources: tuple[str, ...]


@dataclass
class BenchmarkSession:
    runtime: SourceRuntime
    workspace: Path
    options: BenchmarkOptions
    _closed: bool = False

    def close(self, timeout: float = 5.0) -> BenchmarkCloseOutcome:
        if self._closed:
            return BenchmarkCloseOutcome(True, ())
        self._closed = True
        background = shutdown_background_tasks(timeout)
        teammates = stop_all_teammates(timeout)
        cron_stopped = shutdown_cron(timeout)
        live = [f"background:{item}" for item in background.live_job_ids]
        live.extend(f"teammate:{item}" for item in teammates.live_names)
        if not cron_stopped:
            live.append("cron")
        return BenchmarkCloseOutcome(not live, tuple(live))


def validate_clean_workspace(run_dir: Path | str, workspace: Path | str) -> None:
    run_dir = Path(run_dir).resolve()
    workspace = Path(workspace).resolve()
    if workspace == run_dir:
        raise ValueError("agent workspace must not equal run directory")
    try:
        relative = workspace.relative_to(run_dir)
    except ValueError as error:
        raise ValueError("agent workspace must be inside run directory") from error
    current = run_dir
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("agent workspace path must not contain symlinks")
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("agent workspace must be empty")


def _benchmark_tool_tables(
    options: BenchmarkOptions,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forbidden = set(DEFAULT_FORBIDDEN_TOOLS)
    if options.subagent_enabled:
        forbidden.discard("task")
    if options.cron_enabled:
        forbidden.difference_update(
            {"schedule_cron", "list_crons", "cancel_cron"}
        )
    if options.team_enabled:
        forbidden.difference_update(
            {
                "spawn_teammate",
                "send_message",
                "check_inbox",
                "request_shutdown",
                "request_plan",
                "review_plan",
            }
        )
    definitions = [
        item for item in TOOL_DEFINITIONS if item["name"] not in forbidden
    ]
    names = {item["name"] for item in definitions}
    handlers = {name: handler for name, handler in TOOL_HANDLERS.items() if name in names}
    return definitions, handlers


def build_benchmark_runtime(
    *,
    run_dir: Path | str,
    workspace: Path | str,
    provider,
    recorder: TraceRecorder,
    options: BenchmarkOptions | None = None,
) -> BenchmarkSession:
    options = options or BenchmarkOptions()
    run_dir = Path(run_dir).resolve()
    workspace = Path(workspace).resolve()
    validate_clean_workspace(run_dir, workspace)
    workspace.mkdir(parents=True, exist_ok=False)
    config.configure_workspace(workspace)
    initialize_background_tasks()
    definitions, handlers = _benchmark_tool_tables(options)
    runtime = SourceRuntime(
        provider,
        PermissionPolicy(),
        None,
        recorder=recorder,
        tool_definitions=definitions,
        tool_handlers=handlers,
        max_rounds=options.max_rounds,
        memory_enabled=options.memory_enabled,
    )
    return BenchmarkSession(runtime, workspace, options)
