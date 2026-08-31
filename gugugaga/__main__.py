from __future__ import annotations

import argparse
import json
import shlex
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import agent as agent_module
from . import config
from . import context as context_module
from . import subagents
from .agent import SourceRuntime, agent_lock, agent_loop
from .background import initialize_background_tasks, shutdown_background_tasks
from .config import Settings
from .context import block_type, update_context
from .context_modes import (
    ContextModeError,
    SessionContextConfig,
    SessionContextCoordinator,
    TokenCounter,
    TokenCounterRegistry,
)
from .cron import (
    consume_cron_queue,
    initialize_cron,
    shutdown_cron,
)
from .hooks import trigger_hooks
from .models import ChatProvider, ToolCall
from .memory import MemoryService
from .observability import RecordingSystem, set_default_observer
from .permissions import PermissionPolicy
from .provider import SiliconFlowProvider
from .teams import (
    active_teammates,
    consume_lead_inbox,
    set_team_provider,
    stop_all_teammates,
)
from .tasks import run_list_tasks


try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False


PROMPT = "\033[36mgugugaga >> \033[0m"
CLI_ACTIVE = False


@dataclass(frozen=True)
class ApplicationCloseOutcome:
    stopped: bool
    live_threads: tuple[str, ...]


def terminal_print(text: str):
    if threading.current_thread() is threading.main_thread() or not CLI_ACTIVE:
        print(text)
        return
    line = ""
    if READLINE_AVAILABLE:
        try:
            line = readline.get_line_buffer()
        except Exception:
            line = ""
    print(f"\r\033[K{text}")
    print(PROMPT + line, end="", flush=True)


@dataclass
class GugugagaApp:
    settings: Settings
    runtime: SourceRuntime
    stop_event: threading.Event = field(default_factory=threading.Event)
    autorun_thread: threading.Thread | None = None
    _closed: bool = False
    _close_outcome: ApplicationCloseOutcome | None = None
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    def close(self, timeout: float = 5.0) -> ApplicationCloseOutcome:
        with self._close_lock:
            if self._closed:
                return self._close_outcome
            self._closed = True
            deadline = time.monotonic() + max(0.0, timeout)

            def remaining() -> float:
                return max(0.0, deadline - time.monotonic())

            self.stop_event.set()
            self.runtime.memory_service.close(remaining())
            self.runtime.context_coordinator.close()
            background_outcome = shutdown_background_tasks(remaining())
            teammate_outcome = stop_all_teammates(remaining())
            set_team_provider(None)
            subagents.reset_subagent_runtime()
            cron_stopped = shutdown_cron(remaining())
            if (
                self.autorun_thread is not None
                and self.autorun_thread is not threading.current_thread()
                and self.autorun_thread.is_alive()
            ):
                self.autorun_thread.join(timeout=remaining())
            live = [
                *(f"background:{job_id}" for job_id in background_outcome.live_job_ids),
                *(f"teammate:{name}" for name in teammate_outcome.live_names),
            ]
            if not cron_stopped:
                live.append("gugugaga-cron-scheduler")
            if self.autorun_thread is not None and self.autorun_thread.is_alive():
                live.append(self.autorun_thread.name)
            self._close_outcome = ApplicationCloseOutcome(not live, tuple(live))
            return self._close_outcome


def build_runtime(
    settings: Settings,
    approval_callback=None,
    provider: ChatProvider | None = None,
    *,
    context_mode: str | None = None,
    context_config: SessionContextConfig | None = None,
    token_counter: TokenCounter | None = None,
) -> GugugagaApp:
    provider = provider or SiliconFlowProvider(settings)
    config.configure_workspace(settings.workspace)
    config.MODEL = settings.model
    config.PRIMARY_MODEL = settings.model
    config.FALLBACK_MODEL = __import__("os").getenv(
        "SILICONFLOW_FALLBACK_MODEL"
    )
    agent_module.client = provider
    context_module.client = provider
    subagents.client = provider
    subagents.MODEL = settings.model
    if context_config is None:
        selected_mode = context_mode if context_mode is not None else settings.context_mode
        source = (
            "programmatic"
            if context_mode is not None
            else settings.context_mode_source
        )
        context_config = SessionContextConfig.parse(
            selected_mode,
            source=source,
            context_window_tokens=settings.context_window_tokens,
            token_counter_id=settings.token_counter_id,
            token_counter_version=settings.token_counter_version,
            hermes_threshold_ratio=settings.hermes_threshold_ratio,
            hermes_target_ratio=settings.hermes_target_ratio,
            pi_reserve_tokens=settings.pi_reserve_tokens,
            pi_keep_recent_tokens=settings.pi_keep_recent_tokens,
        )
    elif context_mode is not None and context_mode != context_config.mode.value:
        context_config = SessionContextConfig.parse(
            context_mode,
            source="programmatic",
            context_window_tokens=context_config.context_window_tokens,
            token_counter_id=context_config.token_counter_id,
            token_counter_version=context_config.token_counter_version,
            hermes_threshold_ratio=context_config.hermes_threshold_ratio,
            hermes_target_ratio=context_config.hermes_target_ratio,
            pi_reserve_tokens=context_config.pi_reserve_tokens,
            pi_keep_recent_tokens=context_config.pi_keep_recent_tokens,
        )
    counter_registry = TokenCounterRegistry()
    if token_counter is not None:
        counter_registry.register(token_counter)
    coordinator = SessionContextCoordinator(
        context_config,
        counter_registry=counter_registry,
        summary_callback=agent_module._summary_callback(provider),
        workspace=settings.workspace,
        transcripts_dir=config.TRANSCRIPT_DIR,
        memory_dir=config.MEMORY_DIR,
        tool_results_dir=config.TOOL_RESULTS_DIR,
    )
    recording = RecordingSystem(settings.state_dir)
    set_default_observer(recording.observer)
    memory_service = MemoryService(
        settings.state_dir / "state.db",
        provider,
        enabled=settings.memory_enabled,
        explicit_enabled=settings.memory_explicit_enabled,
        consolidation_enabled=settings.memory_consolidation_enabled,
        threshold=settings.memory_consolidation_exchange_threshold,
        model=settings.memory_consolidation_model,
        timeout_seconds=settings.memory_consolidation_timeout_seconds,
        lease_seconds=settings.memory_consolidation_lease_seconds,
        max_facts=settings.memory_consolidation_max_facts,
        min_importance=settings.memory_consolidation_min_importance,
        recall_token_budget=settings.memory_recall_token_budget,
    )
    permissions = PermissionPolicy()
    initialize_background_tasks()
    initialize_cron()
    runtime = SourceRuntime(
        provider,
        permissions,
        approval_callback,
        coordinator,
        recording,
        memory_service,
    )
    context_parent_resolver = lambda: runtime.context_coordinator
    subagents.configure_subagent_runtime(
        provider,
        model=settings.model,
        permissions=permissions,
        approval_callback=approval_callback,
        context_parent_resolver=context_parent_resolver,
        max_rounds=settings.max_rounds,
        max_tokens=settings.max_tokens,
    )
    set_team_provider(
        provider,
        permissions,
        approval_callback,
        context_parent_resolver=context_parent_resolver,
        max_tokens=settings.max_tokens,
        max_rounds_per_burst=settings.max_rounds,
    )
    return GugugagaApp(
        settings=settings,
        runtime=runtime,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gugugaga using SiliconFlow")
    parser.add_argument(
        "--workspace", default=".", help="Workspace the agent may access"
    )
    parser.add_argument("--model", help="Override SILICONFLOW_MODEL")
    parser.add_argument(
        "--context-mode",
        default="cc",
        metavar="{cc,hermes,pi}",
        help="Session context mode; exact lowercase value required (default: cc)",
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=131_072,
        help="Model context window in tokens (default: 131072)",
    )
    parser.add_argument(
        "--token-counter-id",
        default="gugugaga_estimator_v1",
        help="Registered request token counter",
    )
    parser.add_argument("--hermes-threshold-ratio", type=float, default=0.50)
    parser.add_argument("--hermes-target-ratio", type=float, default=0.20)
    parser.add_argument("--pi-reserve-tokens", type=int, default=16_384)
    parser.add_argument("--pi-keep-recent-tokens", type=int, default=20_000)
    return parser


def handle_command(command: str, app: GugugagaApp) -> tuple[bool, str]:
    command = command.strip()
    if command in {"/exit", "/quit"}:
        return True, "__exit__"
    if command == "/help":
        return True, (
            "/help /status /tasks /team /memory /exit\n"
            "/memory [list|status|search <text>|show <id>|update <fact_id> <text>|"
            "forget <id>|retry]"
        )
    if command == "/status":
        teammates = ", ".join(sorted(active_teammates)) or "none"
        context_status = app.runtime.context_status()
        last = context_status["last_result"] or "none"
        return True, (
            f"Workspace: {app.settings.workspace}\n"
            f"Model: {app.settings.model}\n"
            f"Context mode: {context_status['display_name']} "
            f"(source={context_status['source']}, locked={context_status['locked']})\n"
            f"Context counter: {context_status['token_counter_id']} "
            f"{context_status['token_counter_version']} / "
            f"{context_status['context_window_tokens']} tokens\n"
            f"Successful compactions: {context_status['successful_compactions']}\n"
            f"Last context result: {last}\n"
            f"Reactive recovery used: {context_status['recovery_used']}\n"
            f"Active teammates: {teammates}"
        )
    if command == "/tasks":
        return True, run_list_tasks()
    if command == "/team":
        return True, (
            ", ".join(sorted(active_teammates))
            or "No active teammates."
        )
    if command == "/memory" or command.startswith("/memory "):
        try:
            parts = shlex.split(command)
        except ValueError as error:
            return True, f"memory command error: {error}"
        action = parts[1] if len(parts) > 1 else "list"
        service = app.runtime.memory_service
        if action == "status":
            return True, json.dumps(service.status(), ensure_ascii=False, indent=2)
        if action == "retry":
            return True, f"memory retry scheduled for {service.retry_failed()} chat rows"
        if action == "search" and len(parts) >= 3:
            rows = service.list_memories(" ".join(parts[2:]))
        elif action == "show" and len(parts) == 3:
            row = service.get_memory(parts[2])
            return True, json.dumps(row, ensure_ascii=False, indent=2) if row else "not_found"
        elif action == "update" and len(parts) >= 4:
            result = service.update_fact(parts[2], " ".join(parts[3:]))
            return True, result.to_json()
        elif action == "forget" and len(parts) == 3:
            memory_id = parts[2]
            kind = "episode" if memory_id.startswith("episode_") else "fact"
            return True, service.forget(kind, memory_id)
        elif action in {"list", "search"}:
            rows = service.list_memories()
        else:
            return True, (
                "usage: /memory [list|status|search <text>|show <id>|"
                "update <fact_id> <text>|forget <id>|retry]"
            )
        if not rows:
            return True, "(no memories)"
        return True, "\n".join(
            f"{row['id']} [{row['kind']}/{row['status']}/{row['source']}] "
            f"{row['subject'] + ': ' if row['subject'] else ''}{row['text']}"
            for row in rows
        )[:8000]
    return False, ""


def _approval_prompt(call: ToolCall) -> bool:
    if threading.current_thread() is not threading.main_thread():
        print(
            "\n[permission deferred] Run this request in the foreground to "
            f"approve: {call.name}"
        )
        return False
    print(
        f"\nPermission required: {call.name} "
        f"{json.dumps(call.arguments, ensure_ascii=False)}"
    )
    return input("Allow? [y/N] ").strip().lower() in {"y", "yes"}


def print_turn_assistants(messages: list, turn_start: int):
    for message in messages[turn_start:]:
        if message.get("role") != "assistant":
            continue
        for block in message.get("content", []):
            if block_type(block) == "text":
                terminal_print(
                    block["text"] if isinstance(block, dict) else block.text
                )


def cron_autorun_loop(
    history: list,
    context: dict,
    stop_event: threading.Event | None = None,
    permissions: PermissionPolicy | None = None,
    approval_callback=None,
    memory_state: dict | None = None,
    context_coordinator: SessionContextCoordinator | None = None,
    runtime: SourceRuntime | None = None,
):
    stop_event = stop_event or threading.Event()
    while not stop_event.wait(1):
        fired = consume_cron_queue()
        if not fired:
            continue
        if runtime is not None:
            query = "\n".join(f"[Scheduled] {job.prompt}" for job in fired)
            for job in fired:
                terminal_print(
                    f"  \033[35m[cron auto] {job.prompt[:60]}\033[0m"
                )
            reply = runtime.run_turn(query, source="cron")
            if reply:
                terminal_print(reply)
            continue
        with agent_lock:
            turn_start = len(history)
            for job in fired:
                history.append(
                    {"role": "user", "content": f"[Scheduled] {job.prompt}"}
                )
                terminal_print(
                    f"  \033[35m[cron auto] {job.prompt[:60]}\033[0m"
                )
            agent_loop(
                history,
                context,
                permissions,
                approval_callback,
                memory_state,
                context_coordinator,
            )
            memory_index = (
                context_coordinator.memory_dir / "MEMORY.md"
                if context_coordinator is not None
                else None
            )
            context.update(update_context(context, history, memory_index))
            print_turn_assistants(history, turn_start)


def main(argv: list[str] | None = None) -> int:
    global CLI_ACTIVE
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = create_parser().parse_args(raw_argv)
    try:
        settings = Settings.from_env(Path(args.workspace), args.model)
        context_config = SessionContextConfig.parse(
            args.context_mode,
            source="cli" if "--context-mode" in raw_argv else "default",
            context_window_tokens=args.context_window_tokens,
            token_counter_id=args.token_counter_id,
            hermes_threshold_ratio=args.hermes_threshold_ratio,
            hermes_target_ratio=args.hermes_target_ratio,
            pi_reserve_tokens=args.pi_reserve_tokens,
            pi_keep_recent_tokens=args.pi_keep_recent_tokens,
        )
        settings = replace(
            settings,
            context_mode=context_config.mode.value,
            context_mode_source=context_config.source,
            context_window_tokens=context_config.context_window_tokens,
            token_counter_id=context_config.token_counter_id,
            token_counter_version=context_config.token_counter_version,
            hermes_threshold_ratio=context_config.hermes_threshold_ratio,
            hermes_target_ratio=context_config.hermes_target_ratio,
            pi_reserve_tokens=context_config.pi_reserve_tokens,
            pi_keep_recent_tokens=context_config.pi_keep_recent_tokens,
        )
        app = build_runtime(settings, approval_callback=_approval_prompt)
    except ContextModeError as error:
        print(
            f"Configuration error [{error.code}]: {error.safe_message} "
            f"History preserved: {error.history_preserved}. "
            f"Next: {error.suggested_action}",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    CLI_ACTIVE = True
    print(f"gugugaga | model={settings.model} | workspace={settings.workspace}")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = app.runtime.messages
    context = app.runtime.context
    app.autorun_thread = threading.Thread(
        target=cron_autorun_loop,
        args=(
            history,
            context,
            app.stop_event,
            app.runtime.permissions,
            app.runtime.approval_callback,
            getattr(app.runtime, "memory_state", None),
            getattr(app.runtime, "context_coordinator", None),
            app.runtime,
        ),
        name="gugugaga-cron-autorun",
        daemon=True,
    )
    app.autorun_thread.start()
    try:
        while True:
            try:
                query = input(PROMPT)
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in {"q", "exit", "/exit", "/quit", ""}:
                break
            handled, output = handle_command(query, app)
            if handled:
                if output == "__exit__":
                    break
                print(output)
                continue
            reply = app.runtime.run_turn(query, source="cli")
            context = app.runtime.context
            if reply:
                terminal_print(reply)

            inbox = consume_lead_inbox(route_protocol=True)
            if inbox:
                def inbox_label(message):
                    request_id = message.get("metadata", {}).get(
                        "request_id", ""
                    )
                    suffix = f" req:{request_id}" if request_id else ""
                    return f"{message.get('type', 'message')}{suffix}"

                inbox_text = "\n".join(
                    f"From {message['from']} [{inbox_label(message)}]: "
                    f"{message['content'][:200]}"
                    for message in inbox
                )
                history.append(
                    {"role": "user", "content": f"[Inbox]\n{inbox_text}"}
                )
            print()
    finally:
        app.close()
        CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
