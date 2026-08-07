from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import agent as agent_module
from . import config
from . import context as context_module
from . import subagents
from .agent import SourceRuntime, agent_lock, agent_loop
from .background import initialize_background_tasks, shutdown_background_tasks
from .config import Settings
from .context import block_type, update_context
from .cron import (
    consume_cron_queue,
    initialize_cron,
    shutdown_cron,
)
from .hooks import trigger_hooks
from .models import ChatProvider, ToolCall
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


PROMPT = "\033[36msimple-cc >> \033[0m"
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
class SimpleCCApp:
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
            background_outcome = shutdown_background_tasks(remaining())
            teammate_outcome = stop_all_teammates(remaining())
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
                live.append("simple-cc-cron-scheduler")
            if self.autorun_thread is not None and self.autorun_thread.is_alive():
                live.append(self.autorun_thread.name)
            self._close_outcome = ApplicationCloseOutcome(not live, tuple(live))
            return self._close_outcome


def build_runtime(
    settings: Settings,
    approval_callback=None,
    provider: ChatProvider | None = None,
) -> SimpleCCApp:
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
    permissions = PermissionPolicy()
    initialize_background_tasks()
    set_team_provider(provider, permissions, approval_callback)
    initialize_cron()
    return SimpleCCApp(
        settings=settings,
        runtime=SourceRuntime(
            provider,
            permissions,
            approval_callback,
        ),
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple CC using SiliconFlow")
    parser.add_argument(
        "--workspace", default=".", help="Workspace the agent may access"
    )
    parser.add_argument("--model", help="Override SILICONFLOW_MODEL")
    return parser


def handle_command(command: str, app: SimpleCCApp) -> tuple[bool, str]:
    command = command.strip()
    if command in {"/exit", "/quit"}:
        return True, "__exit__"
    if command == "/help":
        return True, "/help /status /tasks /team /memory /exit"
    if command == "/status":
        teammates = ", ".join(sorted(active_teammates)) or "none"
        return True, (
            f"Workspace: {app.settings.workspace}\n"
            f"Model: {app.settings.model}\n"
            f"Active teammates: {teammates}"
        )
    if command == "/tasks":
        return True, run_list_tasks()
    if command == "/team":
        return True, (
            ", ".join(sorted(active_teammates))
            or "No active teammates."
        )
    if command == "/memory":
        if not config.MEMORY_INDEX.exists():
            return True, "(no memories)"
        return True, config.MEMORY_INDEX.read_text()[:4000]
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
):
    stop_event = stop_event or threading.Event()
    while not stop_event.wait(1):
        fired = consume_cron_queue()
        if not fired:
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
            )
            context.update(update_context(context, history))
            print_turn_assistants(history, turn_start)


def main(argv: list[str] | None = None) -> int:
    global CLI_ACTIVE
    args = create_parser().parse_args(argv)
    try:
        settings = Settings.from_env(Path(args.workspace), args.model)
        app = build_runtime(settings, approval_callback=_approval_prompt)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    CLI_ACTIVE = True
    print(f"Simple CC | model={settings.model} | workspace={settings.workspace}")
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
        ),
        name="simple-cc-cron-autorun",
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
            trigger_hooks("UserPromptSubmit", query)
            turn_start = len(history)
            history.append({"role": "user", "content": query})
            with agent_lock:
                agent_loop(
                    history,
                    context,
                    app.runtime.permissions,
                    app.runtime.approval_callback,
                )
                context = update_context(context, history)
                app.runtime.context = context
                print_turn_assistants(history, turn_start)

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
