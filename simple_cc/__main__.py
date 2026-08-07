from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import agent as agent_module
from . import config as config_module
from . import context as context_module
from . import subagents as subagents_module
from .agent import AgentRuntime, SubagentRunner, agent_lock, agent_loop
from .background import BackgroundManager, CronScheduler
from .config import Settings
from .context import (
    ContextManager,
    MemoryStore,
    SkillStore,
    block_type,
    update_context,
)
from .cron import consume_cron_queue, load_durable_jobs
from .hooks import HookManager, install_audit_hooks, trigger_hooks
from .models import ChatProvider, ToolCall
from .permissions import PermissionPolicy
from .planning import TaskStore, TodoStore
from .prompts import PromptAssembler
from .provider import SiliconFlowProvider
from .teams import (
    Mailbox,
    ProtocolStore,
    TeamManager,
    active_teammates,
    consume_lead_inbox,
    set_team_provider,
)
from .tasks import run_list_tasks
from .tools import ToolRegistry, WorkspaceTools


try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False


PROMPT = "\033[36msimple-cc >> \033[0m"
CLI_ACTIVE = False


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


def _schema(properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties or {}, "required": required or []}


@dataclass
class SimpleCCApp:
    settings: Settings
    runtime: AgentRuntime
    tasks: TaskStore
    todos: TodoStore
    memory: MemoryStore
    skills: SkillStore
    background: BackgroundManager
    cron: CronScheduler
    team: TeamManager
    stop_event: threading.Event = field(default_factory=threading.Event)

    def close(self) -> None:
        self.stop_event.set()
        self.team.stop_all()
        self.cron.stop()


def build_runtime(
    settings: Settings,
    approval_callback=None,
    provider: ChatProvider | None = None,
) -> SimpleCCApp:
    provider = provider or SiliconFlowProvider(settings)
    config_module.configure_workspace(settings.workspace)
    load_durable_jobs()
    config_module.MODEL = settings.model
    config_module.PRIMARY_MODEL = settings.model
    config_module.FALLBACK_MODEL = __import__("os").getenv(
        "SILICONFLOW_FALLBACK_MODEL"
    )
    agent_module.client = provider
    context_module.client = provider
    subagents_module.client = provider
    subagents_module.MODEL = settings.model
    set_team_provider(provider)
    tasks = TaskStore(settings.tasks_dir)
    todos = TodoStore()
    memory = MemoryStore(settings.memory_dir)
    skills = SkillStore([
        settings.workspace / ".agents" / "skills",
        settings.skills_dir,
    ])
    background = BackgroundManager()
    cron = CronScheduler(settings.state_dir / "cron.json")
    context = ContextManager(settings.outputs_dir, settings.transcripts_dir)
    hooks = HookManager()
    install_audit_hooks(hooks, settings.state_dir / "hooks.log")
    permissions = PermissionPolicy()
    prompts = PromptAssembler()
    mailbox = Mailbox(settings.mailboxes_dir)
    protocols = ProtocolStore()

    team = TeamManager(
        mailbox,
        tasks,
        protocols,
        runtime_factory=lambda name, role: None,
        poll_seconds=settings.idle_poll_seconds,
        idle_timeout=settings.idle_timeout_seconds,
    )

    def task_list_text() -> str:
        items = tasks.list()
        return json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2) if items else "No tasks."

    def lead_inbox() -> list[str]:
        messages = team.check_inbox()
        return [
            f"<teammate-message from='{item['from']}' type='{item['type']}'>{item['content']}</teammate-message>"
            for item in messages
        ]

    def build_agent(name: str, role: str, teammate: bool = False) -> AgentRuntime:
        registry = ToolRegistry()
        workspace_tools = WorkspaceTools(settings.workspace)
        workspace_tools.register_into(registry)

        registry.register("todo_write", "Replace the session todo list", _schema({"todos": {"type": "array", "items": {"type": "object"}}}, ["todos"]), todos.update)
        registry.register("load_skill", "Load full instructions for a discovered skill", _schema({"name": {"type": "string"}}, ["name"]), skills.load)
        registry.register("remember", "Persist a durable memory", _schema({"title": {"type": "string"}, "content": {"type": "string"}}, ["title", "content"]), memory.remember)
        registry.register("search_memory", "Search durable memories", _schema({"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]), memory.search)
        registry.register("compact", "Compact conversation history now", _schema(), lambda: "Compaction requested")
        registry.register("create_task", "Create a persistent task with dependencies", _schema({"subject": {"type": "string"}, "description": {"type": "string"}, "blocked_by": {"type": "array", "items": {"type": "string"}}}, ["subject"]), lambda subject, description="", blocked_by=None: json.dumps(asdict(tasks.create(subject, description, blocked_by)), ensure_ascii=False))
        registry.register("list_tasks", "List persistent tasks", _schema(), task_list_text)
        registry.register("get_task", "Get one persistent task", _schema({"task_id": {"type": "string"}}, ["task_id"]), lambda task_id: json.dumps(asdict(tasks.get(task_id)), ensure_ascii=False, indent=2))
        registry.register("claim_task", "Claim an available task", _schema({"task_id": {"type": "string"}}, ["task_id"]), lambda task_id: tasks.claim(task_id, name))
        registry.register("complete_task", "Complete an owned task", _schema({"task_id": {"type": "string"}}, ["task_id"]), tasks.complete)
        registry.register("background_run", "Run a shell command in a background thread", _schema({"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]), lambda command, timeout=300: background.start(ToolCall(f"bgcall-{name}", "bash", {"command": command}), lambda: workspace_tools.bash(command, timeout)))
        registry.register("schedule_cron", "Schedule a five-field cron prompt", _schema({"expression": {"type": "string"}, "prompt": {"type": "string"}, "recurring": {"type": "boolean"}}, ["expression", "prompt"]), lambda expression, prompt, recurring=True: json.dumps(asdict(cron.schedule(expression, prompt, recurring))))
        registry.register("list_crons", "List cron jobs", _schema(), lambda: json.dumps([asdict(job) for job in cron.list()], indent=2))
        registry.register("cancel_cron", "Cancel a cron job", _schema({"job_id": {"type": "string"}}, ["job_id"]), cron.cancel)

        subagents = SubagentRunner(lambda agent_type: build_agent(f"subagent-{agent_type}", agent_type, True))
        registry.register("subagent", "Run a one-shot isolated subagent and return its summary", _schema({"prompt": {"type": "string"}, "agent_type": {"type": "string"}}, ["prompt"]), subagents.run)

        if teammate:
            registry.register("send_message", "Send a message to the lead", _schema({"content": {"type": "string"}}, ["content"]), lambda content: (mailbox.send(name, "lead", content), "Sent to lead")[1])
        else:
            registry.register("spawn_teammate", "Spawn an autonomous teammate", _schema({"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, ["name", "role", "prompt"]), team.spawn)
            registry.register("send_message", "Send a message to a teammate", _schema({"to": {"type": "string"}, "content": {"type": "string"}}, ["to", "content"]), team.send)
            registry.register("check_inbox", "Drain the lead teammate inbox", _schema(), lambda: json.dumps(team.check_inbox(), ensure_ascii=False, indent=2))
            registry.register("request_shutdown", "Request graceful teammate shutdown", _schema({"teammate": {"type": "string"}}, ["teammate"]), team.request_shutdown)
            registry.register("request_plan", "Ask a teammate for a plan", _schema({"teammate": {"type": "string"}, "task": {"type": "string"}}, ["teammate", "task"]), team.request_plan)
            registry.register("review_plan", "Approve or reject a teammate plan", _schema({"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, ["request_id", "approve"]), team.review_plan)
            registry.register("review_permission", "Approve or reject a teammate permission request", _schema({"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, ["request_id", "approve"]), team.review_permission)

        def state() -> dict[str, Any]:
            return {
                "workspace": str(settings.workspace),
                "tools": ", ".join(spec.name for spec in registry.specs()),
                "skills": skills.list_text(),
                "memory": memory.index_text(),
                "tasks": task_list_text(),
                "team": team.status(),
                "identity": f"{name} ({role})",
            }

        def teammate_approval(call: ToolCall) -> bool:
            return team.await_permission(
                name, call, timeout=settings.idle_timeout_seconds
            )

        return AgentRuntime(
            provider=provider,
            registry=registry,
            hooks=hooks,
            permissions=permissions,
            context=context,
            prompts=prompts,
            state_builder=state,
            background=background,
            cron=cron,
            approval_callback=teammate_approval if teammate else approval_callback,
            notification_sources=[] if teammate else [lead_inbox],
            max_rounds=settings.max_rounds,
            max_tokens=settings.max_tokens,
        )

    team.runtime_factory = lambda name, role: build_agent(name, role, True)
    runtime = build_agent("lead", "team lead")
    cron.start()
    return SimpleCCApp(settings, runtime, tasks, todos, memory, skills, background, cron, team)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple CC using SiliconFlow")
    parser.add_argument("--workspace", default=".", help="Workspace the agent may access")
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
        return True, ", ".join(sorted(active_teammates)) or "No active teammates."
    if command == "/memory":
        if not config_module.MEMORY_INDEX.exists():
            return True, "(no memories)"
        return True, config_module.MEMORY_INDEX.read_text()[:4000]
    return False, ""


def _approval_prompt(call: ToolCall) -> bool:
    if threading.current_thread() is not threading.main_thread():
        print(f"\n[permission deferred] Run this request in the foreground to approve: {call.name}")
        return False
    print(f"\nPermission required: {call.name} {json.dumps(call.arguments, ensure_ascii=False)}")
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
            agent_loop(history, context)
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
    history = []
    context = update_context({}, [])
    stop_event = getattr(app, "stop_event", threading.Event())
    threading.Thread(
        target=cron_autorun_loop,
        args=(history, context, stop_event),
        name="simple-cc-cron-autorun",
        daemon=True,
    ).start()
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
                agent_loop(history, context)
                context = update_context(context, history)
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
        stop_event.set()
        app.close()
        CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
