from __future__ import annotations

import copy
import json
import threading
import time
from typing import Any, Callable

from . import config
from .background import (
    BackgroundManager,
    CronScheduler,
    should_run_background,
    start_background_task,
)
from .context import (
    ContextManager,
    MemoryStore,
    build_user_content,
    compact_history,
    inject_background_notifications,
    prepare_context,
    reactive_compact,
    update_context,
)
from .context_modes import (
    CompressionReason,
    ContextModeError,
    RequestContext,
    SessionContextConfig,
    SessionContextCoordinator,
    TokenCounterRegistry,
)
from .cron import consume_cron_queue
from .hooks import HookEvent, HookManager
from .hooks import trigger_hooks
from .models import ChatProvider, ToolCall, ToolSpec
from .memory import MemoryService
from .observability import (
    RecordingSystem,
    event_scope,
    notify,
    record_llm_call,
)
from .permissions import PermissionPolicy
from .prompts import PromptAssembler, assemble_system_prompt
from .provider import ContextLengthError
from .recovery import RecoveryState, is_prompt_too_long_error, with_retry
from .subagents import extract_text
from .tools import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    ToolRegistry,
    call_tool_handler,
)


client = None
rounds_since_todo = 0
agent_lock = threading.Lock()
_default_memory_states: dict[str, dict[str, Any]] = {}

SAVE_NOTE_DEFINITION = {
    "name": "save_note",
    "description": (
        "Persist one fact only when the user explicitly asks you to remember "
        "or save it for future conversations. Do not call for ordinary statements."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "minLength": 1, "maxLength": 120},
            "content": {"type": "string", "minLength": 1, "maxLength": 1000},
        },
        "required": ["subject", "content"],
        "additionalProperties": False,
    },
}


class FixedToolRegistry:
    """Read-only compatibility view over the fixed S01-S17 tables."""

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                definition["name"],
                definition["description"],
                definition["input_schema"],
            )
            for definition in TOOL_DEFINITIONS
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return str(
            call_tool_handler(TOOL_HANDLERS.get(name), arguments, name)
        )


class SourceRuntime:
    """Small public wrapper over the retained module-level S20 loop."""

    def __init__(
        self,
        provider: ChatProvider,
        permissions: PermissionPolicy | None = None,
        approval_callback: Callable[[ToolCall], bool] | None = None,
        context_coordinator: SessionContextCoordinator | None = None,
        recording: RecordingSystem | None = None,
        memory_service: MemoryService | None = None,
    ):
        self.provider = provider
        self.permissions = permissions or PermissionPolicy()
        self.approval_callback = approval_callback
        self.registry = FixedToolRegistry()
        self.messages: list[dict[str, Any]] = []
        self.context_coordinator = context_coordinator or SessionContextCoordinator(
            SessionContextConfig.parse(),
            summary_callback=_summary_callback(provider),
            workspace=config.WORKDIR,
            transcripts_dir=config.TRANSCRIPT_DIR,
        )
        self.recording = recording or RecordingSystem(
            self.context_coordinator.workspace / ".gugugaga"
        )
        database = (
            self.recording.chat_log.path
            if self.recording.chat_log is not None
            else self.context_coordinator.workspace / ".gugugaga" / "state.db"
        )
        self.memory_service = memory_service or MemoryService(database, provider)
        self.context: dict[str, Any] = update_context(
            {}, [], self.context_coordinator.memory_dir / "MEMORY.md"
        )
        self.memory_state: dict[str, Any] = {"pending_turns": []}

    @staticmethod
    def _turn_text(messages: list[dict[str, Any]], turn_start: int) -> str:
        texts = []
        for message in messages[turn_start:]:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                block_type = (
                    block.get("type")
                    if isinstance(block, dict)
                    else getattr(block, "type", None)
                )
                if block_type == "text":
                    texts.append(
                        block.get("text", "")
                        if isinstance(block, dict)
                        else block.text
                    )
        return "\n".join(texts)

    def state_builder(self) -> dict[str, Any]:
        state = update_context(
            self.context,
            self.messages,
            self.context_coordinator.memory_dir / "MEMORY.md",
        )
        query = ""
        for message in reversed(self.messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                query = message["content"]
                break
        recalled = self.memory_service.recall(query)
        if recalled:
            state["memories"] = "\n\n".join(
                value for value in (state.get("memories", ""), recalled) if value
            )
        return {
            **state,
            "workspace": str(self.context_coordinator.workspace),
            "tools": ", ".join(spec.name for spec in self.registry.specs()),
        }

    def run_turn(self, query: str, *, source: str = "cli") -> str:
        turn = self.recording.start_turn(
            session_id=self.context_coordinator.session_id,
            user_message=query,
            source=source,
        )
        with turn:
            trigger_hooks("UserPromptSubmit", query)
            with agent_lock:
                turn_start = len(self.messages)
                self.messages.append({"role": "user", "content": query})
                recalled = self.memory_service.recall(query)
                if recalled:
                    legacy = self.context.get("memories", "")
                    self.context["memories"] = "\n\n".join(
                        value for value in (legacy, recalled) if value
                    )
                agent_loop(
                    self.messages,
                    self.context,
                    self.permissions,
                    self.approval_callback,
                    self.memory_state,
                    self.context_coordinator,
                    self.memory_service,
                    turn.turn_id,
                    query,
                )
                self.context = update_context(
                    self.context,
                    self.messages,
                    self.context_coordinator.memory_dir / "MEMORY.md",
                )
                recalled = self.memory_service.recall(query)
                if recalled:
                    legacy = self.context.get("memories", "")
                    self.context["memories"] = "\n\n".join(
                        value for value in (legacy, recalled) if value
                    )
                reply = self._turn_text(self.messages, turn_start)
            turn.finish(
                reply,
                meta={"context": self.context_coordinator.status()},
            )
            self.memory_service.on_exchange_completed(turn_id=turn.turn_id)
            return reply

    def context_status(self) -> dict[str, Any]:
        return self.context_coordinator.status()

    def _replace_session(
        self,
        *,
        session_id: str | None,
        messages: list[dict[str, Any]],
        context_mode: str | None = None,
    ) -> str:
        previous = self.context_coordinator
        previous.close()
        self.context_coordinator = SessionContextCoordinator(
            previous.config,
            counter_registry=previous.registry,
            summary_callback=previous.summary_callback,
            workspace=previous.workspace,
            transcripts_dir=previous.transcripts_dir,
            memory_dir=previous.memory_dir,
            tool_results_dir=previous.tool_results_dir,
            session_id=session_id,
        )
        if context_mode is not None:
            self.context_coordinator.set_mode(context_mode)
        self.messages = copy.deepcopy(messages)
        if self.messages:
            self.context_coordinator.observe_history(self.messages)
        self.context = update_context(
            {}, self.messages, self.context_coordinator.memory_dir / "MEMORY.md"
        )
        self.memory_state = {"pending_turns": []}
        return self.context_coordinator.session_id

    def start_new_session(self, context_mode: str | None = None) -> str:
        """Reset transient conversation state while retaining tools and long-term memory."""
        with agent_lock:
            return self._replace_session(
                session_id=None, messages=[], context_mode=context_mode
            )

    def resume_session(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        context_mode: str | None = None,
    ) -> str:
        """Restore one persisted conversation as the active runtime session."""
        value = session_id.strip()
        if not value:
            raise ValueError("session_id is required")
        if not messages:
            raise ValueError("conversation has no messages")
        if any(message.get("role") not in {"user", "assistant"} for message in messages):
            raise ValueError("conversation contains an invalid role")
        with agent_lock:
            return self._replace_session(
                session_id=value,
                messages=messages,
                context_mode=context_mode,
            )

    def restore_session_state(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        context_mode: str | None = None,
    ) -> str:
        """Restore an active session after an in-process configuration reload."""
        value = session_id.strip()
        if not value:
            raise ValueError("session_id is required")
        if any(message.get("role") not in {"user", "assistant"} for message in messages):
            raise ValueError("conversation contains an invalid role")
        with agent_lock:
            return self._replace_session(
                session_id=value,
                messages=messages,
                context_mode=context_mode,
            )


def _summary_callback(provider: ChatProvider):
    def summarize(system: str, prompt: str, max_tokens: int) -> str:
        response = record_llm_call(
            provider,
            model=config.MODEL or None,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            max_tokens=max_tokens,
            call_type="context_summary",
        )
        return extract_text(response.content)

    return summarize


def _workspace_handlers(
    handlers: dict[str, Callable], workspace
) -> dict[str, Callable]:
    bound = dict(handlers)
    for name in ("bash", "read_file", "write_file", "edit_file", "glob"):
        handler = handlers.get(name)
        if handler is None:
            continue

        def run_in_workspace(_handler=handler, **arguments):
            return _handler(cwd=workspace, **arguments)

        bound[name] = run_in_workspace
    return bound


def _observable_tool_args(name: str, arguments: Any) -> Any:
    if name == "save_note":
        return {"subject": "[REDACTED]", "content": "[REDACTED]"}
    return arguments


def call_llm(
    messages: list,
    context: dict,
    tools: list,
    state: RecoveryState,
    max_tokens: int,
    *,
    system: str | None = None,
):
    if client is None:
        raise RuntimeError("Agent provider is not configured")
    system = system if system is not None else assemble_system_prompt(context)
    return with_retry(
        lambda: record_llm_call(
            client,
            model=state.current_model or None,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        ),
        state,
    )


def agent_loop(
    messages: list,
    context: dict,
    permissions: PermissionPolicy | None = None,
    approval_callback: Callable[[ToolCall], bool] | None = None,
    memory_state: dict[str, Any] | None = None,
    context_coordinator: SessionContextCoordinator | None = None,
    memory_service: MemoryService | None = None,
    turn_id: str | None = None,
    memory_query: str = "",
):
    global rounds_since_todo
    tools = (
        [*TOOL_DEFINITIONS, SAVE_NOTE_DEFINITION]
        if memory_service is not None
        else TOOL_DEFINITIONS
    )
    handlers = TOOL_HANDLERS
    permissions = permissions or PermissionPolicy()
    state = RecoveryState()
    max_tokens = config.DEFAULT_MAX_TOKENS
    if memory_state is None:
        memory_key = (
            context_coordinator.memory_dir
            if context_coordinator is not None
            else config.MEMORY_DIR
        )
        memory_state = _default_memory_states.setdefault(
            str(memory_key.resolve()), {"pending_turns": []}
        )
    if context_coordinator is None:
        context_coordinator = memory_state.get("context_coordinator")
    if context_coordinator is None:
        context_coordinator = SessionContextCoordinator(
            SessionContextConfig.parse(),
            counter_registry=TokenCounterRegistry(),
            summary_callback=_summary_callback(client),
            workspace=config.WORKDIR,
            transcripts_dir=config.TRANSCRIPT_DIR,
        )
        memory_state["context_coordinator"] = context_coordinator
    handlers = _workspace_handlers(
        handlers, context_coordinator.workspace
    )
    memory_store = (
        MemoryStore(context_coordinator.memory_dir, provider=client)
        if memory_service is None
        else None
    )

    iteration = 0
    while True:
        iteration += 1
        fired = consume_cron_queue()
        for job in fired:
            messages.append(
                {"role": "user", "content": f"[Scheduled] {job.prompt}"}
            )
            print(f"  \033[35m[cron inject] {job.prompt[:60]}\033[0m")

        inject_background_notifications(messages)

        if rounds_since_todo >= 3:
            messages.append(
                {
                    "role": "user",
                    "content": "<reminder>Update your todos.</reminder>",
                }
            )
            rounds_since_todo = 0

        context_coordinator.observe_history(messages)
        context = update_context(
            context,
            messages,
            context_coordinator.memory_dir / "MEMORY.md",
        )
        if memory_service is not None:
            recalled = memory_service.recall(memory_query)
            if recalled:
                legacy = context.get("memories", "")
                context["memories"] = "\n\n".join(
                    value for value in (legacy, recalled) if value
                )
        system = assemble_system_prompt(context)
        request_context = RequestContext(system=system, tools=tools)

        try:
            provider_messages = context_coordinator.prepare_request(
                messages, request_context
            )
        except ContextModeError as error:
            if not state.has_attempted_reactive_compact:
                provider_messages = context_coordinator.reactive_recover(
                    messages,
                    request_context,
                    reason=CompressionReason.STRATEGY_FAILURE_RECOVERY,
                )
                state.has_attempted_reactive_compact = True
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"[{error.code}] {error.safe_message} "
                                f"History preserved: {error.history_preserved}. "
                                f"Next: {error.suggested_action}",
                            }
                        ],
                    }
                )
                return

        try:
            with event_scope(iteration=iteration):
                response = call_llm(
                    provider_messages,
                    context,
                    tools,
                    state,
                    max_tokens,
                    system=system,
                )
        except Exception as error:
            if (
                is_prompt_too_long_error(error)
                and not state.has_attempted_reactive_compact
            ):
                context_coordinator.reactive_recover(
                    messages,
                    request_context,
                    reason=CompressionReason.PROVIDER_OVERFLOW,
                )
                state.has_attempted_reactive_compact = True
                continue
            if is_prompt_too_long_error(error):
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[CONTEXT_RECOVERY_EXHAUSTED] The Provider "
                                    "still rejected the rebuilt context. History "
                                    "preserved: True. Next: start a new session or "
                                    "use a model with a larger context window."
                                ),
                            }
                        ],
                    }
                )
                return
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "[PROVIDER_FAILED] The Provider request failed. "
                                "History preserved: True. Next: retry the turn "
                                "or inspect protected local diagnostics."
                            ),
                        }
                    ],
                }
            )
            return

        if response.stop_reason == "max_tokens":
            if not state.has_escalated:
                max_tokens = config.ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(
                    f"  \033[33m[max_tokens] retry with {max_tokens}\033[0m"
                )
                continue
            messages.append(
                {"role": "assistant", "content": response.content}
            )
            if state.recovery_count < config.MAX_RECOVERY_RETRIES:
                messages.append(
                    {"role": "user", "content": config.CONTINUATION_PROMPT}
                )
                state.recovery_count += 1
                continue
            return

        max_tokens = config.DEFAULT_MAX_TOKENS
        state.has_escalated = False
        compact_blocks = [
            block
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "compact"
        ]
        if response.stop_reason == "tool_use" and compact_blocks:
            compact_started = time.monotonic()
            compact_result = context_coordinator.manual_compact(
                messages, request_context
            )
            messages.append({"role": "assistant", "content": response.content})
            compact_ids = {block.id for block in compact_blocks}
            results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.id in compact_ids:
                    output = (
                        f"{compact_result.status}: {compact_result.code}. "
                        f"{compact_result.message}"
                    )
                else:
                    output = (
                        "Tool not executed because the same tool group requested "
                        "context compaction. Retry it after compaction."
                    )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
                notify(
                    "tool",
                    {
                        "iteration": iteration,
                        "tool": block.name,
                        "args": _observable_tool_args(block.name, block.input),
                        "output": output,
                        "status": "ok",
                        "latency_ms": round(
                            (time.monotonic() - compact_started) * 1000
                        ),
                    },
                )
            messages.append({"role": "user", "content": results})
            continue

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            trigger_hooks("Stop", messages)
            if memory_service is None:
                pending_turns = memory_state.setdefault("pending_turns", [])
                pending_turns.append(
                    (
                        copy.deepcopy(messages),
                        extract_text(response.content),
                    )
                )
                if len(pending_turns) >= config.MEMORY_EXTRACTION_INTERVAL:
                    batch = pending_turns[: config.MEMORY_EXTRACTION_INTERVAL]
                    del pending_turns[: config.MEMORY_EXTRACTION_INTERVAL]
                    memory_store.extract_batch(batch)
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")
            tool_started = time.monotonic()

            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                output = str(blocked)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
                notify(
                    "tool",
                    {
                        "iteration": iteration,
                        "tool": block.name,
                        "args": _observable_tool_args(block.name, block.input),
                        "output": output,
                        "status": "blocked",
                        "latency_ms": round(
                            (time.monotonic() - tool_started) * 1000
                        ),
                    },
                )
                continue

            call = ToolCall(block.id, block.name, block.input)
            if not permissions.approve(call, approval_callback):
                output = (
                    f"Permission denied for tool '{block.name}'. "
                    "Choose a safer approach."
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
                notify(
                    "tool",
                    {
                        "iteration": iteration,
                        "tool": block.name,
                        "args": _observable_tool_args(block.name, block.input),
                        "output": output,
                        "status": "denied",
                        "latency_ms": round(
                            (time.monotonic() - tool_started) * 1000
                        ),
                    },
                )
                continue

            if should_run_background(block.name, block.input):
                background_id = start_background_task(block, handlers)
                output = (
                    f"[Background task {background_id} started] "
                    "Result will arrive as a task_notification."
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
                notify(
                    "tool",
                    {
                        "iteration": iteration,
                        "tool": block.name,
                        "args": _observable_tool_args(block.name, block.input),
                        "output": output,
                        "status": "background",
                        "latency_ms": round(
                            (time.monotonic() - tool_started) * 1000
                        ),
                    },
                )
                continue

            handler = handlers.get(block.name)
            try:
                with event_scope(iteration=iteration):
                    if block.name == "save_note" and memory_service is not None:
                        arguments = block.input if isinstance(block.input, dict) else {}
                        output = memory_service.save_note(
                            subject=arguments.get("subject"),
                            content=arguments.get("content"),
                            turn_id=turn_id,
                        ).to_json()
                    else:
                        output = call_tool_handler(handler, block.input, block.name)
            except Exception as error:
                notify(
                    "tool",
                    {
                        "iteration": iteration,
                        "tool": block.name,
                        "args": _observable_tool_args(block.name, block.input),
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "latency_ms": round(
                            (time.monotonic() - tool_started) * 1000
                        ),
                    },
                )
                raise
            trigger_hooks("PostToolUse", block, output)
            print(str(output)[:300])

            if block.name == "todo_write":
                rounds_since_todo = 0
            else:
                rounds_since_todo += 1

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                }
            )
            notify(
                "tool",
                {
                    "iteration": iteration,
                    "tool": block.name,
                    "args": _observable_tool_args(block.name, block.input),
                    "output": output,
                    "status": "ok",
                    "latency_ms": round(
                        (time.monotonic() - tool_started) * 1000
                    ),
                },
            )

        messages.append(
            {"role": "user", "content": build_user_content(results)}
        )


class AgentRuntime:
    def __init__(
        self,
        *,
        provider: ChatProvider,
        registry: ToolRegistry,
        hooks: HookManager,
        permissions: PermissionPolicy,
        context: ContextManager,
        prompts: PromptAssembler,
        state_builder: Callable[[], dict[str, Any]],
        background: BackgroundManager,
        cron: CronScheduler,
        approval_callback: Callable[[ToolCall], bool] | None = None,
        notification_sources: list[Callable[[], list[str]]] | None = None,
        max_rounds: int = 40,
        max_tokens: int = 8192,
        context_coordinator: SessionContextCoordinator | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.hooks = hooks
        self.permissions = permissions
        self.context = context
        self.prompts = prompts
        self.state_builder = state_builder
        self.background = background
        self.cron = cron
        self.approval_callback = approval_callback
        self.notification_sources = notification_sources or []
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []
        self._run_lock = threading.RLock()
        try:
            workspace = self.context.transcripts_dir.resolve().parents[1]
        except (AttributeError, IndexError):
            workspace = config.WORKDIR
        self.context_coordinator = context_coordinator or SessionContextCoordinator(
            SessionContextConfig.parse(),
            summary_callback=_summary_callback(provider),
            workspace=workspace,
            transcripts_dir=getattr(
                self.context, "transcripts_dir", config.TRANSCRIPT_DIR
            ),
        )

    def _drain_notifications(self) -> bool:
        self.cron.fire_due()
        notifications = [*self.cron.drain(), *self.background.drain()]
        for source in self.notification_sources:
            notifications.extend(source())
        if notifications:
            self.messages.append({
                "role": "user",
                "content": "<notifications>\n" + "\n".join(notifications) + "\n</notifications>",
            })
            return True
        return False

    def _summarize(self, messages: list[dict[str, Any]]) -> str:
        transcript = json.dumps(messages, ensure_ascii=False, default=str)
        if len(transcript) > 30_000:
            transcript = transcript[:15_000] + "\n...[middle omitted]...\n" + transcript[-15_000:]
        try:
            response = record_llm_call(
                self.provider,
                system=(
                    "Summarize this coding-agent history. Preserve goals, "
                    "decisions, files changed, pending tasks, and errors."
                ),
                messages=[{"role": "user", "content": transcript}],
                tools=[],
                max_tokens=min(self.max_tokens, 2048),
                call_type="context_summary",
            )
            return self._response_text(response.content) or (
                "Earlier conversation archived."
            )
        except Exception:
            return "Earlier conversation archived after summary generation failed."

    @staticmethod
    def _response_text(content) -> str:
        return "".join(
            getattr(block, "text", "")
            for block in content
            if getattr(block, "type", None) == "text"
        )

    @staticmethod
    def _response_calls(content) -> list[ToolCall]:
        return [
            ToolCall(block.id, block.name, block.input)
            for block in content
            if getattr(block, "type", None) == "tool_use"
        ]

    @staticmethod
    def _content_blocks(content) -> list[dict[str, Any]]:
        blocks = []
        for block in content:
            if isinstance(block, dict):
                blocks.append(dict(block))
            elif getattr(block, "type", None) == "text":
                blocks.append({"type": "text", "text": block.text})
            elif getattr(block, "type", None) == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        return blocks

    def run_turn(self, query: str) -> str:
        with self._run_lock:
            self.hooks.trigger(HookEvent.USER_PROMPT_SUBMIT, query=query)
            self.messages.append({"role": "user", "content": query})
            return self.run_messages()

    def run_pending(self) -> str | None:
        with self._run_lock:
            if not self._drain_notifications():
                return None
            return self.run_messages()

    def run_messages(self) -> str:
        attempted_reactive_compact = False
        for iteration in range(1, self.max_rounds + 1):
            self._drain_notifications()
            system = self.prompts.build(self.state_builder())
            request_context = RequestContext(
                system=system, tools=self.registry.specs()
            )
            try:
                provider_messages = self.context_coordinator.prepare_request(
                    self.messages, request_context
                )
            except ContextModeError:
                if attempted_reactive_compact:
                    raise
                provider_messages = self.context_coordinator.reactive_recover(
                    self.messages,
                    request_context,
                    reason=CompressionReason.STRATEGY_FAILURE_RECOVERY,
                )
                attempted_reactive_compact = True
            try:
                with event_scope(iteration=iteration):
                    response = record_llm_call(
                        self.provider,
                        system=system,
                        messages=provider_messages,
                        tools=self.registry.specs(),
                        max_tokens=self.max_tokens,
                    )
            except ContextLengthError:
                if attempted_reactive_compact:
                    raise ContextModeError(
                        "CONTEXT_RECOVERY_EXHAUSTED",
                        "The Provider still rejected the rebuilt context.",
                        suggested_action=(
                            "Start a new session or use a model with a larger context window."
                        ),
                    )
                self.context_coordinator.reactive_recover(
                    self.messages,
                    request_context,
                    reason=CompressionReason.PROVIDER_OVERFLOW,
                )
                attempted_reactive_compact = True
                continue

            response_text = self._response_text(response.content)
            response_calls = self._response_calls(response.content)
            compact_calls = [
                call for call in response_calls if call.name == "compact"
            ]
            if compact_calls:
                compact_result = self.context_coordinator.manual_compact(
                    self.messages, request_context
                )
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self._content_blocks(response.content),
                    }
                )
                compact_ids = {call.id for call in compact_calls}
                results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": (
                            f"{compact_result.status}: {compact_result.code}. "
                            f"{compact_result.message}"
                            if call.id in compact_ids
                            else "Tool not executed because context compaction was requested."
                        ),
                    }
                    for call in response_calls
                ]
                self.messages.append({"role": "user", "content": results})
                continue
            self.messages.append(
                {
                    "role": "assistant",
                    "content": self._content_blocks(response.content),
                }
            )
            if not response_calls:
                self.hooks.trigger(HookEvent.STOP, messages=self.messages)
                return response_text

            results = []
            for call in response_calls:
                tool_started = time.monotonic()
                hook_results = self.hooks.trigger(HookEvent.PRE_TOOL_USE, call=call)
                blocked = next(
                    (str(result) for result in hook_results if result not in (None, False, "")),
                    "",
                )
                if blocked:
                    output = blocked
                    tool_status = "blocked"
                elif not self.permissions.approve(call, self.approval_callback):
                    output = f"Permission denied for tool '{call.name}'. Choose a safer approach."
                    tool_status = "denied"
                else:
                    output = self.registry.execute(call.name, call.arguments)
                    tool_status = "ok"
                self.hooks.trigger(HookEvent.POST_TOOL_USE, call=call, output=output)
                notify(
                    "tool",
                    {
                        "iteration": iteration,
                        "tool": call.name,
                        "args": _observable_tool_args(call.name, call.arguments),
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
                        "tool_use_id": call.id,
                        "content": output,
                    }
                )
            self.messages.append({"role": "user", "content": results})
        raise RuntimeError(f"agent exceeded maximum rounds ({self.max_rounds})")


class SubagentRunner:
    def __init__(self, runtime_factory: Callable[[str], AgentRuntime]):
        self.runtime_factory = runtime_factory

    def run(self, prompt: str, agent_type: str = "general-purpose") -> str:
        runtime = self.runtime_factory(agent_type)
        return runtime.run_turn(prompt)
