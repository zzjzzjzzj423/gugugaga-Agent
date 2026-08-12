from __future__ import annotations
import copy
import json
import threading
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
    build_user_content,
    compact_history,
    inject_background_notifications,
    prepare_context,
    reactive_compact,
    update_context,
)
from .cron import consume_cron_queue
from .hooks import HookEvent, HookManager
from .hooks import trigger_hooks
from .models import ChatProvider, ToolCall, ToolSpec
from .memory import MemoryStore
from .permissions import PermissionPolicy
from .prompts import PromptAssembler, assemble_system_prompt
from .provider import ContextLengthError
from .recovery import RecoveryState, is_prompt_too_long_error, with_retry
from .subagents import extract_text, has_tool_use
from .tools import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    ToolRegistry,
    call_tool_handler,
)


client = None
rounds_since_todo = 0
agent_lock = threading.Lock()


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
    ):
        self.provider = provider
        self.permissions = permissions or PermissionPolicy()
        self.approval_callback = approval_callback
        self.registry = FixedToolRegistry()
        self.messages: list[dict[str, Any]] = []
        self.context: dict[str, Any] = update_context({}, [])

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
        return {
            **update_context(self.context, self.messages),
            "workspace": str(config.WORKDIR),
            "tools": ", ".join(spec.name for spec in self.registry.specs()),
        }

    def run_turn(self, query: str) -> str:
        trigger_hooks("UserPromptSubmit", query)
        with agent_lock:
            turn_start = len(self.messages)
            self.messages.append({"role": "user", "content": query})
            agent_loop(
                self.messages,
                self.context,
                self.permissions,
                self.approval_callback,
            )
            self.context = update_context(self.context, self.messages)
            return self._turn_text(self.messages, turn_start)


def call_llm(
    messages: list,
    context: dict,
    tools: list,
    state: RecoveryState,
    max_tokens: int,
):
    if client is None:
        raise RuntimeError("Agent provider is not configured")
    system = assemble_system_prompt(context)
    return with_retry(
        lambda: client.create(
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
):
    global rounds_since_todo
    tools, handlers = TOOL_DEFINITIONS, TOOL_HANDLERS
    permissions = permissions or PermissionPolicy()
    state = RecoveryState()
    max_tokens = config.DEFAULT_MAX_TOKENS
    memory_store = None
    turn_prompt = ""
    memory_snapshot = copy.deepcopy(messages)
    relevant_memories = ""
    memory_ready = False

    if config.MEMORY_ENABLED and client is not None:
        memory_store = MemoryStore(
            config.MEMORY_DIR,
            provider=client,
            max_selected=config.MEMORY_MAX_SELECTED,
            max_injected_chars=config.MEMORY_MAX_INJECTED_CHARS,
            consolidate_threshold=config.MEMORY_CONSOLIDATE_THRESHOLD,
            consolidate_target=config.MEMORY_CONSOLIDATE_TARGET,
            consolidate_cooldown_seconds=(
                config.MEMORY_CONSOLIDATE_COOLDOWN_SECONDS
            ),
        )
        turn_prompt = memory_store.turn_prompt(messages)
    while True:
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

        if memory_store is not None and not memory_ready:
            # 这是压缩前快照，供本轮结束后的记忆提取使用。
            memory_snapshot = copy.deepcopy(messages)
            try:
                relevant_memories = memory_store.load_relevant(memory_snapshot)
            except Exception as error:
                memory_store._warn(f"load failed: {error}")
                relevant_memories = ""
            memory_ready = True

        prepare_context(messages)
        context = update_context(context, messages)

        try:
            # inject() 会深拷贝 messages，不会污染真实历史记录。
            request_messages = (
                memory_store.inject(
                    messages,
                    relevant_memories,
                    target_text=turn_prompt,
                )
                if memory_store is not None
                else messages
            )
            response = call_llm(
                request_messages,
                context,
                tools,
                state,
                max_tokens,
            )
        except Exception as error:
            if (
                is_prompt_too_long_error(error)
                and not state.has_attempted_reactive_compact
            ):
                messages[:] = reactive_compact(messages)
                state.has_attempted_reactive_compact = True
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"[Error] {type(error).__name__}: {error}"
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
        messages.append({"role": "assistant", "content": response.content})
        if not has_tool_use(response.content):
            trigger_hooks("Stop", messages)

            # 只有真人触发且正常完成的回合才提取记忆。
            # 错误、max_tokens、工具中间轮次、定时任务都不会进入这里保存。
            if (
                    memory_store is not None
                    and turn_prompt
                    and not turn_prompt.lstrip().startswith(
                (
                        "[Scheduled]",
                        "<reminder>",
                        "[Compacted]",
                        "[Reactive compact]",
                        "<task_notification>",
                        "<teammate-message>",
                )
            )
            ):
                memory_store.extract(
                    memory_snapshot,
                    extract_text(response.content),
                )
                memory_store.consolidate_if_needed()

            return

        results = []
        compacted_now = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            if block.name == "compact":
                messages[:] = compact_history(messages)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[Compacted. Continue with summarized context.]"
                        ),
                    }
                )
                compacted_now = True
                break

            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(blocked),
                    }
                )
                continue

            call = ToolCall(block.id, block.name, block.input)
            if not permissions.approve(call, approval_callback):
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            f"Permission denied for tool '{block.name}'. "
                            "Choose a safer approach."
                        ),
                    }
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
                continue

            handler = handlers.get(block.name)
            output = call_tool_handler(handler, block.input, block.name)
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

        if compacted_now:
            continue

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
            response = self.provider.create(
                system=(
                    "Summarize this coding-agent history. Preserve goals, "
                    "decisions, files changed, pending tasks, and errors."
                ),
                messages=[{"role": "user", "content": transcript}],
                tools=[],
                max_tokens=min(self.max_tokens, 2048),
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
        for _ in range(self.max_rounds):
            self._drain_notifications()
            self.messages = self.context.prepare(self.messages, self._summarize)
            system = self.prompts.build(self.state_builder())
            try:
                response = self.provider.create(
                    system=system,
                    messages=self.messages,
                    tools=self.registry.specs(),
                    max_tokens=self.max_tokens,
                )
            except ContextLengthError:
                if attempted_reactive_compact:
                    raise
                self.messages = self.context.compact(
                    self.messages, self._summarize(self.messages), force=True
                )
                attempted_reactive_compact = True
                continue

            response_text = self._response_text(response.content)
            response_calls = self._response_calls(response.content)
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
                hook_results = self.hooks.trigger(HookEvent.PRE_TOOL_USE, call=call)
                blocked = next(
                    (str(result) for result in hook_results if result not in (None, False, "")),
                    "",
                )
                if blocked:
                    output = blocked
                elif not self.permissions.approve(call, self.approval_callback):
                    output = f"Permission denied for tool '{call.name}'. Choose a safer approach."
                elif call.name == "compact":
                    summary = self._summarize(self.messages)
                    self.messages = self.context.compact(
                        self.messages,
                        summary,
                        force=True,
                    )
                    output = "Conversation compacted and transcript archived."
                else:
                    output = self.registry.execute(call.name, call.arguments)
                self.hooks.trigger(HookEvent.POST_TOOL_USE, call=call, output=output)
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
