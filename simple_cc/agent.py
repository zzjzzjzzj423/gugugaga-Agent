from __future__ import annotations

import json
import threading
from typing import Any, Callable

from .background import BackgroundManager, CronScheduler
from .context import ContextManager
from .hooks import HookEvent, HookManager
from .models import ChatProvider, ToolCall
from .permissions import PermissionPolicy
from .prompts import PromptAssembler
from .provider import ContextLengthError
from .tools import ToolRegistry


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
            response = self.provider.complete(
                "Summarize this coding-agent history. Preserve goals, decisions, files changed, pending tasks, and errors.",
                [{"role": "user", "content": transcript}],
                [],
                min(self.max_tokens, 2048),
            )
            return response.content or "Earlier conversation archived."
        except Exception:
            return "Earlier conversation archived after summary generation failed."

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
                response = self.provider.complete(
                    system, self.messages, self.registry.specs(), self.max_tokens
                )
            except ContextLengthError:
                if attempted_reactive_compact:
                    raise
                self.messages = self.context.compact(
                    self.messages, self._summarize(self.messages), force=True
                )
                attempted_reactive_compact = True
                continue

            self.messages.append(response.as_assistant_message())
            if not response.tool_calls:
                self.hooks.trigger(HookEvent.STOP, messages=self.messages)
                return response.content

            for call in response.tool_calls:
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
                    self.messages = self.context.compact(
                        self.messages,
                        "Conversation compacted by explicit tool request.",
                        force=True,
                    )
                    output = "Conversation compacted and transcript archived."
                else:
                    output = self.registry.execute(call.name, call.arguments)
                self.hooks.trigger(HookEvent.POST_TOOL_USE, call=call, output=output)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": output,
                })
        raise RuntimeError(f"agent exceeded maximum rounds ({self.max_rounds})")


class SubagentRunner:
    def __init__(self, runtime_factory: Callable[[str], AgentRuntime]):
        self.runtime_factory = runtime_factory

    def run(self, prompt: str, agent_type: str = "general-purpose") -> str:
        runtime = self.runtime_factory(agent_type)
        return runtime.run_turn(prompt)
