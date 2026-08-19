from __future__ import annotations
import copy
import json
import threading
import time
import uuid
from dataclasses import dataclass
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
from .permissions import PermissionDecision, PermissionPolicy
from .prompts import (
    PromptAssembler,
    assemble_system_prompt,
    ordinary_system_prompt,
    research_execution_prompt,
)
from .provider import ContextLengthError
from .recovery import RecoveryState, is_prompt_too_long_error, with_retry
from .subagents import extract_text, has_tool_use
from .evidence import (
    CutoffMismatch,
    evidence_record_from_result,
    link_final_answer_sources,
    prepare_research_arguments,
    record_research_evidence,
    registered_source_map,
)
from .research_models import (
    EvidenceRegistry,
    ResearchPlan,
    ResearchRank,
    TaskKind,
    normalize_task_kind,
)
from .research_workflow import ResearchWorkflow
from .telemetry import ToolCapture, TracingProvider, bind_tool_capture
from .trace import RunContext, TraceRecorder, TraceWriteError, bind_run_context
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

    def __init__(
        self,
        definitions: list[dict[str, Any]] | None = None,
        handlers: dict[str, Callable] | None = None,
    ):
        self.definitions = definitions if definitions is not None else TOOL_DEFINITIONS
        self.handlers = handlers if handlers is not None else TOOL_HANDLERS

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                definition["name"],
                definition["description"],
                definition["input_schema"],
            )
            for definition in self.definitions
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return str(
            call_tool_handler(self.handlers.get(name), arguments, name)
        )


@dataclass(frozen=True)
class AgentLoopOutcome:
    status: str
    final_text: str
    failure_class: str | None = None
    failure_message: str | None = None
    rounds_used: int = 0


class SourceRuntime:
    """Small public wrapper over the retained module-level S20 loop."""

    def __init__(
        self,
        provider: ChatProvider,
        permissions: PermissionPolicy | None = None,
        approval_callback: Callable[[ToolCall], bool] | None = None,
        *,
        recorder: TraceRecorder | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Callable] | None = None,
        max_rounds: int = 40,
        memory_enabled: bool | None = None,
    ):
        self.provider = provider
        self.tracing_provider = TracingProvider(provider)
        self.permissions = permissions or PermissionPolicy()
        self.approval_callback = approval_callback
        self.recorder = recorder
        self.tool_definitions = (
            tool_definitions if tool_definitions is not None else TOOL_DEFINITIONS
        )
        self.tool_handlers = (
            tool_handlers if tool_handlers is not None else TOOL_HANDLERS
        )
        self.max_rounds = max_rounds
        self.memory_enabled = (
            config.MEMORY_ENABLED if memory_enabled is None else memory_enabled
        )
        self.registry = FixedToolRegistry(self.tool_definitions, self.tool_handlers)
        self.messages: list[dict[str, Any]] = []
        self.context: dict[str, Any] = update_context({}, [])
        self.last_outcome: AgentLoopOutcome | None = None
        self.registered_sources: dict[str, str] = {}
        self.todo_state = {"rounds_since_todo": 0}

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

    def run_turn(
        self,
        query: str,
        *,
        task_id: str | None = None,
        cutoff: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata = dict(run_metadata or {})
        raw_task_type = metadata.get("task_type")
        task_kind = normalize_task_kind(raw_task_type)
        normalized_raw_type = str(raw_task_type or "").strip().lower()
        routing_reason = (
            "explicit"
            if normalized_raw_type in {"normal", "research", "research_analysis"}
            else "default"
        )
        trigger_hooks("UserPromptSubmit", query)
        with agent_lock:
            turn_start = len(self.messages)
            self.messages.append({"role": "user", "content": query})
            run = (
                RunContext(
                    self.recorder,
                    self.recorder.run_id,
                    task_id,
                    cutoff,
                )
                if self.recorder is not None and task_id is not None
                else None
            )

            def execute_routed_turn() -> tuple[
                AgentLoopOutcome, dict[str, str]
            ]:
                if run is not None:
                    run.recorder.record(
                        "task_routed",
                        {
                            "raw_task_type": raw_task_type,
                            "normalized_task_kind": task_kind.value,
                            "reason": routing_reason,
                        },
                        parent_span_id=run.parent_span_id,
                        agent_id=run.agent_id,
                    )

                if task_kind is TaskKind.NORMAL:
                    outcome = agent_loop(
                        self.messages,
                        self.context,
                        self.permissions,
                        self.approval_callback,
                        provider=self.tracing_provider,
                        tools=self.tool_definitions,
                        handlers=self.tool_handlers,
                        max_rounds=self.max_rounds,
                        memory_enabled=self.memory_enabled,
                        run_context=run,
                        registered_sources={},
                        todo_state=self.todo_state,
                        system_prompt=ordinary_system_prompt(self.state_builder()),
                    )
                    return outcome, {}

                evidence_registry = EvidenceRegistry()
                research_sources: dict[str, str] = {}

                def execute_research(
                    prompt: str,
                    max_rounds: int,
                    registry: EvidenceRegistry,
                ) -> AgentLoopOutcome:
                    stage = json.loads(prompt)
                    plan = ResearchPlan(
                        ResearchRank(stage["rank"]),
                        tuple(stage["directions"]),
                        "research execution",
                    )
                    gaps = tuple(stage.get("research_gaps") or ())
                    system_prompt = research_execution_prompt(
                        self.state_builder(),
                        question=str(stage.get("question") or query),
                        cutoff=stage.get("cutoff", cutoff),
                        plan=plan,
                        gaps=gaps,
                        remaining_rounds=max_rounds,
                    )
                    research_messages = [{"role": "user", "content": prompt}]
                    return agent_loop(
                        research_messages,
                        self.context,
                        self.permissions,
                        self.approval_callback,
                        provider=self.tracing_provider,
                        tools=self.tool_definitions,
                        handlers=self.tool_handlers,
                        max_rounds=max_rounds,
                        memory_enabled=self.memory_enabled,
                        run_context=run,
                        registered_sources=research_sources,
                        todo_state=self.todo_state,
                        system_prompt=system_prompt,
                        evidence_registry=registry,
                        finalize_user_turn=False,
                    )

                try:
                    result = ResearchWorkflow(
                        self.tracing_provider,
                        execute_research,
                        run_context=run,
                    ).run(query, cutoff, registry=evidence_registry)
                except TraceWriteError:
                    raise
                except Exception as error:
                    return AgentLoopOutcome(
                        "failed",
                        "",
                        getattr(error, "failure_class", None)
                        or type(error).__name__,
                        getattr(error, "failure_message", None) or str(error),
                    ), registered_source_map(evidence_registry)

                outcome = AgentLoopOutcome(
                    "completed",
                    result.final_text,
                    rounds_used=result.research_rounds_used,
                )
                self.messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": result.final_text}],
                })
                trigger_hooks("Stop", self.messages)
                return outcome, registered_source_map(evidence_registry)

            if run is not None:
                with bind_run_context(run):
                    outcome, final_source_map = execute_routed_turn()
                if outcome.status == "completed":
                    linkage = link_final_answer_sources(
                        outcome.final_text, final_source_map
                    )
                    run.recorder.record(
                        "final_answer",
                        {"text": outcome.final_text, **linkage},
                        parent_span_id=run.parent_span_id,
                        agent_id=run.agent_id,
                    )
                else:
                    run.recorder.record(
                        "run_failed",
                        {
                            "status": outcome.status,
                            "failure_class": outcome.failure_class,
                            "message": outcome.failure_message,
                        },
                        parent_span_id=run.parent_span_id,
                        agent_id=run.agent_id,
                    )
            else:
                outcome, _ = execute_routed_turn()
            self.last_outcome = outcome
            self.context = update_context(self.context, self.messages)
            return self._turn_text(self.messages, turn_start)


def call_llm(
    messages: list,
    context: dict,
    tools: list,
    state: RecoveryState,
    max_tokens: int,
    provider: ChatProvider | None = None,
    system_prompt: str | None = None,
):
    selected_provider = provider or client
    if selected_provider is None:
        raise RuntimeError("Agent provider is not configured")
    system = system_prompt if system_prompt is not None else assemble_system_prompt(context)
    return with_retry(
        lambda: selected_provider.create(
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
    *,
    provider: ChatProvider | None = None,
    tools: list[dict[str, Any]] | None = None,
    handlers: dict[str, Callable] | None = None,
    max_rounds: int = 40,
    memory_enabled: bool | None = None,
    run_context: RunContext | None = None,
    registered_sources: dict[str, str] | None = None,
    todo_state: dict[str, int] | None = None,
    system_prompt: str | None = None,
    evidence_registry: EvidenceRegistry | None = None,
    finalize_user_turn: bool = True,
) -> AgentLoopOutcome:
    global rounds_since_todo
    tools = tools if tools is not None else TOOL_DEFINITIONS
    handlers = handlers if handlers is not None else TOOL_HANDLERS
    permissions = permissions or PermissionPolicy()
    selected_provider = provider or client
    memory_enabled = (
        config.MEMORY_ENABLED if memory_enabled is None else memory_enabled
    )
    registered_sources = registered_sources if registered_sources is not None else {}
    required_cutoff = (
        run_context.cutoff
        if run_context is not None and evidence_registry is not None
        else None
    )

    def record_compaction(report) -> None:
        if run_context is None:
            return
        run_context.recorder.record(
            "context_compaction",
            {
                "method": report.method,
                "original_message_count": report.original_message_count,
                "retained_message_count": report.retained_message_count,
                "original_chars": report.original_chars,
                "retained_chars": report.retained_chars,
                "transcript_path": report.transcript_path,
            },
            agent_id=run_context.agent_id,
        )
    state = RecoveryState()
    max_tokens = config.DEFAULT_MAX_TOKENS
    memory_store = None
    turn_prompt = ""
    memory_snapshot = copy.deepcopy(messages)
    relevant_memories = ""
    memory_ready = False

    if memory_enabled and selected_provider is not None:
        memory_store = MemoryStore(
            config.MEMORY_DIR,
            provider=selected_provider,
            max_selected=config.MEMORY_MAX_SELECTED,
            max_injected_chars=config.MEMORY_MAX_INJECTED_CHARS,
            consolidate_threshold=config.MEMORY_CONSOLIDATE_THRESHOLD,
            consolidate_target=config.MEMORY_CONSOLIDATE_TARGET,
            consolidate_cooldown_seconds=(
                config.MEMORY_CONSOLIDATE_COOLDOWN_SECONDS
            ),
        )
        turn_prompt = memory_store.turn_prompt(messages)
    for _round_index in range(max_rounds):
        fired = consume_cron_queue()
        for job in fired:
            messages.append(
                {"role": "user", "content": f"[Scheduled] {job.prompt}"}
            )
            print(f"  \033[35m[cron inject] {job.prompt[:60]}\033[0m")

        inject_background_notifications(messages)

        todo_rounds = (
            todo_state["rounds_since_todo"]
            if todo_state is not None
            else rounds_since_todo
        )
        if todo_rounds >= 3:
            messages.append(
                {
                    "role": "user",
                    "content": "<reminder>Update your todos.</reminder>",
                }
            )
            if todo_state is not None:
                todo_state["rounds_since_todo"] = 0
            else:
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

        prepare_context(messages, on_compaction=record_compaction)
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
                selected_provider,
                system_prompt,
            )
        except Exception as error:
            if (
                is_prompt_too_long_error(error)
                and not state.has_attempted_reactive_compact
            ):
                messages[:] = reactive_compact(
                    messages, on_compaction=record_compaction
                )
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
            return AgentLoopOutcome(
                "failed", "", type(error).__name__, str(error), _round_index + 1
            )

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
            return AgentLoopOutcome(
                "failed",
                extract_text(response.content),
                "max_tokens",
                "model exhausted maximum-token recovery",
                _round_index + 1,
            )

        max_tokens = config.DEFAULT_MAX_TOKENS
        state.has_escalated = False
        messages.append({"role": "assistant", "content": response.content})

        if not has_tool_use(response.content):
            final_text = extract_text(response.content)

            if finalize_user_turn:
                trigger_hooks("Stop", messages)

            # 只有真人触发且正常完成的回合才提取记忆。
            if (
                finalize_user_turn
                and memory_store is not None
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
                    final_text,
                )
                memory_store.consolidate_if_needed()

            return AgentLoopOutcome("completed", final_text, rounds_used=_round_index + 1)

        results = []
        compacted_now = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            span_id = f"tool_{uuid.uuid4().hex}"
            if run_context is not None:
                run_context.recorder.record(
                    "tool_requested",
                    {
                        "tool_call_id": block.id,
                        "tool_name": block.name,
                        "arguments": block.input,
                    },
                    span_id=span_id,
                    agent_id=run_context.agent_id,
                )

            try:
                prepared = prepare_research_arguments(
                    block.name,
                    block.input,
                    required_cutoff=required_cutoff,
                )
            except CutoffMismatch as error:
                if run_context is not None:
                    run_context.recorder.record(
                        "cutoff_validation",
                        {
                            "decision": "rejected",
                            "required_cutoff": required_cutoff,
                            "supplied_cutoff": block.input.get("cutoff"),
                            "error": str(error),
                        },
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
                    run_context.recorder.record(
                        "tool_result",
                        {
                            "success": False,
                            "error_code": "cutoff_mismatch",
                            "message": str(error),
                        },
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(
                            {
                                "ok": False,
                                "error": {
                                    "code": "cutoff_mismatch",
                                    "message": str(error),
                                },
                            }
                        ),
                    }
                )
                continue
            arguments = prepared.arguments
            if run_context is not None:
                run_context.recorder.record(
                    "cutoff_validation",
                    {
                        "decision": prepared.decision,
                        "required_cutoff": required_cutoff,
                        "supplied_cutoff": prepared.supplied_cutoff,
                        "normalized_cutoff": arguments.get("cutoff"),
                    },
                    span_id=span_id,
                    agent_id=run_context.agent_id,
                )

            if block.name == "compact":
                messages[:] = compact_history(
                    messages,
                    on_compaction=record_compaction,
                    method="manual",
                )
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
                if run_context is not None:
                    run_context.recorder.record(
                        "permission_decision",
                        {"decision": "deny", "reason": "pre_tool_hook"},
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
                    run_context.recorder.record(
                        "tool_result",
                        {"success": False, "output": str(blocked)},
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(blocked),
                    }
                )
                continue

            call = ToolCall(block.id, block.name, arguments)
            permission_decision = permissions.decide(call)
            if not permissions.approve(call, approval_callback):
                if run_context is not None:
                    run_context.recorder.record(
                        "permission_decision",
                        {
                            "decision": "deny",
                            "policy_decision": permission_decision.value,
                            "human_approval": approval_callback is not None,
                        },
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
                    run_context.recorder.record(
                        "tool_result",
                        {"success": False, "error_code": "permission_denied"},
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
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

            if run_context is not None:
                run_context.recorder.record(
                    "permission_decision",
                    {
                        "decision": "allow",
                        "policy_decision": permission_decision.value,
                        "human_approval": (
                            permission_decision is PermissionDecision.ASK
                        ),
                    },
                    span_id=span_id,
                    agent_id=run_context.agent_id,
                )

            if should_run_background(block.name, arguments):
                background_id = start_background_task(
                    block, handlers, parent_span_id=span_id
                )
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
                if run_context is not None:
                    run_context.recorder.record(
                        "tool_result",
                        {
                            "success": True,
                            "background_id": background_id,
                            "output": output,
                        },
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
                continue

            handler = handlers.get(block.name)
            capture = (
                ToolCapture(run_context.recorder, span_id)
                if run_context is not None
                else None
            )
            if run_context is not None:
                run_context.recorder.record(
                    "tool_started",
                    {
                        "tool_call_id": block.id,
                        "tool_name": block.name,
                        "arguments": arguments,
                    },
                    span_id=span_id,
                    agent_id=run_context.agent_id,
                )
            started = time.monotonic()
            tool_error = None
            try:
                with bind_tool_capture(capture):
                    output = call_tool_handler(
                        handler, arguments, block.name, capture=capture
                    )
            except Exception as error:
                tool_error = error
                output = f"Error: {type(error).__name__}: {error}"
            evidence_record = None
            if tool_error is None:
                evidence_record = evidence_record_from_result(
                    block.name, str(output)
                )
                if evidence_record is not None and evidence_registry is not None:
                    evidence_registry.register(evidence_record)
            if run_context is not None:
                output_ref = run_context.recorder.store_artifact(
                    str(output),
                    media_type="application/json"
                    if str(output).lstrip().startswith(("{", "["))
                    else "text/plain",
                    source=f"tool_result:{block.name}",
                    suffix=".json"
                    if str(output).lstrip().startswith(("{", "["))
                    else ".txt",
                )
                terminal_payload = {
                    "latency_ms": round(
                        (time.monotonic() - started) * 1000, 3
                    ),
                    "output_artifact": output_ref.as_dict(),
                    "raw_artifacts": [
                        item.as_dict()
                        for item in (capture.artifacts if capture else [])
                    ],
                }
                if tool_error is not None:
                    run_context.recorder.record(
                        "tool_error",
                        {
                            **terminal_payload,
                            "exception_class": type(tool_error).__name__,
                            "message": str(tool_error),
                        },
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
                else:
                    run_context.recorder.record(
                        "tool_result",
                        {**terminal_payload, "success": True},
                        span_id=span_id,
                        agent_id=run_context.agent_id,
                    )
                    record_research_evidence(
                        run_context,
                        block.name,
                        str(output),
                        output_artifact=output_ref,
                        raw_artifacts=capture.artifacts if capture else [],
                        span_id=span_id,
                        registered_sources=registered_sources,
                        record=evidence_record,
                    )
            trigger_hooks("PostToolUse", block, output)
            print(str(output)[:300])

            if block.name == "todo_write":
                if todo_state is not None:
                    todo_state["rounds_since_todo"] = 0
                else:
                    rounds_since_todo = 0
            else:
                if todo_state is not None:
                    todo_state["rounds_since_todo"] += 1
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
    return AgentLoopOutcome(
        "max_rounds",
        "",
        "max_rounds",
        f"agent exceeded maximum rounds ({max_rounds})",
        max_rounds,
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
                    "Summarize this financial-research history. Preserve goals, "
                    "evidence, source URLs, decisions, pending tasks, and errors."
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
