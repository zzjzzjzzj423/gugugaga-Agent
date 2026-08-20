import json

import pytest

from simple_cc import config
from simple_cc import context as context_module
from simple_cc import recovery as recovery_module
from simple_cc.context import (
    reactive_compact,
    snip_compact,
    summarize_history,
    tool_result_budget,
    write_transcript,
)
from simple_cc.prompts import PROMPT_SECTIONS
from simple_cc.provider import ProviderResponse, TextBlock
from simple_cc.recovery import RecoveryState, is_prompt_too_long_error, with_retry
from simple_cc.memory import MemoryStore
from simple_cc.telemetry import TracingProvider
from simple_cc.tools import TOOL_DEFINITIONS, TOOL_HANDLERS
from simple_cc.trace import (
    RunContext,
    TraceRecorder,
    TraceWriteError,
    bind_run_context,
)


@pytest.fixture
def configured_workspace(tmp_path):
    original = config.WORKDIR
    config.configure_workspace(tmp_path)
    try:
        yield tmp_path
    finally:
        config.configure_workspace(original)


class ScriptedSummaryProvider:
    def __init__(self, text):
        self.text = text
        self.requests = []

    def create(self, messages, system, tools, max_tokens, model=None):
        self.requests.append(
            {
                "messages": messages,
                "system": system,
                "tools": tools,
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        return ProviderResponse(
            content=[TextBlock(text=self.text)], stop_reason="end_turn"
        )


def test_large_tool_result_is_persisted_and_replaced_by_s20_reference(
    configured_workspace,
):
    output = "x" * (config.PERSIST_THRESHOLD + 1)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_read",
                    "content": output,
                }
            ],
        }
    ]

    budgeted = tool_result_budget(messages, max_bytes=10)

    persisted = (
        configured_workspace
        / ".task_outputs"
        / "tool-results"
        / "call_read.txt"
    )
    assert budgeted is messages
    assert persisted.read_text() == output
    assert budgeted[0]["content"][0]["content"] == (
        f"<persisted-output>\nFull output: {persisted}\n"
        f"Preview:\n{output[:2000]}\n</persisted-output>"
    )


def test_snip_compact_keeps_anthropic_tool_use_with_its_tool_result():
    messages = [
        {"role": "user", "content": "head-0"},
        {"role": "assistant", "content": [{"type": "text", "text": "head-1"}]},
        {"role": "user", "content": "head-2"},
        {"role": "assistant", "content": [{"type": "text", "text": "old"}]},
        {"role": "user", "content": "old-result"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_glob",
                    "name": "glob",
                    "input": {"pattern": "*.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_glob",
                    "content": "a.py",
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]

    compacted = snip_compact(messages, max_messages=5)

    assert compacted[-3:] == messages[-3:]
    assert compacted[3] == {"role": "user", "content": "[snipped 2 messages]"}


def test_write_transcript_stays_below_configured_workspace(configured_workspace):
    messages = [{"role": "user", "content": "inspect"}]

    transcript = write_transcript(messages)

    assert transcript.parent == configured_workspace / ".transcripts"
    assert transcript.relative_to(configured_workspace)
    assert [json.loads(line) for line in transcript.read_text().splitlines()] == messages


def test_summarize_history_calls_provider_with_anthropic_shaped_message(monkeypatch):
    provider = ScriptedSummaryProvider("Goal and constraints preserved.")
    monkeypatch.setattr(context_module, "client", provider)

    summary = summarize_history([{"role": "user", "content": "fix the parser"}])

    assert summary == "Goal and constraints preserved."
    assert provider.requests == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Summarize this financial-research conversation so work can continue. "
                        "Preserve the current question, key findings, source URLs, remaining "
                        "research, uncertainties, and user constraints.\n\n"
                        '[{"role": "user", "content": "fix the parser"}]'
                    ),
                }
            ],
            "system": "",
            "tools": [],
            "max_tokens": 2000,
            "model": config.MODEL,
        }
    ]


def test_with_retry_recovers_a_scripted_transient_rate_limit(monkeypatch):
    outcomes = [RuntimeError("429 rate limit"), "ok"]
    monkeypatch.setattr(recovery_module, "retry_delay", lambda attempt: 0)
    monkeypatch.setattr(recovery_module.time, "sleep", lambda delay: None)

    def operation():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert with_retry(operation, RecoveryState()) == "ok"
    assert outcomes == []


def test_with_retry_never_classifies_trace_failure_as_transient(monkeypatch):
    error = TraceWriteError("trace unavailable: 429 overloaded 529")
    calls = 0
    sleeps = []
    monkeypatch.setattr(recovery_module, "retry_delay", lambda attempt: 0)
    monkeypatch.setattr(
        recovery_module.time,
        "sleep",
        lambda delay: sleeps.append(delay),
    )

    def operation():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(TraceWriteError) as caught:
        with_retry(operation, RecoveryState())

    assert caught.value is error
    assert str(caught.value) == "trace unavailable: 429 overloaded 529"
    assert calls == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("call_kind", "operation"),
    (
        ("memory_retrieval", "retrieve"),
        ("memory_extraction", "extract"),
        ("memory_consolidation", "consolidate"),
    ),
)
def test_memory_model_trace_failures_escape_internal_fallbacks(
    tmp_path,
    monkeypatch,
    call_kind,
    operation,
):
    delegate = ScriptedSummaryProvider('["project-alpha.md"]')
    recorder = TraceRecorder(
        tmp_path / f"trace-{operation}",
        run_id=f"memory-{operation}-run",
    )
    recorder.start_run(
        task_id="research-task",
        question="Remember project alpha",
        cutoff=None,
        metadata={"task_type": "research"},
    )
    original_record = recorder.record
    trace_error = TraceWriteError(f"{call_kind} trace unavailable")

    def fail_internal_memory_trace(event_type, payload, **kwargs):
        if (
            event_type == "llm_request_started"
            and payload.get("call_kind") == call_kind
        ):
            raise trace_error
        return original_record(event_type, payload, **kwargs)

    monkeypatch.setattr(recorder, "record", fail_internal_memory_trace)
    store = MemoryStore(
        tmp_path / f"memory-{operation}",
        provider=TracingProvider(delegate),
        consolidate_threshold=1,
        consolidate_cooldown_seconds=0,
    )
    assert store.upsert(
        action="create",
        name="project-alpha",
        mem_type="project",
        description="Project alpha requirements",
        body="Use the verified research workflow.",
    )
    run = RunContext(
        recorder,
        f"memory-{operation}-run",
        "research-task",
        None,
    )

    with bind_run_context(run), pytest.raises(TraceWriteError) as caught:
        if operation == "retrieve":
            store.load_relevant([
                {"role": "user", "content": "Research project alpha"}
            ])
        elif operation == "extract":
            store.extract(
                [{"role": "user", "content": "Remember project alpha"}],
                "Acknowledged.",
            )
        else:
            store.consolidate_if_needed()

    assert caught.value is trace_error
    assert delegate.requests == []


def test_prompt_too_long_classification_allows_one_reactive_compaction(
    configured_workspace, monkeypatch
):
    state = RecoveryState()
    messages = [{"role": "user", "content": f"message-{index}"} for index in range(7)]
    monkeypatch.setattr(
        context_module, "summarize_history", lambda history: "Earlier work summary."
    )

    for error in (
        RuntimeError("prompt is too long"),
        RuntimeError("context_length_exceeded"),
    ):
        if (
            is_prompt_too_long_error(error)
            and not state.has_attempted_reactive_compact
        ):
            messages[:] = reactive_compact(messages)
            state.has_attempted_reactive_compact = True

    assert state.has_attempted_reactive_compact
    assert messages[0] == {
        "role": "user",
        "content": "[Reactive compact]\n\nEarlier work summary.",
    }
    assert len(list((configured_workspace / ".transcripts").glob("*.jsonl"))) == 1


def test_compact_tool_is_advertised_with_the_s20_schema():
    compact = next(tool for tool in TOOL_DEFINITIONS if tool["name"] == "compact")

    assert "compact" in PROMPT_SECTIONS["tools"]
    assert compact == {
        "name": "compact",
        "description": (
            "Summarize earlier conversation and continue with compacted context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"focus": {"type": "string"}},
            "required": [],
        },
    }


def test_compact_tool_has_a_fixed_dispatch_placeholder_for_agent_loop_handling():
    assert TOOL_HANDLERS["compact"](focus="preserve test failures") == (
        "[Compaction requested.]"
    )
def test_prepare_context_reports_proactive_compaction(tmp_path, monkeypatch):
    from simple_cc import config, context

    old_workspace = config.WORKDIR
    config.configure_workspace(tmp_path)
    monkeypatch.setattr(config, "CONTEXT_LIMIT", 10)
    monkeypatch.setattr(context, "summarize_history", lambda messages: "summary")
    reports = []
    messages = [{"role": "user", "content": "x" * 100}]
    try:
        context.prepare_context(messages, on_compaction=reports.append)
    finally:
        config.configure_workspace(old_workspace)

    assert reports[0].method == "proactive"
    assert reports[0].original_message_count == 1
    assert reports[0].retained_message_count == 1
    assert reports[0].transcript_path.endswith(".jsonl")
