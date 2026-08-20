from __future__ import annotations

import copy
import json

import pytest

from simple_cc import agent, config, context, subagents, teams, tools
from simple_cc.provider import (
    ContextLengthError,
    ProviderResponse,
    TextBlock,
    ToolUseBlock,
)
from simple_cc.research_models import (
    EvidenceRegistry,
    ResearchPlan,
    ResearchRank,
    ResearchWorkflowResult,
)
from simple_cc.research_workflow import ResearchWorkflowError
from simple_cc.trace import RunContext, TraceRecorder, read_trace_lines


class ScriptedProvider:
    def __init__(self, responses, events=None):
        self.responses = list(responses)
        self.requests = []
        self.events = events

    def create(self, messages, system, tools, max_tokens, model=None):
        if self.events is not None:
            self.events.append("provider")
        self.requests.append({
            "messages": copy.deepcopy(messages),
            "system": system,
            "tools": copy.deepcopy(tools),
            "max_tokens": max_tokens,
            "model": model,
        })
        assert self.responses, "unexpected provider call"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def source_loop(tmp_path, monkeypatch):
    original_workspace = config.WORKDIR
    config.configure_workspace(tmp_path)
    monkeypatch.setattr(agent, "rounds_since_todo", 0, raising=False)
    yield
    config.configure_workspace(original_workspace)


def install(provider, monkeypatch):
    monkeypatch.setattr(agent, "client", provider)
    monkeypatch.setattr(context, "client", provider)
    monkeypatch.setattr(subagents, "client", provider)
    teams.set_team_provider(provider)


def test_agent_loop_writes_file_then_returns_exact_tool_result_order(
    source_loop, tmp_path, monkeypatch
):
    assert callable(getattr(agent, "agent_loop", None)), (
        "Task 6 must expose the S20 agent_loop"
    )
    events = []
    provider = ScriptedProvider(
        [
            ProviderResponse(
                content=[
                    TextBlock(text="I will create it."),
                    ToolUseBlock(
                        id="toolu_write",
                        name="write_file",
                        input={"path": "hello.txt", "content": "hello"},
                    ),
                ],
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content=[TextBlock(text="Created hello.txt.")],
                stop_reason="end_turn",
            ),
        ],
        events,
    )
    install(provider, monkeypatch)
    real_write = tools.TOOL_HANDLERS["write_file"]

    def observed_write(**arguments):
        events.append("write_file")
        return real_write(**arguments)

    monkeypatch.setitem(tools.TOOL_HANDLERS, "write_file", observed_write)
    messages = [{"role": "user", "content": "Create hello.txt"}]

    agent.agent_loop(messages, {})

    assert (tmp_path / "hello.txt").read_text() == "hello"
    assert events == ["provider", "write_file", "provider"]
    assert provider.requests[0]["messages"] == [
        {"role": "user", "content": "Create hello.txt"}
    ]
    assert provider.requests[1]["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_write",
                "content": "Wrote 5 bytes to hello.txt",
            }
        ],
    }
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_agent_loop_returns_pdf_page_result_before_cited_answer(
    monkeypatch,
):
    monkeypatch.setattr(agent.config, "MEMORY_ENABLED", False)
    monkeypatch.setattr(agent, "rounds_since_todo", 0, raising=False)
    tool_output = (
        '{"ok":true,"pages":[{"page_number":11,'
        '"content":"--- PAGE 11 START ---\\nRevenue 120\\n'
        '--- PAGE 11 END ---"}]}'
    )
    provider = ScriptedProvider(
        [
            ProviderResponse(
                content=[
                    ToolUseBlock(
                        id="toolu_pdf",
                        name="pdf_fetch",
                        input={
                            "url": "https://example.com/report.pdf",
                            "start_page": 11,
                            "page_count": 1,
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content=[TextBlock(text="Revenue was 120 [PDF p. 11].")],
                stop_reason="end_turn",
            ),
        ]
    )
    install(provider, monkeypatch)
    monkeypatch.setitem(
        tools.TOOL_HANDLERS,
        "pdf_fetch",
        lambda **arguments: tool_output,
    )
    messages = [{"role": "user", "content": "Read the annual report"}]

    agent.agent_loop(messages, {})

    assert provider.requests[1]["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_pdf",
                "content": tool_output,
            }
        ],
    }
    assert messages[-1] == {
        "role": "assistant",
        "content": [TextBlock(text="Revenue was 120 [PDF p. 11].")],
    }


def test_prompt_too_long_runs_one_real_reactive_compaction_retry(
    source_loop, tmp_path, monkeypatch
):
    assert callable(getattr(agent, "agent_loop", None)), (
        "Task 6 must expose the S20 agent_loop"
    )
    provider = ScriptedProvider([
        RuntimeError("prompt is too long"),
        ProviderResponse(
            content=[TextBlock(text="Earlier goal and constraints.")],
            stop_reason="end_turn",
        ),
        ProviderResponse(
            content=[TextBlock(text="Recovered once.")],
            stop_reason="end_turn",
        ),
    ])
    install(provider, monkeypatch)
    messages = [{"role": "user", "content": f"message-{index}"} for index in range(7)]

    agent.agent_loop(messages, {})

    assert len(provider.requests) == 3
    assert provider.requests[0]["tools"] == tools.TOOL_DEFINITIONS
    assert provider.requests[1]["tools"] == []
    assert provider.requests[2]["tools"] == tools.TOOL_DEFINITIONS
    assert provider.requests[2]["messages"][0] == {
        "role": "user",
        "content": "[Reactive compact]\n\nEarlier goal and constraints.",
    }
    assert len(list((tmp_path / ".transcripts").glob("*.jsonl"))) == 1


def test_maximum_context_length_error_retries_once_then_records_second_error(
    source_loop, tmp_path, monkeypatch
):
    provider = ScriptedProvider(
        [
            ContextLengthError("maximum context length exceeded"),
            ContextLengthError("maximum context length exceeded"),
        ]
    )
    install(provider, monkeypatch)
    monkeypatch.setattr(
        context, "summarize_history", lambda messages: "Earlier work."
    )
    messages = [
        {"role": "user", "content": f"message-{index}"}
        for index in range(7)
    ]

    agent.agent_loop(messages, {})

    assert len(provider.requests) == 2
    assert len(list((tmp_path / ".transcripts").glob("*.jsonl"))) == 1
    assert messages[-1] == {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": (
                    "[Error] ContextLengthError: "
                    "maximum context length exceeded"
                ),
            }
        ],
    }


def test_compact_tool_mutates_history_without_dispatching_placeholder(
    source_loop, monkeypatch
):
    assert callable(getattr(agent, "agent_loop", None)), (
        "Task 6 must expose the S20 agent_loop"
    )
    provider = ScriptedProvider([
        ProviderResponse(
            content=[ToolUseBlock(id="toolu_compact", name="compact", input={})],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            content=[TextBlock(text="Preserve the current goal.")],
            stop_reason="end_turn",
        ),
        ProviderResponse(
            content=[TextBlock(text="Continued after compaction.")],
            stop_reason="end_turn",
        ),
    ])
    install(provider, monkeypatch)
    messages = [{"role": "user", "content": "Compact now"}]

    agent.agent_loop(messages, {})

    assert messages[0] == {
        "role": "user",
        "content": "[Compacted]\n\nPreserve the current goal.",
    }
    assert messages[1] == {
        "role": "user",
        "content": "[Compacted. Continue with summarized context.]",
    }
    assert not any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict)
            and block.get("tool_use_id") == "toolu_compact"
            for block in message["content"]
        )
        for message in messages
    )


def test_agent_loop_uses_injected_prompt_and_reports_rounds():
    provider = ScriptedProvider([
        ProviderResponse(
            content=[TextBlock(text="stage notes")],
            stop_reason="end_turn",
        )
    ])

    outcome = agent.agent_loop(
        [{"role": "user", "content": "research"}],
        {},
        provider=provider,
        tools=[],
        handlers={},
        memory_enabled=False,
        system_prompt="STAGE PROMPT",
        finalize_user_turn=False,
    )

    assert outcome.final_text == "stage notes"
    assert outcome.rounds_used == 1
    assert provider.requests[0]["system"] == "STAGE PROMPT"


def test_agent_loop_registers_foreground_fetch_without_trace():
    provider = ScriptedProvider([
        ProviderResponse(
            content=[
                ToolUseBlock(
                    id="fetch-1",
                    name="web_fetch",
                    input={"url": "https://example.com/report"},
                )
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            content=[TextBlock(text="research notes")],
            stop_reason="end_turn",
        ),
    ])
    registry = EvidenceRegistry()

    outcome = agent.agent_loop(
        [{"role": "user", "content": "fetch the report"}],
        {},
        provider=provider,
        tools=[{"name": "web_fetch", "description": "fetch", "input_schema": {}}],
        handlers={
            "web_fetch": lambda **_: (
                '{"ok": true, "operation": "fetch", '
                '"url": "https://example.com/report", "title": "Report", '
                '"content": "fetched facts", "published_at": null, '
                '"date_status": "unknown", "cutoff": null}'
            )
        },
        memory_enabled=False,
        evidence_registry=registry,
    )

    assert outcome.final_text == "research notes"
    assert len(registry.records) == 1
    assert registry.records[0].canonical_url == "https://example.com/report"


def test_traced_ordinary_loop_no_longer_applies_research_gate(tmp_path):
    provider = ScriptedProvider([
        ProviderResponse(
            content=[TextBlock(text="ordinary answer")],
            stop_reason="end_turn",
        )
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="ordinary-run")
    recorder.start_run(task_id="ordinary", question="q", cutoff=None, metadata={})
    runtime = agent.SourceRuntime(provider, recorder=recorder, memory_enabled=False)

    assert runtime.run_turn(
        "q",
        task_id="ordinary",
        run_metadata={"task_type": "normal"},
    ) == "ordinary answer"
    assert len(provider.requests) == 1


@pytest.mark.parametrize("task_type", [None, "normal", "future_kind"])
def test_source_runtime_routes_only_explicit_research(monkeypatch, task_type):
    provider = ScriptedProvider([
        ProviderResponse(
            content=[TextBlock(text="ordinary")],
            stop_reason="end_turn",
        )
    ])
    runtime = agent.SourceRuntime(provider, memory_enabled=False)

    class ForbiddenWorkflow:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ordinary task must not construct research workflow")

    monkeypatch.setattr(agent, "ResearchWorkflow", ForbiddenWorkflow)

    metadata = {} if task_type is None else {"task_type": task_type}
    assert runtime.run_turn("q", run_metadata=metadata) == "ordinary"
    assert "financial research agent" not in provider.requests[0]["system"]


@pytest.mark.parametrize("task_type", ["research", "research_analysis"])
def test_source_runtime_routes_research_aliases(monkeypatch, task_type):
    provider = ScriptedProvider([])
    runtime = agent.SourceRuntime(provider, memory_enabled=False)
    events = []
    seen = {}

    class FakeWorkflow:
        def __init__(self, provider, executor, *, run_context=None):
            seen["provider"] = provider
            seen["run_context"] = run_context

        def run(self, question, cutoff, *, registry=None):
            seen.update({"question": question, "cutoff": cutoff, "registry": registry})
            return ResearchWorkflowResult(
                "research answer",
                ResearchPlan(ResearchRank.LIGHT, ("facts",), "narrow"),
                2,
                False,
                False,
            )

    monkeypatch.setattr(agent, "ResearchWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        agent,
        "trigger_hooks",
        lambda event, *args: events.append(event),
    )

    assert runtime.run_turn(
        "q",
        cutoff="2025-05-01",
        run_metadata={"task_type": task_type},
    ) == "research answer"
    assert isinstance(seen["registry"], EvidenceRegistry)
    assert seen["provider"] is runtime.tracing_provider
    assert seen["question"] == "q"
    assert seen["cutoff"] == "2025-05-01"
    assert events == ["UserPromptSubmit", "Stop"]
    assert runtime.messages == [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "research answer"}],
        },
    ]


def test_source_runtime_research_executor_uses_shared_registry_and_private_notes(
    monkeypatch,
):
    provider = ScriptedProvider([])
    runtime = agent.SourceRuntime(
        provider,
        tool_definitions=[{
            "name": "web_fetch",
            "description": "fetch",
            "input_schema": {},
        }],
        tool_handlers={"web_fetch": lambda **_: "unused"},
        max_rounds=40,
        memory_enabled=False,
    )
    captured = {}

    def fake_agent_loop(messages, context, permissions, approval_callback, **kwargs):
        captured.update({
            "messages": copy.deepcopy(messages),
            "context": context,
            "permissions": permissions,
            "approval_callback": approval_callback,
            **kwargs,
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "private research notes"}],
        })
        return agent.AgentLoopOutcome("completed", "private research notes", rounds_used=3)

    class ExercisingWorkflow:
        def __init__(self, provider, executor, *, run_context=None):
            self.executor = executor

        def run(self, question, cutoff, *, registry=None):
            prompt = json.dumps({
                "question": question,
                "cutoff": cutoff,
                "rank": "light",
                "directions": ["primary facts"],
                "research_gaps": ["missing filing"],
                "remaining_rounds": 7,
            })
            outcome = self.executor(prompt, 7, registry)
            assert outcome.rounds_used == 3
            return ResearchWorkflowResult(
                "public report",
                ResearchPlan(ResearchRank.LIGHT, ("primary facts",), "narrow"),
                3,
                False,
                False,
            )

    monkeypatch.setattr(agent, "agent_loop", fake_agent_loop)
    monkeypatch.setattr(agent, "ResearchWorkflow", ExercisingWorkflow)
    monkeypatch.setattr(agent, "trigger_hooks", lambda *args: None)

    assert runtime.run_turn(
        "question",
        cutoff="2025-05-01",
        run_metadata={"task_type": "research"},
    ) == "public report"
    assert captured["messages"][0]["role"] == "user"
    assert "primary facts" in captured["messages"][0]["content"]
    assert captured["tools"] is runtime.tool_definitions
    assert captured["handlers"] is runtime.tool_handlers
    assert captured["permissions"] is runtime.permissions
    assert captured["max_rounds"] == 7
    assert captured["memory_enabled"] is False
    assert captured["finalize_user_turn"] is False
    assert isinstance(captured["evidence_registry"], EvidenceRegistry)
    assert captured["research_cutoff"] == "2025-05-01"
    assert "evidence-gathering research executor" in captured["system_prompt"]
    assert "final report is produced only" in captured["system_prompt"]
    assert "untrusted data" in captured["system_prompt"]
    assert "missing filing" in captured["system_prompt"]
    assert "private research notes" not in repr(runtime.messages)


def _single_execution_workflow():
    class SingleExecutionWorkflow:
        def __init__(self, provider, executor, *, run_context=None):
            self.executor = executor

        def run(self, question, cutoff, *, registry=None):
            prompt = json.dumps({
                "question": question,
                "cutoff": cutoff,
                "rank": "light",
                "directions": ["primary facts"],
                "research_gaps": [],
                "remaining_rounds": 2,
            })
            outcome = self.executor(prompt, 2, registry)
            return ResearchWorkflowResult(
                "public report",
                ResearchPlan(ResearchRank.LIGHT, ("primary facts",), "narrow"),
                outcome.rounds_used,
                False,
                False,
            )

    return SingleExecutionWorkflow


def test_untraced_explicit_research_injects_cutoff(monkeypatch):
    seen = []
    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock("search-1", "web_search", {"query": "q"})],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="research notes")], "end_turn"),
    ])
    runtime = agent.SourceRuntime(
        provider,
        tool_definitions=[{
            "name": "web_search",
            "description": "search",
            "input_schema": {},
        }],
        tool_handlers={
            "web_search": lambda **arguments: (
                seen.append(arguments)
                or json.dumps({"ok": True, "results": []})
            )
        },
        memory_enabled=False,
    )
    monkeypatch.setattr(agent, "ResearchWorkflow", _single_execution_workflow())

    assert runtime.run_turn(
        "q",
        cutoff="2025-05-01",
        run_metadata={"task_type": "research"},
    ) == "public report"
    assert seen == [{"query": "q", "cutoff": "2025-05-01"}]


def test_untraced_explicit_research_rejects_cutoff_mismatch(monkeypatch):
    called = False

    def fetch(**arguments):
        nonlocal called
        called = True
        return "should not run"

    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock(
                "fetch-1",
                "web_fetch",
                {
                    "url": "https://example.com/report",
                    "cutoff": "2025-05-02",
                },
            )],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="research notes")], "end_turn"),
    ])
    runtime = agent.SourceRuntime(
        provider,
        tool_definitions=[{
            "name": "web_fetch",
            "description": "fetch",
            "input_schema": {},
        }],
        tool_handlers={"web_fetch": fetch},
        memory_enabled=False,
    )
    monkeypatch.setattr(agent, "ResearchWorkflow", _single_execution_workflow())

    assert runtime.run_turn(
        "q",
        cutoff="2025-05-01",
        run_metadata={"task_type": "research_analysis"},
    ) == "public report"
    result = provider.requests[1]["messages"][-1]["content"][0]
    assert called is False
    assert json.loads(result["content"])["error"]["code"] == "cutoff_mismatch"


def test_untraced_ordinary_task_does_not_enforce_research_cutoff(monkeypatch):
    seen = []
    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock("search-1", "web_search", {"query": "q"})],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="ordinary answer")], "end_turn"),
    ])
    runtime = agent.SourceRuntime(
        provider,
        tool_definitions=[{
            "name": "web_search",
            "description": "search",
            "input_schema": {},
        }],
        tool_handlers={
            "web_search": lambda **arguments: (
                seen.append(arguments)
                or json.dumps({"ok": True, "results": []})
            )
        },
        memory_enabled=False,
    )
    monkeypatch.setattr(
        agent,
        "ResearchWorkflow",
        lambda *args, **kwargs: pytest.fail("ordinary task constructed workflow"),
    )

    assert runtime.run_turn(
        "q",
        cutoff="2025-05-01",
        run_metadata={"task_type": "normal"},
    ) == "ordinary answer"
    assert seen == [{"query": "q"}]


def _direct_research_run(tmp_path, run_id, cutoff):
    recorder = TraceRecorder(tmp_path / run_id, run_id=run_id)
    recorder.start_run(
        task_id=f"{run_id}-task",
        question="q",
        cutoff=cutoff,
        metadata={},
    )
    return RunContext(recorder, run_id, f"{run_id}-task", cutoff)


def test_direct_agent_loop_falls_back_to_run_context_cutoff(tmp_path):
    seen = []
    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock("search-1", "web_search", {"query": "q"})],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="research notes")], "end_turn"),
    ])
    registry = EvidenceRegistry()
    run = _direct_research_run(tmp_path, "legacy-inject", "2025-05-01")

    outcome = agent.agent_loop(
        [{"role": "user", "content": "q"}],
        {},
        provider=provider,
        tools=[{
            "name": "web_search",
            "description": "search",
            "input_schema": {},
        }],
        handlers={
            "web_search": lambda **arguments: (
                seen.append(arguments)
                or json.dumps({"ok": True, "results": []})
            )
        },
        memory_enabled=False,
        run_context=run,
        evidence_registry=registry,
        finalize_user_turn=False,
    )

    assert outcome.status == "completed"
    assert seen == [{"query": "q", "cutoff": "2025-05-01"}]


def test_direct_agent_loop_fallback_rejects_cutoff_mismatch(tmp_path):
    called = False

    def fetch(**arguments):
        nonlocal called
        called = True
        return "should not run"

    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock(
                "fetch-1",
                "web_fetch",
                {
                    "url": "https://example.com/report",
                    "cutoff": "2025-05-02",
                },
            )],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="research notes")], "end_turn"),
    ])
    registry = EvidenceRegistry()
    run = _direct_research_run(tmp_path, "legacy-mismatch", "2025-05-01")

    outcome = agent.agent_loop(
        [{"role": "user", "content": "q"}],
        {},
        provider=provider,
        tools=[{
            "name": "web_fetch",
            "description": "fetch",
            "input_schema": {},
        }],
        handlers={"web_fetch": fetch},
        memory_enabled=False,
        run_context=run,
        evidence_registry=registry,
        finalize_user_turn=False,
    )
    result = provider.requests[1]["messages"][-1]["content"][0]

    assert outcome.status == "completed"
    assert called is False
    assert json.loads(result["content"])["error"]["code"] == "cutoff_mismatch"


def test_direct_agent_loop_explicit_research_cutoff_precedes_run_context(tmp_path):
    seen = []
    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock("search-1", "web_search", {"query": "q"})],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="research notes")], "end_turn"),
    ])
    registry = EvidenceRegistry()
    run = _direct_research_run(tmp_path, "explicit-cutoff", "2025-04-01")

    outcome = agent.agent_loop(
        [{"role": "user", "content": "q"}],
        {},
        provider=provider,
        tools=[{
            "name": "web_search",
            "description": "search",
            "input_schema": {},
        }],
        handlers={
            "web_search": lambda **arguments: (
                seen.append(arguments)
                or json.dumps({"ok": True, "results": []})
            )
        },
        memory_enabled=False,
        run_context=run,
        evidence_registry=registry,
        research_cutoff="2025-05-01",
        finalize_user_turn=False,
    )

    assert outcome.status == "completed"
    assert seen == [{"query": "q", "cutoff": "2025-05-01"}]


def test_source_runtime_traces_explicit_and_default_routing(tmp_path):
    provider = ScriptedProvider([
        ProviderResponse([TextBlock(text="one")], "end_turn"),
        ProviderResponse([TextBlock(text="two")], "end_turn"),
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="route-run")
    recorder.start_run(task_id="route-task", question="q", cutoff=None, metadata={})
    runtime = agent.SourceRuntime(provider, recorder=recorder, memory_enabled=False)

    runtime.run_turn(
        "q1",
        task_id="route-task",
        run_metadata={"task_type": "normal"},
    )
    runtime.run_turn(
        "q2",
        task_id="route-task",
        run_metadata={"task_type": "unexpected"},
    )

    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    routed = [row["payload"] for row in rows if row["event_type"] == "task_routed"]
    assert incomplete is False
    assert routed == [
        {
            "raw_task_type": "normal",
            "normalized_task_kind": "normal",
            "reason": "explicit",
        },
        {
            "raw_task_type": "unexpected",
            "normalized_task_kind": "normal",
            "reason": "default",
        },
    ]


def test_source_runtime_adapts_research_workflow_failure(monkeypatch):
    runtime = agent.SourceRuntime(ScriptedProvider([]), memory_enabled=False)

    class FailedWorkflow:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, question, cutoff, *, registry=None):
            raise ResearchWorkflowError("ProviderUnavailable", "planner failed")

    monkeypatch.setattr(agent, "ResearchWorkflow", FailedWorkflow)

    assert runtime.run_turn(
        "q", run_metadata={"task_type": "research"}
    ) == ""
    assert runtime.last_outcome == agent.AgentLoopOutcome(
        "failed",
        "",
        "ProviderUnavailable",
        "planner failed",
        0,
    )
