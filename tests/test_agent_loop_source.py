from __future__ import annotations

import copy
import json

import pytest

from simple_cc import agent, config, context, recovery, subagents, teams, tools
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
from simple_cc.trace import (
    RunContext,
    TraceRecorder,
    TraceWriteError,
    read_trace_lines,
)


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
    assert [item["name"] for item in provider.requests[0]["tools"]] == [
        item["name"] for item in runtime.tool_definitions
    ]
    assert {"bash", "task", "schedule_cron", "spawn_teammate"} <= {
        item["name"] for item in provider.requests[0]["tools"]
    }


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


def test_research_stop_hook_follows_successful_final_answer_trace(
    tmp_path, monkeypatch
):
    runtime_events = []
    recorder = TraceRecorder(tmp_path / "run", run_id="publication-order-run")
    recorder.start_run(
        task_id="research-task",
        question="q",
        cutoff=None,
        metadata={"task_type": "research"},
    )
    original_record = recorder.record

    def record_with_order(event_type, payload, **kwargs):
        if event_type == "final_answer":
            runtime_events.append("trace:final_answer")
        return original_record(event_type, payload, **kwargs)

    class SuccessfulWorkflow:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, question, cutoff, *, registry=None):
            return ResearchWorkflowResult(
                "research answer",
                ResearchPlan(ResearchRank.LIGHT, ("facts",), "narrow"),
                0,
                False,
                False,
            )

    monkeypatch.setattr(recorder, "record", record_with_order)
    monkeypatch.setattr(agent, "ResearchWorkflow", SuccessfulWorkflow)
    monkeypatch.setattr(
        agent,
        "trigger_hooks",
        lambda event, *args: runtime_events.append(f"hook:{event}"),
    )
    runtime = agent.SourceRuntime(
        ScriptedProvider([]),
        recorder=recorder,
        memory_enabled=False,
    )

    assert runtime.run_turn(
        "q",
        task_id="research-task",
        run_metadata={"task_type": "research"},
    ) == "research answer"

    assert runtime_events == [
        "hook:UserPromptSubmit",
        "trace:final_answer",
        "hook:Stop",
    ]


def test_final_answer_trace_failure_rolls_back_without_stop(tmp_path, monkeypatch):
    events = []
    recorder = TraceRecorder(tmp_path / "run", run_id="publication-failure-run")
    recorder.start_run(
        task_id="research-task",
        question="q",
        cutoff=None,
        metadata={"task_type": "research"},
    )
    original_record = recorder.record
    trace_error = TraceWriteError("final answer trace unavailable")

    def fail_final_answer(event_type, payload, **kwargs):
        if event_type == "final_answer":
            raise trace_error
        return original_record(event_type, payload, **kwargs)

    class SuccessfulWorkflow:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, question, cutoff, *, registry=None):
            return ResearchWorkflowResult(
                "research answer",
                ResearchPlan(ResearchRank.LIGHT, ("facts",), "narrow"),
                0,
                False,
                False,
            )

    monkeypatch.setattr(recorder, "record", fail_final_answer)
    monkeypatch.setattr(agent, "ResearchWorkflow", SuccessfulWorkflow)
    monkeypatch.setattr(
        agent,
        "trigger_hooks",
        lambda event, *args: events.append(event),
    )
    runtime = agent.SourceRuntime(
        ScriptedProvider([]),
        recorder=recorder,
        memory_enabled=False,
    )

    with pytest.raises(TraceWriteError) as caught:
        runtime.run_turn(
            "q",
            task_id="research-task",
            run_metadata={"task_type": "research"},
        )

    assert caught.value is trace_error
    assert events == ["UserPromptSubmit"]
    assert runtime.messages == []
    assert runtime.last_outcome is None


def test_source_runtime_research_executor_uses_shared_registry_and_private_notes(
    monkeypatch,
):
    provider = ScriptedProvider([])
    runtime = agent.SourceRuntime(
        provider,
        tool_definitions=[
            {"name": "bash", "description": "shell", "input_schema": {}},
            {"name": "web_fetch", "description": "fetch", "input_schema": {}},
            {"name": "web_search", "description": "search", "input_schema": {}},
        ],
        tool_handlers={
            "bash": lambda **_: "forbidden",
            "web_fetch": lambda **_: "unused",
            "pdf_fetch": lambda **_: "handler without definition",
        },
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
    assert [item["name"] for item in captured["tools"]] == ["web_fetch"]
    assert set(captured["handlers"]) == {"web_fetch"}
    assert captured["handlers"]["web_fetch"] is runtime.tool_handlers["web_fetch"]
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
    available_line = next(
        line
        for line in captured["system_prompt"].splitlines()
        if line.startswith("Available tools:")
    )
    assert available_line == "Available tools: web_fetch."
    assert "private research notes" not in repr(runtime.messages)


def test_default_source_runtime_research_path_uses_only_configured_research_tools(
    monkeypatch,
):
    runtime = agent.SourceRuntime(ScriptedProvider([]), memory_enabled=False)
    captured = {}

    def fake_agent_loop(messages, context, permissions, approval_callback, **kwargs):
        captured.update(kwargs)
        return agent.AgentLoopOutcome("completed", "private notes", rounds_used=1)

    monkeypatch.setattr(agent, "agent_loop", fake_agent_loop)
    monkeypatch.setattr(agent, "ResearchWorkflow", _single_execution_workflow())
    monkeypatch.setattr(agent, "trigger_hooks", lambda *args: None)

    assert runtime.run_turn(
        "question",
        run_metadata={"task_type": "research"},
    ) == "public report"

    expected_names = ["web_search", "web_fetch", "pdf_fetch"]
    assert [item["name"] for item in captured["tools"]] == expected_names
    assert list(captured["handlers"]) == expected_names
    assert captured["execution_policy"] is (
        agent.AgentExecutionPolicy.RESEARCH_ISOLATED
    )
    assert "strict_tool_allowlist" not in captured
    available_line = next(
        line
        for line in captured["system_prompt"].splitlines()
        if line.startswith("Available tools:")
    )
    assert available_line == "Available tools: web_search, web_fetch, pdf_fetch."
    assert not {
        "bash",
        "task",
        "create_task",
        "schedule_cron",
        "spawn_teammate",
    } & set(captured["handlers"])


def test_research_path_rejects_hallucinated_background_tool_synchronously(
    monkeypatch,
):
    forbidden_handler_called = False

    def forbidden_bash(**arguments):
        nonlocal forbidden_handler_called
        forbidden_handler_called = True
        return "forbidden"

    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock(
                "bash-1",
                "bash",
                {"command": "pytest", "run_in_background": True},
            )],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="private notes")], "end_turn"),
    ])
    monkeypatch.setitem(agent.TOOL_HANDLERS, "bash", forbidden_bash)
    monkeypatch.setattr(
        agent,
        "start_background_task",
        lambda *args, **kwargs: pytest.fail("research started an async tool"),
    )
    monkeypatch.setattr(
        agent,
        "build_user_content",
        lambda results: pytest.fail("research drained background results"),
    )
    monkeypatch.setattr(agent, "ResearchWorkflow", _single_execution_workflow())
    runtime = agent.SourceRuntime(
        provider,
        approval_callback=lambda call: True,
        memory_enabled=False,
    )

    assert runtime.run_turn(
        "question",
        run_metadata={"task_type": "research"},
    ) == "public report"

    assert forbidden_handler_called is False
    assert [item["name"] for item in provider.requests[0]["tools"]] == [
        "web_search",
        "web_fetch",
        "pdf_fetch",
    ]
    result = provider.requests[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "bash-1"
    assert json.loads(result["content"])["error"]["code"] == (
        "tool_not_available"
    )


@pytest.mark.parametrize("explicit_full_tables", (False, True))
def test_isolated_policy_refilters_default_or_explicit_full_runtime_tables(
    source_loop,
    monkeypatch,
    explicit_full_tables,
):
    forbidden_handler_called = False

    def forbidden_bash(**arguments):
        nonlocal forbidden_handler_called
        forbidden_handler_called = True
        return "forbidden"

    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock("bash-direct", "bash", {"command": "echo nope"})],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="private notes")], "end_turn"),
    ])
    monkeypatch.setitem(agent.TOOL_HANDLERS, "bash", forbidden_bash)
    kwargs = {}
    if explicit_full_tables:
        kwargs.update(tools=agent.TOOL_DEFINITIONS, handlers=agent.TOOL_HANDLERS)

    outcome = agent.agent_loop(
        [{"role": "user", "content": "research"}],
        {},
        provider=provider,
        approval_callback=lambda call: True,
        system_prompt="RESEARCH SYSTEM",
        execution_policy=agent.AgentExecutionPolicy.RESEARCH_ISOLATED,
        **kwargs,
    )

    assert outcome.status == "completed"
    assert forbidden_handler_called is False
    assert [item["name"] for item in provider.requests[0]["tools"]] == [
        "web_search",
        "web_fetch",
        "pdf_fetch",
    ]
    result = provider.requests[1]["messages"][-1]["content"][0]
    assert json.loads(result["content"])["error"]["code"] == (
        "tool_not_available"
    )


def test_isolated_policy_refilters_bash_only_and_malformed_tables(
    source_loop,
):
    forbidden_handler_called = False

    def forbidden_bash(**arguments):
        nonlocal forbidden_handler_called
        forbidden_handler_called = True
        return "forbidden"

    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock("bash-malicious", "bash", {"command": "echo nope"})],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="private notes")], "end_turn"),
    ])

    outcome = agent.agent_loop(
        [{"role": "user", "content": "research"}],
        {},
        provider=provider,
        approval_callback=lambda call: True,
        tools=[
            "not a definition",
            {"name": [], "description": "bad name", "input_schema": {}},
            {"name": "bash", "description": "shell", "input_schema": {}},
            {
                "name": "web_fetch",
                "description": "non-callable handler",
                "input_schema": {},
            },
        ],
        handlers={"bash": forbidden_bash, "web_fetch": "not callable"},
        system_prompt="RESEARCH SYSTEM",
        execution_policy=agent.AgentExecutionPolicy.RESEARCH_ISOLATED,
    )

    assert outcome.status == "completed"
    assert provider.requests[0]["tools"] == []
    assert forbidden_handler_called is False
    result = provider.requests[1]["messages"][-1]["content"][0]
    assert json.loads(result["content"])["error"]["code"] == (
        "tool_not_available"
    )


def test_research_tool_view_fails_closed_on_malformed_and_conflicting_entries():
    fetch = lambda **_: "fetch"
    pdf = lambda **_: "pdf"
    definitions = [
        "not a definition",
        ["also", "invalid"],
        {"name": [], "description": "bad name", "input_schema": {}},
        {"name": "web_fetch", "description": "fetch", "input_schema": {}},
        {
            "name": "web_search",
            "description": "search",
            "input_schema": {},
        },
        {"name": "pdf_fetch", "description": "pdf one", "input_schema": {}},
        {"name": "pdf_fetch", "description": "pdf two", "input_schema": {}},
        {
            "name": "pdf_fetch",
            "description": "bad schema",
            "input_schema": [],
        },
        {"name": "web_search", "input_schema": {}},
    ]

    selected, handlers = agent._research_tool_view(
        definitions,
        {"web_fetch": fetch, "web_search": "not callable", "pdf_fetch": pdf},
    )

    assert selected == [
        {"name": "web_fetch", "description": "fetch", "input_schema": {}}
    ]
    assert handlers == {"web_fetch": fetch}


def test_research_tool_view_deduplicates_only_identical_safe_entries():
    fetch = lambda **_: "fetch"
    definition = {
        "name": "web_fetch",
        "description": "fetch",
        "input_schema": {"type": "object"},
    }

    selected, handlers = agent._research_tool_view(
        [definition, copy.deepcopy(definition)],
        {"web_fetch": fetch},
    )

    assert selected == [definition]
    assert handlers == {"web_fetch": fetch}
    assert agent._research_tool_view([], {"web_fetch": fetch}) == ([], {})


def test_research_tool_view_skips_unhashable_string_subclass_name():
    class UnhashableToolName(str):
        __hash__ = None

    selected, handlers = agent._research_tool_view(
        [{
            "name": UnhashableToolName("web_fetch"),
            "description": "fetch",
            "input_schema": {},
        }],
        {"web_fetch": lambda **_: "fetch"},
    )

    assert selected == []
    assert handlers == {}


def test_research_prompt_does_not_touch_malformed_full_runtime_registry(
    monkeypatch,
):
    runtime = agent.SourceRuntime(
        ScriptedProvider([]),
        tool_definitions=[
            "malformed",
            {"name": [], "description": "bad", "input_schema": {}},
            {"name": "web_fetch", "description": "fetch", "input_schema": {}},
        ],
        tool_handlers={"web_fetch": lambda **_: "unused"},
        memory_enabled=False,
    )
    captured = {}

    def fake_agent_loop(messages, context, permissions, approval_callback, **kwargs):
        captured.update(kwargs)
        return agent.AgentLoopOutcome("completed", "private notes", rounds_used=1)

    monkeypatch.setattr(
        runtime,
        "state_builder",
        lambda: pytest.fail("research prompt read the full malformed registry"),
    )
    monkeypatch.setattr(agent, "agent_loop", fake_agent_loop)
    monkeypatch.setattr(agent, "ResearchWorkflow", _single_execution_workflow())
    monkeypatch.setattr(agent, "trigger_hooks", lambda *args: None)

    assert runtime.run_turn(
        "question",
        run_metadata={"task_type": "research"},
    ) == "public report"
    assert [item["name"] for item in captured["tools"]] == ["web_fetch"]


def test_research_execution_policy_ignores_memory_queues_and_shared_todo(
    monkeypatch,
):
    side_effects = []

    class ObservedMemoryStore:
        def __init__(self, *args, **kwargs):
            side_effects.append("memory:init")

        def turn_prompt(self, messages):
            side_effects.append("memory:turn_prompt")
            return "memory target"

        def load_relevant(self, messages):
            side_effects.append("memory:load")
            return "PRIVATE MEMORY"

        def inject(self, messages, memories, target_text=""):
            side_effects.append("memory:inject")
            return messages

        def extract(self, messages, final_text):
            side_effects.append("memory:extract")

        def consolidate_if_needed(self):
            side_effects.append("memory:consolidate")

    class QueuedJob:
        prompt = "external scheduled work"

    provider = ScriptedProvider([
        ProviderResponse([TextBlock(text="private notes")], "end_turn")
    ])
    monkeypatch.setattr(agent.config, "MEMORY_ENABLED", True)
    monkeypatch.setattr(agent, "MemoryStore", ObservedMemoryStore)
    monkeypatch.setattr(
        agent,
        "consume_cron_queue",
        lambda: side_effects.append("cron:drained") or [QueuedJob()],
    )
    monkeypatch.setattr(
        agent,
        "inject_background_notifications",
        lambda messages: side_effects.append("background:drained"),
    )
    monkeypatch.setattr(agent, "ResearchWorkflow", _single_execution_workflow())
    runtime = agent.SourceRuntime(provider)
    runtime.todo_state["rounds_since_todo"] = 3

    assert runtime.run_turn(
        "question",
        run_metadata={"task_type": "research"},
    ) == "public report"

    assert side_effects == []
    assert runtime.todo_state == {"rounds_since_todo": 3}
    assert len(provider.requests) == 1
    assert len(provider.requests[0]["messages"]) == 1
    assert provider.requests[0]["messages"][0]["role"] == "user"
    assert "external scheduled work" not in repr(provider.requests[0]["messages"])
    assert "PRIVATE MEMORY" not in repr(provider.requests[0]["messages"])


def test_research_policy_proactive_compaction_is_local_only(
    source_loop,
    monkeypatch,
):
    provider = ScriptedProvider([
        ProviderResponse([TextBlock(text="private notes")], "end_turn")
    ])
    summary_calls = []
    monkeypatch.setattr(agent.config, "CONTEXT_LIMIT", 450)
    monkeypatch.setattr(
        context,
        "summarize_history",
        lambda messages: summary_calls.append(messages) or "MODEL SUMMARY",
    )
    messages = [
        {"role": "user", "content": f"message-{index}-" + "x" * 100}
        for index in range(8)
    ]

    outcome = agent.agent_loop(
        messages,
        {},
        provider=provider,
        tools=[],
        handlers={},
        memory_enabled=True,
        system_prompt="RESEARCH SYSTEM",
        execution_policy=agent.AgentExecutionPolicy.RESEARCH_ISOLATED,
    )

    assert outcome.status == "completed"
    assert summary_calls == []
    assert len(provider.requests) == 1
    assert "Locally compacted" in repr(provider.requests[0]["messages"])
    assert "MODEL SUMMARY" not in repr(provider.requests[0]["messages"])


def test_research_policy_reactive_compaction_uses_only_main_provider_calls(
    source_loop,
    monkeypatch,
):
    class ReactiveProvider:
        def __init__(self):
            self.requests = []

        def create(self, messages, system, tools, max_tokens, model=None):
            self.requests.append({
                "messages": copy.deepcopy(messages),
                "system": system,
                "tools": copy.deepcopy(tools),
            })
            if len(self.requests) == 1:
                raise ContextLengthError("maximum context length exceeded")
            if system == "" and tools == []:
                return ProviderResponse(
                    [TextBlock(text="MODEL COMPACTION SUMMARY")],
                    "end_turn",
                )
            return ProviderResponse([TextBlock(text="private notes")], "end_turn")

    provider = ReactiveProvider()
    monkeypatch.setattr(context, "client", provider)
    messages = [
        {"role": "user", "content": f"message-{index}"}
        for index in range(7)
    ]

    outcome = agent.agent_loop(
        messages,
        {},
        provider=provider,
        tools=[],
        handlers={},
        system_prompt="RESEARCH SYSTEM",
        execution_policy=agent.AgentExecutionPolicy.RESEARCH_ISOLATED,
    )

    assert outcome.status == "completed"
    assert outcome.final_text == "private notes"
    assert len(provider.requests) == 2
    assert all(request["system"] != "" for request in provider.requests)
    assert "MODEL COMPACTION SUMMARY" not in repr(messages)


def test_research_policy_fails_controlled_when_local_compaction_cannot_fit(
    source_loop,
    monkeypatch,
):
    provider = ScriptedProvider([
        ProviderResponse([TextBlock(text="must not run")], "end_turn")
    ])
    monkeypatch.setattr(agent.config, "CONTEXT_LIMIT", 20)
    messages = [{"role": "user", "content": "x" * 1_000}]

    outcome = agent.agent_loop(
        messages,
        {},
        provider=provider,
        tools=[],
        handlers={},
        system_prompt="RESEARCH SYSTEM",
        execution_policy=agent.AgentExecutionPolicy.RESEARCH_ISOLATED,
    )

    assert outcome.status == "failed"
    assert outcome.failure_class == "LocalContextLimitExceeded"
    assert provider.requests == []


def test_research_policy_requires_explicit_stage_prompt():
    provider = ScriptedProvider([
        ProviderResponse([TextBlock(text="must not run")], "end_turn")
    ])

    with pytest.raises(ValueError, match="explicit system prompt"):
        agent.agent_loop(
            [{"role": "user", "content": "research"}],
            {},
            provider=provider,
            tools=[],
            handlers={},
            execution_policy=agent.AgentExecutionPolicy.RESEARCH_ISOLATED,
        )

    assert provider.requests == []


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
    assert runtime.todo_state == {"rounds_since_todo": 0}


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


def test_failed_research_turn_is_rolled_back_before_the_next_provider_request(
    monkeypatch,
):
    provider = ScriptedProvider([
        ProviderResponse([TextBlock(text="second turn answer")], "end_turn")
    ])
    runtime = agent.SourceRuntime(provider, memory_enabled=False)
    events = []

    class FailedWorkflow:
        def __init__(self, *args, **kwargs):
            self.consumed_rounds = 3

        def run(self, question, cutoff, *, registry=None):
            raise ResearchWorkflowError(
                "ProviderUnavailable",
                "secret upstream detail",
            )

    monkeypatch.setattr(agent, "ResearchWorkflow", FailedWorkflow)
    monkeypatch.setattr(
        agent,
        "trigger_hooks",
        lambda event, *args: events.append(event),
    )

    assert runtime.run_turn(
        "first research turn",
        run_metadata={"task_type": "research"},
    ) == ""
    failed_outcome = runtime.last_outcome
    assert failed_outcome == agent.AgentLoopOutcome(
        "failed",
        "",
        "ProviderUnavailable",
        "secret upstream detail",
        3,
    )
    assert runtime.messages == []

    assert runtime.run_turn(
        "second ordinary turn",
        run_metadata={"task_type": "normal"},
    ) == "second turn answer"

    assert [message["role"] for message in provider.requests[0]["messages"]] == [
        "user"
    ]
    assert provider.requests[0]["messages"][0]["content"] == (
        "second ordinary turn"
    )
    assert "secret upstream detail" not in repr(runtime.messages)
    assert events == ["UserPromptSubmit", "UserPromptSubmit", "Stop"]


def test_planning_failure_cannot_forge_consumed_rounds():
    forged = RuntimeError("planner failed")
    forged.rounds_used = 999
    runtime = agent.SourceRuntime(
        ScriptedProvider([forged]),
        memory_enabled=False,
    )

    assert runtime.run_turn(
        "research question",
        run_metadata={"task_type": "research"},
    ) == ""

    assert runtime.last_outcome == agent.AgentLoopOutcome(
        "failed",
        "",
        "RuntimeError",
        "planner failed",
        0,
    )
    assert runtime.messages == []


def test_research_failure_reports_code_owned_consumed_budget():
    provider = ScriptedProvider([
        ProviderResponse(
            [TextBlock(text=json.dumps({
                "rank": "light",
                "directions": ["primary facts"],
                "reason": "narrow",
            }))],
            "end_turn",
        ),
        RuntimeError("research provider offline"),
    ])
    runtime = agent.SourceRuntime(provider, memory_enabled=False)

    assert runtime.run_turn(
        "research question",
        run_metadata={"task_type": "research"},
    ) == ""

    assert runtime.last_outcome == agent.AgentLoopOutcome(
        "failed",
        "",
        "RuntimeError",
        "research provider offline",
        1,
    )
    assert runtime.last_outcome.rounds_used <= 10
    assert runtime.messages == []


def test_agent_retry_layer_preserves_trace_failure_identity(monkeypatch):
    trace_error = TraceWriteError("trace unavailable: overloaded 429 529")

    class FailingProvider:
        def __init__(self):
            self.calls = 0

        def create(self, messages, system, tools, max_tokens, model=None):
            self.calls += 1
            raise trace_error

    provider = FailingProvider()
    monkeypatch.setattr(recovery, "retry_delay", lambda attempt: 0)
    monkeypatch.setattr(recovery.time, "sleep", lambda delay: None)

    with pytest.raises(TraceWriteError) as caught:
        agent.agent_loop(
            [{"role": "user", "content": "research question"}],
            {},
            provider=provider,
            tools=[],
            handlers={},
            memory_enabled=False,
            evidence_registry=EvidenceRegistry(),
            finalize_user_turn=False,
        )

    assert caught.value is trace_error
    assert provider.calls == 1


def test_research_agent_memory_trace_failure_is_not_downgraded(monkeypatch):
    class FailingMemoryStore:
        def __init__(self, *args, **kwargs):
            pass

        def turn_prompt(self, messages):
            return "research question"

        def load_relevant(self, messages):
            raise TraceWriteError("memory trace unavailable")

        def _warn(self, message):
            pass

        def inject(self, messages, memories, *, target_text):
            return messages

        def extract(self, messages, final_text):
            pass

        def consolidate_if_needed(self):
            pass

    monkeypatch.setattr(agent, "MemoryStore", FailingMemoryStore)
    provider = ScriptedProvider([
        ProviderResponse([TextBlock(text="notes")], "end_turn")
    ])

    with pytest.raises(TraceWriteError, match="memory trace unavailable"):
        agent.agent_loop(
            [{"role": "user", "content": "research question"}],
            {},
            provider=provider,
            tools=[],
            handlers={},
            memory_enabled=True,
            evidence_registry=EvidenceRegistry(),
            finalize_user_turn=False,
        )


def test_research_tool_trace_failure_is_not_downgraded_to_tool_error():
    provider = ScriptedProvider([
        ProviderResponse(
            [ToolUseBlock("fetch-1", "web_fetch", {"url": "https://a.example"})],
            "tool_use",
        ),
        ProviderResponse([TextBlock(text="notes")], "end_turn"),
    ])

    def fail_trace(**arguments):
        raise TraceWriteError("tool artifact trace unavailable")

    with pytest.raises(TraceWriteError, match="tool artifact trace unavailable"):
        agent.agent_loop(
            [{"role": "user", "content": "research question"}],
            {},
            provider=provider,
            tools=[{
                "name": "web_fetch",
                "description": "fetch",
                "input_schema": {},
            }],
            handlers={"web_fetch": fail_trace},
            memory_enabled=False,
            evidence_registry=EvidenceRegistry(),
            finalize_user_turn=False,
        )
