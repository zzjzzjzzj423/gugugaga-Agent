from __future__ import annotations

import copy

import pytest

from gugugaga import agent, config, context, subagents, teams, tools
from gugugaga.provider import (
    ContextLengthError,
    ProviderResponse,
    TextBlock,
    ToolUseBlock,
)
from gugugaga.context_modes import SessionContextConfig, SessionContextCoordinator


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


def test_completed_turn_extracts_memory_from_pre_compaction_snapshot(
    source_loop, monkeypatch
):
    provider = ScriptedProvider(
        [
            ProviderResponse(
                content=[TextBlock(text=f"Finished {index}.")],
                stop_reason="end_turn",
            )
            for index in range(10)
        ]
    )
    install(provider, monkeypatch)
    captured = {}

    class RecordingMemoryStore:
        def __init__(self, directory, provider=None):
            pass

        def extract_batch(self, turns):
            captured["turns"] = copy.deepcopy(turns)
            return 0

    def compact_in_place(messages):
        messages[:] = [{"role": "user", "content": "[Compacted]"}]
        return messages

    monkeypatch.setattr(agent, "MemoryStore", RecordingMemoryStore)
    monkeypatch.setattr(agent, "prepare_context", compact_in_place)
    memory_state = {"pending_turns": []}

    for index in range(9):
        messages = [
            {
                "role": "user",
                "content": f"Always keep original constraint {index}.",
            }
        ]
        agent.agent_loop(messages, {}, memory_state=memory_state)
        assert "turns" not in captured

    messages = [
        {"role": "user", "content": "Always keep original constraint 9."}
    ]
    agent.agent_loop(messages, {}, memory_state=memory_state)

    assert len(captured["turns"]) == 10
    first_snapshot, first_text = captured["turns"][0]
    assert first_text == "Finished 0."
    assert first_snapshot[0]["content"] == "Always keep original constraint 0."
    assert first_snapshot[1]["role"] == "assistant"
    last_snapshot, last_text = captured["turns"][-1]
    assert last_text == "Finished 9."
    assert last_snapshot[0]["content"] == "Always keep original constraint 9."
    assert last_snapshot[1]["role"] == "assistant"
    assert messages[0]["content"] == "Always keep original constraint 9."
    assert memory_state["pending_turns"] == []


def test_stop_reason_is_authoritative_for_turn_completion(
    source_loop, tmp_path, monkeypatch
):
    provider = ScriptedProvider([
        ProviderResponse(
            content=[
                ToolUseBlock(
                    id="unexpected-tool",
                    name="write_file",
                    input={"path": "should-not-exist.txt", "content": "no"},
                )
            ],
            stop_reason="end_turn",
        )
    ])
    install(provider, monkeypatch)

    agent.agent_loop([{"role": "user", "content": "Finish now"}], {})

    assert not (tmp_path / "should-not-exist.txt").exists()


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
            ProviderResponse(
                content=[TextBlock(text="Earlier work.")],
                stop_reason="end_turn",
            ),
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

    assert len(provider.requests) == 3
    assert len(list((tmp_path / ".transcripts").glob("*.jsonl"))) == 1
    assert messages[-1] == {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": (
                    "[CONTEXT_RECOVERY_EXHAUSTED] The Provider still rejected "
                    "the rebuilt context. History preserved: True. Next: start "
                    "a new session or use a model with a larger context window."
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
    messages = [
        {"role": "user", "content": "Old goal " + "x" * 2_000},
        {"role": "assistant", "content": [{"type": "text", "text": "Old work " + "y" * 2_000}]},
        {"role": "user", "content": "Compact now " + "z" * 2_000},
    ]
    memory_state = {"pending_turns": []}

    agent.agent_loop(messages, {}, memory_state=memory_state)

    assert messages[0]["content"].startswith("Old goal ")
    coordinator = memory_state["context_coordinator"]
    assert coordinator.status()["successful_compactions"] == 1
    assert coordinator.project(messages)[0]["content"].startswith("[Compacted]")
    assert any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict)
            and block.get("tool_use_id") == "toolu_compact"
            for block in message["content"]
        )
        for message in messages
    )


def test_two_source_runtimes_bind_files_and_memory_to_their_own_workspace(
    source_loop, tmp_path, monkeypatch
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    (first_root / ".memory").mkdir(parents=True)
    (second_root / ".memory").mkdir(parents=True)
    (first_root / ".memory" / "MEMORY.md").write_text("first-memory")
    (second_root / ".memory" / "MEMORY.md").write_text("second-memory")
    provider = ScriptedProvider(
        [
            ProviderResponse(
                content=[ToolUseBlock(id="first-write", name="write_file", input={"path": "first.txt", "content": "one"})],
                stop_reason="tool_use",
            ),
            ProviderResponse(content=[TextBlock(text="first done")], stop_reason="end_turn"),
            ProviderResponse(
                content=[ToolUseBlock(id="second-write", name="write_file", input={"path": "second.txt", "content": "two"})],
                stop_reason="tool_use",
            ),
            ProviderResponse(content=[TextBlock(text="second done")], stop_reason="end_turn"),
        ]
    )
    install(provider, monkeypatch)

    def runtime(root):
        coordinator = SessionContextCoordinator(
            SessionContextConfig.parse(),
            summary_callback=agent._summary_callback(provider),
            workspace=root,
        )
        return agent.SourceRuntime(provider, context_coordinator=coordinator)

    first = runtime(first_root)
    second = runtime(second_root)
    config.configure_workspace(second_root)

    assert first.run_turn("write first") == "first done"
    assert second.run_turn("write second") == "second done"
    assert (first_root / "first.txt").read_text() == "one"
    assert (second_root / "second.txt").read_text() == "two"
    assert not (first_root / "second.txt").exists()
    assert not (second_root / "first.txt").exists()
    assert "first-memory" in first.context["memories"]
    assert "second-memory" in second.context["memories"]


def test_source_runtime_new_session_resets_transient_state_only(
    source_loop, tmp_path, monkeypatch
):
    provider = ScriptedProvider([])
    install(provider, monkeypatch)
    coordinator = SessionContextCoordinator(
        SessionContextConfig.parse(),
        summary_callback=agent._summary_callback(provider),
        workspace=tmp_path,
    )
    runtime = agent.SourceRuntime(provider, context_coordinator=coordinator)
    original_session = runtime.context_coordinator.session_id
    original_recording = runtime.recording
    original_memory_service = runtime.memory_service
    runtime.messages.append({"role": "user", "content": "old conversation"})
    runtime.memory_state["pending_turns"].append("turn_old")

    try:
        new_session = runtime.start_new_session("hermes")

        assert new_session != original_session
        assert runtime.context_coordinator.session_id == new_session
        assert runtime.context_coordinator.status()["lifecycle"] == "configuring"
        assert runtime.context_coordinator.status()["mode"] == "hermes"
        assert runtime.context_coordinator.status()["locked"] is False
        assert runtime.messages == []
        assert runtime.memory_state == {"pending_turns": []}
        assert runtime.recording is original_recording
        assert runtime.memory_service is original_memory_service

        restored_messages = [
            {"role": "user", "content": "saved question"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "saved answer"}],
            },
        ]
        resumed_session = runtime.resume_session(
            "session_saved", restored_messages, "pi"
        )
        restored_messages[0]["content"] = "mutated outside runtime"

        assert resumed_session == "session_saved"
        assert runtime.context_coordinator.session_id == "session_saved"
        assert runtime.context_coordinator.status()["mode"] == "pi"
        assert runtime.context_coordinator.status()["locked"] is True
        assert runtime.messages[0]["content"] == "saved question"
        assert runtime.recording is original_recording
        assert runtime.memory_service is original_memory_service
    finally:
        runtime.memory_service.close(1)
        runtime.context_coordinator.close()
