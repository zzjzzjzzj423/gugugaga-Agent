from __future__ import annotations

import copy

import pytest

from simple_cc import agent, config, context, subagents, teams, tools
from simple_cc.provider import (
    ContextLengthError,
    ProviderResponse,
    TextBlock,
    ToolUseBlock,
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
