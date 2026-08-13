from __future__ import annotations

from types import SimpleNamespace

import simple_cc.config as config
import simple_cc.hooks as hooks
import simple_cc.subagents as subagents
from simple_cc.provider import ProviderResponse, TextBlock, ToolUseBlock
from simple_cc.trace import RunContext, TraceRecorder, bind_run_context, current_run_context


def test_hooks_run_in_registration_order_and_stop_on_first_result(monkeypatch):
    ordered = {event: [] for event in hooks.HOOKS}
    monkeypatch.setattr(hooks, "HOOKS", ordered)
    calls = []

    hooks.register_hook("PreToolUse", lambda block: calls.append("first"))

    def reject(block):
        calls.append("second")
        return "rejected"

    hooks.register_hook("PreToolUse", reject)
    hooks.register_hook("PreToolUse", lambda block: calls.append("third"))

    assert hooks.trigger_hooks("PreToolUse", object()) == "rejected"
    assert calls == ["first", "second"]


def test_permission_hook_rejects_deny_list_and_workspace_escape(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)

    denied_bash = SimpleNamespace(
        name="bash", input={"command": "sudo shutdown now"}
    )
    denied_write = SimpleNamespace(
        name="write_file", input={"path": "../outside.txt"}
    )

    assert hooks.permission_hook(denied_bash) == (
        "Permission denied: 'sudo' is on the deny list"
    )
    assert hooks.permission_hook(denied_write) == (
        "Permission denied: path escapes workspace: ../outside.txt"
    )


def test_scripted_one_shot_subagent_returns_final_summary(monkeypatch):
    class ScriptedProvider:
        def __init__(self):
            self.requests = []

        def create(self, messages, system, tools, max_tokens, model=None):
            self.requests.append(
                {
                    "messages": [dict(message) for message in messages],
                    "system": system,
                    "tools": tools,
                    "max_tokens": max_tokens,
                    "model": model,
                }
            )
            return ProviderResponse(
                content=[TextBlock("Inspected the workspace and found no issues.")],
                stop_reason="end_turn",
            )

    provider = ScriptedProvider()
    monkeypatch.setattr(subagents, "client", provider)
    monkeypatch.setattr(subagents, "MODEL", "test-model")

    result = subagents.spawn_subagent("Inspect the workspace")

    assert result == "Inspected the workspace and found no issues."
    assert len(provider.requests) == 1
    assert provider.requests[0]["messages"] == [
        {"role": "user", "content": "Inspect the workspace"}
    ]
    assert provider.requests[0]["model"] == "test-model"
    assert provider.requests[0]["tools"] == subagents.SUB_TOOLS


def test_subagent_tool_execution_stays_in_child_trace_context(tmp_path, monkeypatch):
    class ScriptedProvider:
        def __init__(self):
            self.responses = [
                ProviderResponse(
                    [ToolUseBlock("read-1", "read_file", {"path": "x"})],
                    "tool_use",
                ),
                ProviderResponse([TextBlock("done")], "end_turn"),
            ]

        def create(self, *args, **kwargs):
            return self.responses.pop(0)

    observed = []

    def read_file(**arguments):
        observed.append(current_run_context().agent_id)
        return "contents"

    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    run = RunContext(recorder, "run-1", "task-1", None)
    monkeypatch.setattr(subagents, "client", ScriptedProvider())
    monkeypatch.setitem(subagents.SUB_HANDLERS, "read_file", read_file)

    with bind_run_context(run):
        assert subagents.spawn_subagent("inspect") == "done"

    rows = [
        __import__("json").loads(line)
        for line in recorder.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    tool_rows = [row for row in rows if row["event_type"].startswith("tool_")]
    assert observed and observed[0].startswith("subagent:")
    assert [row["event_type"] for row in tool_rows] == [
        "tool_requested",
        "tool_started",
        "tool_result",
    ]
    assert all(row["agent_id"].startswith("subagent:") for row in tool_rows)
