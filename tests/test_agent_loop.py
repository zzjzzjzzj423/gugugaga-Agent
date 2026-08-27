from gugugaga.agent import AgentRuntime
from gugugaga import config
from gugugaga.background import BackgroundManager, CronScheduler
from gugugaga.context import ContextManager
from gugugaga.hooks import HookEvent, HookManager
from gugugaga.models import ModelResponse, ToolCall
from gugugaga.permissions import PermissionPolicy
from gugugaga.prompts import PromptAssembler
from gugugaga.provider import ContextLengthError
from gugugaga.tools import ToolRegistry, WorkspaceTools
from tests.fakes import ScriptedProvider


def make_runtime(tmp_path, provider, approval=lambda _: True):
    registry = ToolRegistry()
    WorkspaceTools(tmp_path).register_into(registry)
    return AgentRuntime(
        provider=provider,
        registry=registry,
        hooks=HookManager(),
        permissions=PermissionPolicy(),
        context=ContextManager(tmp_path / ".gugugaga/outputs", tmp_path / ".gugugaga/transcripts"),
        prompts=PromptAssembler(),
        state_builder=lambda: {"workspace": str(tmp_path), "tools": "files"},
        background=BackgroundManager(),
        cron=CronScheduler(tmp_path / ".gugugaga/cron.json"),
        approval_callback=approval,
        max_rounds=5,
    )


def test_tool_call_result_returns_to_model(tmp_path):
    provider = ScriptedProvider([
        ModelResponse("", [ToolCall("c1", "write_file", {"path": "hello.txt", "content": "hi"})], "tool_calls"),
        ModelResponse("Created hello.txt", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    assert runtime.run_turn("create hello.txt") == "Created hello.txt"
    assert (tmp_path / "hello.txt").read_text() == "hi"
    assert provider.requests[-1]["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "c1",
                "content": "Wrote 2 characters to hello.txt",
            }
        ],
    }


def test_denied_command_becomes_tool_result(tmp_path):
    provider = ScriptedProvider([
        ModelResponse("", [ToolCall("c1", "bash", {"command": "git reset --hard"})], "tool_calls"),
        ModelResponse("I will use a safer approach.", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider, approval=lambda _: False)
    assert "safer" in runtime.run_turn("reset it")
    assert "denied" in (
        provider.requests[-1]["messages"][-1]["content"][0]["content"].lower()
    )


def test_pre_tool_hook_can_block_execution(tmp_path):
    provider = ScriptedProvider([
        ModelResponse("", [ToolCall("c1", "write_file", {
            "path": "blocked.txt", "content": "no"})], "tool_calls"),
        ModelResponse("Used another approach", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    runtime.hooks.register(
        HookEvent.PRE_TOOL_USE, lambda call: "Blocked by project hook"
    )
    assert "another" in runtime.run_turn("write it")
    assert not (tmp_path / "blocked.txt").exists()
    assert (
        provider.requests[-1]["messages"][-1]["content"][0]["content"]
        == "Blocked by project hook"
    )


def test_context_error_compacts_once_then_retries(tmp_path):
    provider = ScriptedProvider([
        ContextLengthError("too long"),
        ModelResponse("Recovered context summary", [], "stop"),
        ModelResponse("Recovered", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    runtime.messages.extend({"role": "user", "content": str(i)} for i in range(5))
    assert runtime.run_turn("continue") == "Recovered"
    assert len(provider.requests) == 3


def test_compact_tool_forces_transcript_archive(tmp_path):
    provider = ScriptedProvider([
        ModelResponse("", [ToolCall("c1", "compact", {})], "tool_calls"),
        ModelResponse("Manual summary with goals and decisions", [], "stop"),
        ModelResponse("Compacted", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    runtime.registry.register("compact", "Compact", {"type": "object"}, lambda: "requested")
    runtime.messages.extend(
        {"role": "user", "content": f"{i}-" + "x" * 1_000}
        for i in range(5)
    )
    assert runtime.run_turn("compact now") == "Compacted"
    assert list((tmp_path / ".gugugaga/transcripts").glob("*.jsonl"))
    assert "Manual summary with goals and decisions" in (
        runtime.context_coordinator.project(runtime.messages)[0]["content"]
    )


def test_automatic_compaction_uses_provider_summary(tmp_path, monkeypatch):
    provider = ScriptedProvider([
        ModelResponse("A concise history summary", [], "stop"),
        ModelResponse("Done", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    monkeypatch.setattr(config, "CONTEXT_LIMIT", 100)
    runtime.messages.extend(
        {"role": "user", "content": f"{i}-" + "x" * 200}
        for i in range(4)
    )
    assert runtime.run_turn("continue") == "Done"
    assert provider.requests[0]["tools"] == []
    assert "A concise history summary" in provider.requests[1]["messages"][0]["content"]


def test_pending_cron_prompt_can_wake_agent_without_user_turn(tmp_path):
    provider = ScriptedProvider([ModelResponse("Scheduled work handled", [], "stop")])
    runtime = make_runtime(tmp_path, provider)
    runtime.cron.schedule("* * * * *", "run scheduled check", recurring=False)
    runtime.cron.fire_due()
    assert runtime.run_pending() == "Scheduled work handled"
    assert "run scheduled check" in provider.requests[0]["messages"][-1]["content"]
