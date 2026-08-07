from simple_cc.agent import AgentRuntime
from simple_cc.background import BackgroundManager, CronScheduler
from simple_cc.context import ContextManager
from simple_cc.hooks import HookManager
from simple_cc.models import ModelResponse, ToolCall
from simple_cc.permissions import PermissionPolicy
from simple_cc.prompts import PromptAssembler
from simple_cc.provider import ContextLengthError
from simple_cc.tools import ToolRegistry, WorkspaceTools
from tests.fakes import ScriptedProvider


def make_runtime(tmp_path, provider, approval=lambda _: True):
    registry = ToolRegistry()
    WorkspaceTools(tmp_path).register_into(registry)
    return AgentRuntime(
        provider=provider,
        registry=registry,
        hooks=HookManager(),
        permissions=PermissionPolicy(),
        context=ContextManager(tmp_path / ".simple_cc/outputs", tmp_path / ".simple_cc/transcripts"),
        prompts=PromptAssembler(),
        state_builder=lambda: {"workspace": str(tmp_path), "tools": "files"},
        background=BackgroundManager(),
        cron=CronScheduler(tmp_path / ".simple_cc/cron.json"),
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
    assert provider.requests[-1]["messages"][-1]["role"] == "tool"
    assert provider.requests[-1]["messages"][-1]["tool_call_id"] == "c1"


def test_denied_command_becomes_tool_result(tmp_path):
    provider = ScriptedProvider([
        ModelResponse("", [ToolCall("c1", "bash", {"command": "git reset --hard"})], "tool_calls"),
        ModelResponse("I will use a safer approach.", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider, approval=lambda _: False)
    assert "safer" in runtime.run_turn("reset it")
    assert "denied" in provider.requests[-1]["messages"][-1]["content"].lower()


def test_context_error_compacts_once_then_retries(tmp_path):
    provider = ScriptedProvider([
        ContextLengthError("too long"),
        ModelResponse("Recovered", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    runtime.messages.extend({"role": "user", "content": str(i)} for i in range(70))
    assert runtime.run_turn("continue") == "Recovered"
    assert len(provider.requests) == 2


def test_compact_tool_forces_transcript_archive(tmp_path):
    provider = ScriptedProvider([
        ModelResponse("", [ToolCall("c1", "compact", {})], "tool_calls"),
        ModelResponse("Compacted", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    runtime.registry.register("compact", "Compact", {"type": "object"}, lambda: "requested")
    runtime.messages.extend({"role": "user", "content": str(i)} for i in range(5))
    assert runtime.run_turn("compact now") == "Compacted"
    assert list((tmp_path / ".simple_cc/transcripts").glob("*.json"))
