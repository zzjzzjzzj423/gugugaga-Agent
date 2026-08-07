from simple_cc.agent import AgentRuntime
from simple_cc.background import BackgroundManager, CronScheduler
from simple_cc.context import ContextManager
from simple_cc.hooks import HookEvent, HookManager
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
    assert "Blocked by project hook" in provider.requests[-1]["messages"][-1]["content"]


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
        ModelResponse("Compacted", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    runtime.registry.register("compact", "Compact", {"type": "object"}, lambda: "requested")
    runtime.messages.extend({"role": "user", "content": str(i)} for i in range(5))
    assert runtime.run_turn("compact now") == "Compacted"
    assert list((tmp_path / ".simple_cc/transcripts").glob("*.json"))


def test_automatic_compaction_uses_provider_summary(tmp_path):
    provider = ScriptedProvider([
        ModelResponse("A concise history summary", [], "stop"),
        ModelResponse("Done", [], "stop"),
    ])
    runtime = make_runtime(tmp_path, provider)
    runtime.context.max_messages = 3
    runtime.messages.extend({"role": "user", "content": str(i)} for i in range(4))
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
