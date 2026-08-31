from __future__ import annotations

from types import SimpleNamespace

import pytest

import gugugaga.config as config
import gugugaga.hooks as hooks
import gugugaga.subagents as subagents
from gugugaga.permissions import PermissionPolicy
from gugugaga.provider import ProviderResponse, TextBlock, ToolUseBlock


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


def test_subagent_bash_uses_main_permission_policy(tmp_path, monkeypatch):
    class ScriptedProvider:
        def __init__(self):
            self.responses = [
                ProviderResponse(
                    content=[
                        ToolUseBlock(
                            id="toolu_bash",
                            name="bash",
                            input={"command": "echo unsafe> subagent-marker.txt"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content=[TextBlock("The unsafe command was denied.")],
                    stop_reason="end_turn",
                ),
            ]
            self.requests = []

        def create(self, messages, system, tools, max_tokens, model=None):
            self.requests.append([dict(message) for message in messages])
            return self.responses.pop(0)

    provider = ScriptedProvider()
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(subagents, "client", provider)
    monkeypatch.setattr(subagents, "_permissions", PermissionPolicy())
    monkeypatch.setattr(subagents, "_approval_callback", None)
    monkeypatch.setattr(subagents, "_context_parent_resolver", None)

    assert subagents.spawn_subagent("Run the command") == (
        "The unsafe command was denied."
    )
    assert not (tmp_path / "subagent-marker.txt").exists()
    assert provider.requests[1][-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_bash",
            "content": (
                "Permission denied for tool 'bash'. Choose a safer approach."
            ),
        }
    ]


def test_subagent_round_limit_is_an_explicit_failure(monkeypatch):
    class ToolLoopProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            return ProviderResponse(
                content=[
                    ToolUseBlock(
                        id=f"toolu_{len(messages)}",
                        name="read_file",
                        input={"path": "missing.txt"},
                    )
                ],
                stop_reason="tool_use",
            )

    monkeypatch.setattr(subagents, "client", ToolLoopProvider())
    monkeypatch.setattr(subagents, "_context_parent_resolver", None)
    monkeypatch.setattr(subagents, "_max_rounds", 2)

    with pytest.raises(RuntimeError, match=r"exceeded maximum rounds \(2\)"):
        subagents.spawn_subagent("Never finish")
