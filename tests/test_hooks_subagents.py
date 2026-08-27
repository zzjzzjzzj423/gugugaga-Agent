from __future__ import annotations

from types import SimpleNamespace

import gugugaga.config as config
import gugugaga.hooks as hooks
import gugugaga.subagents as subagents
from gugugaga.provider import ProviderResponse, TextBlock


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
