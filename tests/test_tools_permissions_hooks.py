from simple_cc.hooks import HookEvent, HookManager
from simple_cc.models import ToolCall
from simple_cc.permissions import PermissionDecision, PermissionPolicy
from simple_cc.tools import ToolRegistry, WorkspaceTools


def test_workspace_tools_write_read_edit_and_glob(tmp_path):
    tools = WorkspaceTools(tmp_path)
    assert tools.write_file("src/a.txt", "hello").startswith("Wrote")
    assert tools.read_file("src/a.txt") == "hello"
    assert tools.edit_file("src/a.txt", "hello", "world").startswith("Edited")
    assert tools.read_file("src/a.txt") == "world"
    assert "src/a.txt" in tools.glob("**/*.txt")


def test_read_rejects_parent_escape(tmp_path):
    assert WorkspaceTools(tmp_path).read_file("../secret.txt").startswith(
        "Error: path escapes workspace"
    )


def test_registry_executes_registered_handler():
    registry = ToolRegistry()
    registry.register("echo", "Echo", {"type": "object"}, lambda value: value)
    assert registry.execute("echo", {"value": "ok"}) == "ok"


def test_dangerous_commands_require_approval():
    policy = PermissionPolicy()
    for tool in ("bash", "background_run"):
        for command in ("rm -rf build", "git reset --hard", "sudo apt update"):
            assert policy.decide(ToolCall("1", tool, {"command": command})) is PermissionDecision.ASK


def test_all_shell_commands_require_explicit_approval():
    policy = PermissionPolicy()
    for tool in ("bash", "background_run"):
        call = ToolCall("1", tool, {"command": "python -c \"print('hello')\""})
        assert policy.decide(call) is PermissionDecision.ASK


def test_hook_manager_preserves_registration_order():
    hooks = HookManager()
    calls = []
    hooks.register(HookEvent.PRE_TOOL_USE, lambda **_: calls.append("first"))
    hooks.register(HookEvent.PRE_TOOL_USE, lambda **_: calls.append("second"))
    hooks.trigger(HookEvent.PRE_TOOL_USE, call=None)
    assert calls == ["first", "second"]
