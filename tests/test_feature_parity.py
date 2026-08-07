from simple_cc.__main__ import build_runtime
from simple_cc.config import Settings
from tests.fakes import ScriptedProvider


EXPECTED_TOOLS = {
    "bash", "read_file", "write_file", "edit_file", "glob",
    "todo_write", "subagent", "load_skill", "compact", "remember",
    "create_task", "list_tasks", "get_task", "claim_task", "complete_task",
    "background_run", "schedule_cron", "list_crons", "cancel_cron",
    "spawn_teammate", "send_message", "check_inbox",
    "request_shutdown", "request_plan", "review_plan",
}


def test_s01_through_s17_tools_are_registered(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    app = build_runtime(Settings.from_env(tmp_path), provider=ScriptedProvider())
    assert EXPECTED_TOOLS <= {spec.name for spec in app.runtime.registry.specs()}
