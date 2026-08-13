from __future__ import annotations

import pytest

from simple_cc.benchmark import (
    BenchmarkOptions,
    build_benchmark_runtime,
    validate_clean_workspace,
)
from simple_cc.models import ModelResponse
from simple_cc.trace import TraceRecorder
from tests.fakes import ScriptedProvider


def test_validate_clean_workspace_rejects_dirty_or_outside_paths(tmp_path):
    run_dir = tmp_path / "run"
    workspace = run_dir / "agent_workspace"
    workspace.mkdir(parents=True)
    (workspace / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        validate_clean_workspace(run_dir, workspace)
    with pytest.raises(ValueError, match="inside"):
        validate_clean_workspace(run_dir, tmp_path / "outside")


def test_benchmark_runtime_filters_stateful_tools_and_disables_memory(tmp_path):
    run_dir = tmp_path / "run"
    workspace = run_dir / "agent_workspace"
    run_dir.mkdir()
    recorder = TraceRecorder(run_dir, run_id="run-1")
    provider = ScriptedProvider([ModelResponse("done")])

    session = build_benchmark_runtime(
        run_dir=run_dir,
        workspace=workspace,
        provider=provider,
        recorder=recorder,
        options=BenchmarkOptions(),
    )
    try:
        names = {item["name"] for item in session.runtime.tool_definitions}
        assert "web_search" in names
        assert "pdf_fetch" in names
        assert not names.intersection(
            {
                "task",
                "schedule_cron",
                "list_crons",
                "cancel_cron",
                "spawn_teammate",
                "send_message",
                "check_inbox",
                "request_shutdown",
                "request_plan",
                "review_plan",
            }
        )
        assert session.runtime.memory_enabled is False
        assert session.runtime.approval_callback is None
    finally:
        outcome = session.close()
    assert outcome.stopped is True
