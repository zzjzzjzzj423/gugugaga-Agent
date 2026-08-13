from __future__ import annotations

import json

import pytest

from eval.run_task import TaskInput, execute_task, load_task_input
from simple_cc.models import ModelResponse
from tests.fakes import ScriptedProvider


def test_load_task_input_validates_required_fields_and_cutoff(tmp_path):
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task_id": "task-1",
                "question": "Research",
                "cutoff": "2025-05-01",
                "benchmark": "financegym",
                "task_type": "research_analysis",
            }
        ),
        encoding="utf-8",
    )
    task = load_task_input(path)
    assert task.task_id == "task-1"
    path.write_text(path.read_text().replace("2025-05-01", "05/01/2025"))
    with pytest.raises(ValueError, match="cutoff"):
        load_task_input(path)


def test_execute_task_writes_completed_run_after_clean_shutdown(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    workspace = run_dir / "agent_workspace"
    task = TaskInput(
        "run-1",
        "task-1",
        "Research",
        "2025-05-01",
        "financegym",
        "research_analysis",
    )
    exit_code = execute_task(
        task,
        run_dir,
        workspace,
        ScriptedProvider([ModelResponse("answer")]),
        model="test-model",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["status"] == "completed"
    assert (run_dir / "final_answer.txt").read_text(encoding="utf-8") == "answer"
    assert manifest["worker_pid"] > 0


def test_execute_task_marks_provider_failure_without_final_answer(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = TaskInput("run-1", "task-1", "Research", None, "test", "research")
    exit_code = execute_task(
        task,
        run_dir,
        run_dir / "agent_workspace",
        ScriptedProvider([RuntimeError("failed")]),
        model="test-model",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code != 0
    assert manifest["status"] == "failed"
    assert not (run_dir / "final_answer.txt").exists()
