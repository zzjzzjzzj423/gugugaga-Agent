from __future__ import annotations

import json

import pytest

from eval.run_task import TaskInput, execute_task, load_task_input, main
from simple_cc.eval_metrics import derive_metrics, validate_trace
from simple_cc.models import ModelResponse, ToolCall
from simple_cc.telemetry import capture_tool_artifact
from simple_cc.trace import read_trace_lines
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


def test_worker_cli_does_not_dirty_workspace_before_isolation_check(
    tmp_path, monkeypatch
):
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "run_id": "run-cli",
                "task_id": "task-cli",
                "question": "Research",
                "cutoff": "2025-05-01",
                "benchmark": "test",
                "task_type": "research",
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    workspace = run_dir / "agent_workspace"
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    monkeypatch.setattr(
        "eval.run_task.SiliconFlowProvider",
        lambda settings: ScriptedProvider([ModelResponse("answer")]),
    )

    exit_code = main(
        [
            "--task-input",
            str(task_path),
            "--run-dir",
            str(run_dir),
            "--workspace",
            str(workspace),
        ]
    )

    assert exit_code == 0
    assert (run_dir / "final_answer.txt").read_text(encoding="utf-8") == "answer"


def test_offline_worker_search_fetch_answer_trace_is_self_consistent(
    tmp_path, monkeypatch
):
    cutoff = "2025-05-01"
    url = "https://example.com/report?a=1&b=2"
    seen_arguments = []

    def search(**arguments):
        seen_arguments.append(("search", arguments))
        return json.dumps(
            {
                "ok": True,
                "operation": "search",
                "query": arguments["query"],
                "cutoff": arguments["cutoff"],
                "results": [{"url": url, "title": "Report"}],
            }
        )

    def fetch(**arguments):
        seen_arguments.append(("fetch", arguments))
        capture_tool_artifact(
            b"<html>accepted evidence</html>",
            media_type="text/html",
            source=url,
            suffix=".html",
        )
        return json.dumps(
            {
                "ok": True,
                "operation": "fetch",
                "url": url,
                "cutoff": arguments["cutoff"],
                "published_at": "2025-04-01",
                "date_status": "known",
                "content": "accepted evidence",
            }
        )

    monkeypatch.setattr(
        "simple_cc.benchmark._benchmark_tool_tables",
        lambda options: (
            [
                {"name": "web_search", "description": "search", "input_schema": {}},
                {"name": "web_fetch", "description": "fetch", "input_schema": {}},
            ],
            {"web_search": search, "web_fetch": fetch},
        ),
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                [ToolCall("search-1", "web_search", {"query": "report"})],
                "tool_calls",
            ),
            ModelResponse(
                "",
                [ToolCall("fetch-1", "web_fetch", {"url": url})],
                "tool_calls",
            ),
            ModelResponse(f"Answer: {url}", [], "stop"),
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = TaskInput(
        "run-e2e", "task-e2e", "Research", cutoff, "test", "research"
    )

    exit_code = execute_task(
        task,
        run_dir,
        run_dir / "agent_workspace",
        provider,
        model="test-model",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows, incomplete = read_trace_lines(run_dir / "trajectory.jsonl")
    source = next(row for row in rows if row["event_type"] == "source_registered")
    final = next(row for row in rows if row["event_type"] == "final_answer")
    llm = [row for row in rows if row["event_type"] == "llm_response"]
    tools = [row for row in rows if row["event_type"] == "tool_result"]
    valid, errors = validate_trace(run_dir)
    metrics = derive_metrics(run_dir)

    assert exit_code == 0
    assert manifest["status"] == "completed"
    assert manifest["pit_mode"] == "non_strict_live_web"
    assert incomplete is False
    assert valid is True, errors
    assert seen_arguments == [
        ("search", {"query": "report", "cutoff": cutoff}),
        ("fetch", {"url": url, "cutoff": cutoff}),
    ]
    assert source["payload"]["source_id"] in final["payload"]["matched_source_ids"]
    assert all(row["payload"]["latency_ms"] >= 0 for row in llm + tools)
    assert metrics.model_calls == 3
    assert metrics.searches == 1
    assert metrics.fetches == 1
    assert metrics.all_in_prompt_tokens is None
