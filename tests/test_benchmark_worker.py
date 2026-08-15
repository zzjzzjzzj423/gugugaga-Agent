from __future__ import annotations

import json
import hashlib

import pytest

from eval.run_task import TaskInput, execute_task, load_task_input, main
from simple_cc.eval_metrics import derive_metrics, validate_trace
from simple_cc.models import ModelResponse, ToolCall
from simple_cc.telemetry import capture_tool_artifact
from simple_cc.trace import TraceRecorder, TraceWriteError, read_trace_lines
from tests.fakes import ScriptedProvider


def unsupported_research_provider(text: str = "answer") -> ScriptedProvider:
    return ScriptedProvider([ModelResponse(text), ModelResponse(text)])


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

    path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task_id": "task-1",
                "question": "Research",
                "benchmark": "test",
                "task_type": "research",
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown task fields"):
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
        unsupported_research_provider(),
        model="test-model",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["status"] == "completed"
    final_answer = (run_dir / "final_answer.txt").read_text(encoding="utf-8")
    assert final_answer.startswith("INSUFFICIENT_EVIDENCE")
    assert manifest["worker_pid"] > 0


def test_execute_task_publishes_only_terminal_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "simple_cc.benchmark._benchmark_tool_tables",
        lambda options: (
            [{"name": "noop", "description": "noop", "input_schema": {}}],
            {"noop": lambda: "ok"},
        ),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = TaskInput("run-1", "task-1", "Research", None, "test", "research")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "intermediate narration",
                [ToolCall("tool-1", "noop", {})],
                "tool_calls",
            ),
            ModelResponse("terminal answer"),
            ModelResponse("terminal answer"),
        ]
    )

    exit_code = execute_task(
        task,
        run_dir,
        run_dir / "agent_workspace",
        provider,
        model="test-model",
    )

    rows, _ = read_trace_lines(run_dir / "trajectory.jsonl")
    traced_answer = next(
        row["payload"]["text"]
        for row in rows
        if row["event_type"] == "final_answer"
    )
    saved_answer = (run_dir / "final_answer.txt").read_text(encoding="utf-8")

    assert exit_code == 0
    assert traced_answer.startswith("INSUFFICIENT_EVIDENCE")
    assert saved_answer == traced_answer


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
        lambda settings: unsupported_research_provider(),
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
    final_answer = (run_dir / "final_answer.txt").read_text(encoding="utf-8")
    assert final_answer.startswith("INSUFFICIENT_EVIDENCE")


def test_offline_worker_search_fetch_answer_trace_is_self_consistent(
    tmp_path, monkeypatch
):
    cutoff = "2025-05-01"
    url_a = "https://alpha.example/report?a=1&b=2"
    url_b = "https://beta.example/data"
    seen_arguments = []

    def search(**arguments):
        seen_arguments.append(("search", arguments))
        return json.dumps(
            {
                "ok": True,
                "operation": "search",
                "query": arguments["query"],
                "cutoff": arguments["cutoff"],
                "results": [
                    {"url": url_a, "title": "Report"},
                    {"url": url_b, "title": "Data"},
                ],
            }
        )

    def fetch(**arguments):
        seen_arguments.append(("fetch", arguments))
        capture_tool_artifact(
            b"<html>accepted evidence</html>",
            media_type="text/html",
            source=arguments["url"],
            suffix=".html",
        )
        return json.dumps(
            {
                "ok": True,
                "operation": "fetch",
                "url": arguments["url"],
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
                [ToolCall("fetch-1", "web_fetch", {"url": url_a})],
                "tool_calls",
            ),
            ModelResponse(f"Draft answer: {url_a}", [], "stop"),
            ModelResponse(
                "",
                [ToolCall("fetch-2", "web_fetch", {"url": url_b})],
                "tool_calls",
            ),
            ModelResponse(
                f"Final answer supported by {url_a} and {url_b}.",
                [],
                "stop",
            ),
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
    sources = [row for row in rows if row["event_type"] == "source_registered"]
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
        ("fetch", {"url": url_a, "cutoff": cutoff}),
        ("fetch", {"url": url_b, "cutoff": cutoff}),
    ]
    assert len(sources) == 2
    assert {
        row["payload"]["source_id"] for row in sources
    } == set(final["payload"]["matched_source_ids"])
    assert all(row["payload"]["latency_ms"] >= 0 for row in llm + tools)
    assert metrics.model_calls == 5
    assert metrics.searches == 1
    assert metrics.fetches == 2
    assert metrics.all_in_prompt_tokens is None


def test_worker_redacts_configured_secret_everywhere(tmp_path):
    secret = "sk-test-super-secret-value"
    provider = unsupported_research_provider(f"answer {secret}")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = TaskInput(
        "run-secret", "task-secret", f"question {secret}", None, "test", "research"
    )
    exit_code = execute_task(
        task,
        run_dir,
        run_dir / "agent_workspace",
        provider,
        model="test-model",
        secrets=(secret,),
    )

    assert exit_code == 0
    for path in [
        run_dir / "manifest.json",
        run_dir / "trajectory.jsonl",
        run_dir / "final_answer.txt",
        *list((run_dir / "artifacts").iterdir()),
    ]:
        assert secret.encode() not in path.read_bytes()


def test_manifest_hashes_match_first_captured_request(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = TaskInput("run-hash", "task-hash", "q", None, "test", "research")
    assert execute_task(
        task,
        run_dir,
        run_dir / "agent_workspace",
        unsupported_research_provider(),
        model="test-model",
    ) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows, _ = read_trace_lines(run_dir / "trajectory.jsonl")
    request_event = next(row for row in rows if row["event_type"] == "llm_request_started")
    request_ref = request_event["payload"]["request_artifact"]
    request = json.loads((run_dir / request_ref["path"]).read_text(encoding="utf-8"))
    assert manifest["prompt_sha256"] == hashlib.sha256(
        request["system"].encode("utf-8")
    ).hexdigest()
    assert manifest["tool_schema_sha256"] == hashlib.sha256(
        json.dumps(
            request["tools"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_trace_finalize_failure_never_publishes_official_answer(tmp_path, monkeypatch):
    original_finalize = TraceRecorder.finalize

    def fail_completed(self, status, details=None):
        if status == "completed":
            raise TraceWriteError("final manifest unavailable")
        return original_finalize(self, status, details)

    monkeypatch.setattr(TraceRecorder, "finalize", fail_completed)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exit_code = execute_task(
        TaskInput("run-1", "task-1", "q", None, "test", "research"),
        run_dir,
        run_dir / "agent_workspace",
        unsupported_research_provider(),
        model="test-model",
    )
    assert exit_code == 2
    assert not (run_dir / "final_answer.txt").exists()
    assert not (run_dir / ".final_answer.staged").exists()
