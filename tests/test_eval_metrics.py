from __future__ import annotations

import json

from simple_cc.eval_metrics import derive_metrics, validate_trace
from simple_cc.trace import TraceRecorder


def test_metrics_separate_core_and_all_in_usage(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    recorder.record(
        "llm_response",
        {
            "call_kind": "agent",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "latency_ms": 20,
        },
    )
    recorder.record(
        "llm_response",
        {
            "call_kind": "memory_retrieval",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "latency_ms": 5,
        },
    )
    recorder.record("search_result", {"query": "Rates", "candidate_urls": []})
    recorder.record("search_result", {"query": " rates ", "candidate_urls": []})
    recorder.finalize("completed")

    metrics = derive_metrics(recorder.run_dir)
    assert metrics.trace_valid is True
    assert metrics.model_calls == 2
    assert metrics.core_prompt_tokens == 10
    assert metrics.all_in_prompt_tokens == 13
    assert metrics.repeated_query_rate == 0.5


def test_missing_usage_makes_total_unknown_instead_of_zero(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    recorder.record(
        "llm_response",
        {
            "call_kind": "agent",
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "latency_ms": 1,
        },
    )
    recorder.finalize("completed")
    metrics = derive_metrics(recorder.run_dir)
    assert metrics.core_prompt_tokens is None
    assert metrics.all_in_prompt_tokens is None


def test_trace_validation_rejects_sequence_gap(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    rows = [json.loads(line) for line in recorder.trajectory_path.read_text().splitlines()]
    rows[0]["sequence"] = 2
    recorder.trajectory_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    valid, errors = validate_trace(recorder.run_dir)
    assert valid is False
    assert any("sequence" in error for error in errors)
