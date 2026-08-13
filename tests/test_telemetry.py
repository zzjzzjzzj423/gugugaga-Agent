from __future__ import annotations

import json

import pytest

from simple_cc.provider import ProviderResponse, ProviderUsage, TextBlock
from simple_cc.telemetry import TracingProvider, model_call_scope
from simple_cc.trace import RunContext, TraceRecorder, bind_run_context


class Delegate:
    def __init__(self, response):
        self.response = response

    def create(self, messages, system, tools, max_tokens=8192, model=None):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _start(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    return recorder, RunContext(recorder, "run-1", "task-1", None)


def _rows(recorder):
    return [
        json.loads(line)
        for line in recorder.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]


def test_tracing_provider_records_request_response_usage_and_latency(tmp_path):
    recorder, run = _start(tmp_path)
    delegate = Delegate(
        ProviderResponse(
            [TextBlock("answer")],
            "end_turn",
            usage=ProviderUsage(5, 3, 8),
            request_id="req-1",
            attempts=2,
        )
    )
    provider = TracingProvider(delegate)

    with bind_run_context(run), model_call_scope("agent"):
        response = provider.create(
            [{"role": "user", "content": "secret-free"}],
            "system",
            [],
            100,
            "model-a",
        )

    assert response.stop_reason == "end_turn"
    rows = _rows(recorder)
    request, result = rows[-2:]
    assert request["event_type"] == "llm_request_started"
    assert result["event_type"] == "llm_response"
    assert request["span_id"] == result["span_id"]
    assert result["payload"]["usage"]["total_tokens"] == 8
    assert result["payload"]["attempts"] == 2
    assert result["payload"]["latency_ms"] >= 0
    artifact = recorder.run_dir / request["payload"]["request_artifact"]["path"]
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["messages"][0]["content"] == "secret-free"


def test_tracing_provider_records_safe_error(tmp_path):
    recorder, run = _start(tmp_path)
    provider = TracingProvider(Delegate(RuntimeError("provider failed")))

    with bind_run_context(run), pytest.raises(RuntimeError, match="provider failed"):
        provider.create([], "", [], 100, "model-a")

    event = _rows(recorder)[-1]
    assert event["event_type"] == "llm_error"
    assert event["payload"]["exception_class"] == "RuntimeError"
    assert event["payload"]["latency_ms"] >= 0


def test_background_thread_inherits_run_context(tmp_path):
    import time
    from types import SimpleNamespace

    from simple_cc.background import (
        background_is_quiescent,
        initialize_background_tasks,
        shutdown_background_tasks,
        start_background_task,
    )
    from simple_cc.trace import current_run_context

    recorder, run = _start(tmp_path)
    observed = []

    def handler(**kwargs):
        observed.append(current_run_context().run_id)
        return "ok"

    initialize_background_tasks()
    block = SimpleNamespace(
        id="tool-1", name="bash", input={"command": "noop"}
    )
    with bind_run_context(run):
        start_background_task(block, {"bash": handler}, parent_span_id="tool-parent")
    deadline = time.monotonic() + 2
    while not background_is_quiescent() and time.monotonic() < deadline:
        time.sleep(0.01)
    shutdown_background_tasks()

    assert observed == ["run-1"]
    completed = next(
        row for row in _rows(recorder) if row["event_type"] == "background_completed"
    )
    assert completed["parent_span_id"] == "tool-parent"
