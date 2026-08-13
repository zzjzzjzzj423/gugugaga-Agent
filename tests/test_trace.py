from __future__ import annotations

import json

import pytest

from simple_cc.trace import (
    SCHEMA_VERSION,
    TraceRecorder,
    TraceWriteError,
    read_trace_lines,
    supervisor_finalize_manifest,
)


def _rows(recorder: TraceRecorder) -> list[dict]:
    return [
        json.loads(line)
        for line in recorder.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]


def test_record_is_immediately_readable_and_monotonic(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(
        task_id="task-1",
        question="q",
        cutoff="2025-01-01",
        metadata={"benchmark": "test"},
    )
    recorder.record("tool_started", {"name": "web_search"})

    rows = _rows(recorder)
    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[1]["event_type"] == "tool_started"
    assert rows[1]["task_id"] == "task-1"
    assert rows[1]["elapsed_ms"] >= 0


def test_artifacts_are_deduplicated_by_sha256(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    first = recorder.store_artifact(
        "same", media_type="text/plain", source="test", suffix=".txt"
    )
    second = recorder.store_artifact(
        "same", media_type="text/plain", source="test", suffix=".txt"
    )

    assert first == second
    assert len(list((tmp_path / "run" / "artifacts").iterdir())) == 1
    assert (tmp_path / "run" / first.path).read_text(encoding="utf-8") == "same"


def test_artifact_deduplication_does_not_depend_on_requested_suffix(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    first = recorder.store_artifact(
        "same", media_type="text/plain", source="one", suffix=".txt"
    )
    second = recorder.store_artifact(
        "same", media_type="application/json", source="two", suffix=".json"
    )
    assert first.path == second.path
    assert len(list(recorder.artifacts_dir.iterdir())) == 1


def test_redaction_happens_before_disk_write(tmp_path):
    recorder = TraceRecorder(
        tmp_path / "run", run_id="run-1", secrets=["secret-value"]
    )
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    recorder.record(
        "tool_requested",
        {"api_key": "secret-value", "text": "x secret-value y"},
    )

    disk = recorder.trajectory_path.read_text(encoding="utf-8")
    assert "secret-value" not in disk
    assert "[REDACTED:api_key]" in disk
    assert "[REDACTED:configured_secret]" in disk


def test_incomplete_final_line_is_reported_without_losing_prior_rows(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    with recorder.trajectory_path.open("ab") as handle:
        handle.write(b'{"broken"')

    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert len(rows) == 1
    assert incomplete is True


def test_terminal_manifest_cannot_be_rewritten(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    recorder.finalize("completed")

    with pytest.raises(ValueError, match="terminal"):
        recorder.finalize("failed")


def test_supervisor_can_finalize_worker_that_never_wrote_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    supervisor_finalize_manifest(
        run_dir,
        "worker_crashed",
        {"run_id": "run-1", "task_id": "task-1"},
        {"exit_code": 3},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "worker_crashed"
    assert manifest["exit_code"] == 3


def test_record_failure_is_fail_closed(tmp_path, monkeypatch):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(recorder, "_append_line", fail)
    with pytest.raises(TraceWriteError, match="disk unavailable"):
        recorder.record("tool_started", {"name": "x"})


def test_manifest_identity_and_running_status_cannot_be_overridden(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="actual-run")
    recorder.start_run(
        task_id="actual-task",
        question="q",
        cutoff=None,
        metadata={
            "run_id": "forged-run",
            "task_id": "forged-task",
            "status": "completed",
            "schema_version": "forged",
        },
    )

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "actual-run"
    assert manifest["task_id"] == "actual-task"
    assert manifest["status"] == "running"
    assert manifest["schema_version"] == SCHEMA_VERSION

    recorder.finalize(
        "failed",
        {
            "run_id": "forged-final-run",
            "task_id": "forged-final-task",
            "schema_version": "forged-final",
        },
    )
    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "actual-run"
    assert manifest["task_id"] == "actual-task"
    assert manifest["schema_version"] == SCHEMA_VERSION


def test_supervisor_details_cannot_override_parent_owned_identity(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    supervisor_finalize_manifest(
        run_dir,
        "trace_invalid",
        {"run_id": "actual-run", "task_id": "actual-task"},
        {"run_id": "forged-run", "task_id": "forged-task"},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "actual-run"
    assert manifest["task_id"] == "actual-task"


def test_invalidation_preserves_exact_prior_terminal_manifest(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    recorder.finalize("completed")
    original = recorder.manifest_path.read_bytes()

    from simple_cc.trace import supervisor_invalidate_manifest

    supervisor_invalidate_manifest(
        recorder.run_dir,
        {"run_id": "run-1", "task_id": "task-1"},
        ["answer missing"],
    )
    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    ref = manifest["superseded_manifest_artifact"]
    assert (recorder.run_dir / ref["path"]).read_bytes() == original
    assert manifest["previous_status"] == "completed"
