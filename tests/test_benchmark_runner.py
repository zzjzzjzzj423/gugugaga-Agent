from __future__ import annotations

import json
import sys

from eval.run_benchmark import RunnerOptions, allocate_run, run_benchmark


def _dataset(tmp_path, count=2):
    path = tmp_path / "tasks.jsonl"
    rows = [
        {
            "task_id": f"task-{index}",
            "question": f"question {index}",
            "cutoff": "2025-01-01",
        }
        for index in range(count)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_runner_launches_fresh_process_and_workspace_per_task(tmp_path):
    output = tmp_path / "runs"
    results = run_benchmark(
        RunnerOptions(_dataset(tmp_path), output, workers=2, timeout_seconds=10),
        worker_command=[sys.executable, "tests/fixtures/isolation_probe.py"],
    )
    manifests = [item.manifest for item in results]
    assert {item["status"] for item in manifests} == {"completed"}
    assert len({item["worker_pid"] for item in manifests}) == 2
    assert len({item["agent_workspace"] for item in manifests}) == 2
    assert not any(item["other_marker_visible"] for item in manifests)
    assert not any(item["preexisting_state"] for item in manifests)


def test_runner_timeout_never_becomes_completed(tmp_path):
    results = run_benchmark(
        RunnerOptions(_dataset(tmp_path, 1), tmp_path / "runs", workers=1, timeout_seconds=0.1),
        worker_command=[
            sys.executable,
            "tests/fixtures/isolation_probe.py",
            "--sleep",
            "2",
        ],
    )
    assert results[0].manifest["status"] == "timed_out"


def test_trace_writer_exit_code_is_classified_as_trace_invalid(tmp_path):
    result = run_benchmark(
        RunnerOptions(
            _dataset(tmp_path, 1),
            tmp_path / "runs",
            workers=1,
            timeout_seconds=10,
        ),
        worker_command=[sys.executable, "-c", "raise SystemExit(2)"],
    )[0]

    assert result.exit_code == 2
    assert result.manifest["status"] == "trace_invalid"


def test_resume_skips_completed_and_retry_links_incomplete(tmp_path):
    output = tmp_path / "runs"
    options = RunnerOptions(_dataset(tmp_path, 1), output, workers=1, timeout_seconds=10)
    first = run_benchmark(
        options,
        worker_command=[sys.executable, "tests/fixtures/isolation_probe.py"],
    )[0]
    resumed = run_benchmark(
        RunnerOptions(options.dataset, output, workers=1, timeout_seconds=10, resume=True),
        worker_command=[sys.executable, "tests/fixtures/isolation_probe.py"],
    )[0]
    assert resumed.run_id == first.run_id
    assert resumed.skipped is True


def test_resume_retries_failed_run_without_overwriting_prior_attempt(tmp_path):
    output = tmp_path / "runs"
    dataset = _dataset(tmp_path, 1)
    failed = run_benchmark(
        RunnerOptions(dataset, output, workers=1, timeout_seconds=0.1),
        worker_command=[
            sys.executable,
            "tests/fixtures/isolation_probe.py",
            "--sleep",
            "2",
        ],
    )[0]
    retried = run_benchmark(
        RunnerOptions(
            dataset, output, workers=1, timeout_seconds=10, resume=True
        ),
        worker_command=[sys.executable, "tests/fixtures/isolation_probe.py"],
    )[0]
    task_input = json.loads(
        (retried.run_dir / "task_input.json").read_text(encoding="utf-8")
    )

    assert failed.manifest["status"] == "timed_out"
    assert retried.manifest["status"] == "completed"
    assert retried.run_id != failed.run_id
    assert task_input["retry_of_run_id"] == failed.run_id
    assert (failed.run_dir / "manifest.json").exists()


def test_resume_invalidates_corrupt_completed_trace_and_retries(tmp_path):
    output = tmp_path / "runs"
    dataset = _dataset(tmp_path, 1)
    first = run_benchmark(
        RunnerOptions(dataset, output, workers=1, timeout_seconds=10),
        worker_command=[sys.executable, "tests/fixtures/isolation_probe.py"],
    )[0]
    with (first.run_dir / "trajectory.jsonl").open("ab") as handle:
        handle.write(b'{"truncated"')

    retried = run_benchmark(
        RunnerOptions(dataset, output, workers=1, timeout_seconds=10, resume=True),
        worker_command=[sys.executable, "tests/fixtures/isolation_probe.py"],
    )[0]
    old_manifest = json.loads(
        (first.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    task_input = json.loads(
        (retried.run_dir / "task_input.json").read_text(encoding="utf-8")
    )
    assert old_manifest["status"] == "trace_invalid"
    assert retried.run_id != first.run_id
    assert task_input["retry_of_run_id"] == first.run_id


def test_malformed_worker_manifest_does_not_crash_coordinator(tmp_path):
    result = run_benchmark(
        RunnerOptions(_dataset(tmp_path, 1), tmp_path / "runs", workers=1),
        worker_command=[
            sys.executable,
            "tests/fixtures/isolation_probe.py",
            "--malformed-manifest",
        ],
    )[0]
    assert result.manifest["status"] == "trace_invalid"
    assert list((result.run_dir / "artifacts").glob("corrupt-manifest-*"))


def test_successful_process_with_nonterminal_manifest_is_trace_invalid(tmp_path):
    result = run_benchmark(
        RunnerOptions(_dataset(tmp_path, 1), tmp_path / "runs", workers=1),
        worker_command=[
            sys.executable,
            "tests/fixtures/isolation_probe.py",
            "--running-manifest",
        ],
    )[0]
    assert result.exit_code == 0
    assert result.manifest["status"] == "trace_invalid"
    assert "not terminal" in " ".join(result.manifest["trace_validation_errors"])


def test_terminal_manifest_identity_mismatch_is_trace_invalid(tmp_path):
    result = run_benchmark(
        RunnerOptions(_dataset(tmp_path, 1), tmp_path / "runs", workers=1),
        worker_command=[
            sys.executable,
            "tests/fixtures/isolation_probe.py",
            "--mismatched-identity",
        ],
    )[0]
    assert result.manifest["status"] == "trace_invalid"
    assert result.manifest["run_id"] == result.run_id
    assert "identity mismatch" in " ".join(
        result.manifest["trace_validation_errors"]
    )


def test_allocated_task_input_is_redacted_before_durable_write(tmp_path, monkeypatch):
    secret = "sk-parent-environment-secret"
    monkeypatch.setenv("SILICONFLOW_API_KEY", secret)
    task = {
        "task_id": "task-secret",
        "question": f"Do not persist {secret}",
        "cutoff": None,
    }
    assignment = allocate_run(task, tmp_path / "runs")
    disk = assignment.task_input.read_text(encoding="utf-8")
    assert secret not in disk
    assert "[REDACTED:configured_secret]" in disk
