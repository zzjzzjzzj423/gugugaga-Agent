from __future__ import annotations

import json
import sys

from eval.run_benchmark import RunnerOptions, run_benchmark


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
