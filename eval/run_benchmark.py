from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_cc.eval_metrics import validate_completed_run
from simple_cc.trace import (
    TERMINAL_STATUSES,
    configured_secrets_from_environment,
    redact_value,
    supervisor_finalize_manifest,
    supervisor_invalidate_manifest,
)


@dataclass(frozen=True)
class RunnerOptions:
    dataset: Path
    output_dir: Path
    workers: int = 2
    timeout_seconds: float = 900.0
    limit: int | None = None
    task_ids: frozenset[str] = frozenset()
    resume: bool = False


@dataclass(frozen=True)
class RunAssignment:
    task: dict[str, Any]
    run_id: str
    run_dir: Path
    workspace: Path
    task_input: Path
    retry_of_run_id: str | None


@dataclass(frozen=True)
class RunResult:
    task_id: str
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    exit_code: int | None
    skipped: bool = False


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_tasks(options: RunnerOptions) -> list[dict[str, Any]]:
    if options.workers < 1:
        raise ValueError("workers must be at least 1")
    if options.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    tasks = []
    seen = set()
    for line_number, line in enumerate(
        Path(options.dataset).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        task = json.loads(line)
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"line {line_number} has no task_id")
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        if options.task_ids and task_id not in options.task_ids:
            continue
        tasks.append(task)
        if options.limit is not None and len(tasks) >= options.limit:
            break
    return tasks


def _manifests_for_task(output_dir: Path, task_id: str) -> list[tuple[Path, dict]]:
    found = []
    if not output_dir.exists():
        return found
    for manifest_path in output_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            task_input_path = manifest_path.parent / "task_input.json"
            try:
                task_input = json.loads(task_input_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if task_input.get("task_id") != task_id:
                continue
            supervisor_invalidate_manifest(
                manifest_path.parent,
                {"run_id": task_input.get("run_id"), "task_id": task_id},
                ["manifest is malformed"],
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("task_id") == task_id:
            found.append((manifest_path.parent, manifest))
    found.sort(key=lambda item: item[1].get("started_at") or "")
    return found


def allocate_run(
    task: dict[str, Any], output_dir: Path, retry_of_run_id: str | None = None
) -> RunAssignment:
    run_id = str(uuid.uuid4())
    run_dir = output_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "agent_workspace"
    payload = {
        "run_id": run_id,
        "task_id": task["task_id"],
        "question": task.get("question", ""),
        "cutoff": task.get("cutoff"),
        "benchmark": task.get("benchmark", "financegym"),
        "task_type": task.get("task_type", "research_analysis"),
        "retry_of_run_id": retry_of_run_id,
    }
    task_input = run_dir / "task_input.json"
    _atomic_json(
        task_input,
        redact_value(payload, configured_secrets_from_environment()),
    )
    return RunAssignment(
        task, run_id, run_dir, workspace, task_input, retry_of_run_id
    )


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def launch_assignment(
    assignment: RunAssignment,
    timeout_seconds: float,
    worker_command: list[str] | None = None,
) -> RunResult:
    command = list(worker_command or [sys.executable, "-m", "eval.run_task"])
    command.extend(
        [
            "--task-input",
            str(assignment.task_input),
            "--run-dir",
            str(assignment.run_dir),
            "--workspace",
            str(assignment.workspace),
        ]
    )
    kwargs: dict[str, Any] = {
        "cwd": Path(__file__).resolve().parents[1],
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "shell": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    try:
        process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        process.communicate()
        supervisor_finalize_manifest(
            assignment.run_dir,
            "timed_out",
            {"run_id": assignment.run_id, "task_id": assignment.task["task_id"]},
            {"worker_pid": process.pid, "timeout_seconds": timeout_seconds},
        )
    manifest_path = assignment.run_dir / "manifest.json"
    if process.returncode not in (0, None):
        current = None
        if manifest_path.exists():
            try:
                current = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                current = None
        if not current or current.get("status") == "running":
            failure_status = (
                "trace_invalid" if process.returncode == 2 else "worker_crashed"
            )
            supervisor_finalize_manifest(
                assignment.run_dir,
                failure_status,
                {"run_id": assignment.run_id, "task_id": assignment.task["task_id"]},
                {"worker_pid": process.pid, "exit_code": process.returncode},
            )
    if not manifest_path.exists():
        supervisor_finalize_manifest(
            assignment.run_dir,
            "worker_crashed",
            {"run_id": assignment.run_id, "task_id": assignment.task["task_id"]},
            {"worker_pid": process.pid, "exit_code": process.returncode},
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        supervisor_invalidate_manifest(
            assignment.run_dir,
            {"run_id": assignment.run_id, "task_id": assignment.task["task_id"]},
            [f"manifest unreadable: {type(error).__name__}"],
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_errors = []
    if manifest.get("run_id") != assignment.run_id or manifest.get("task_id") != assignment.task["task_id"]:
        validation_errors.append("manifest identity mismatch")
    if manifest.get("status") not in TERMINAL_STATUSES:
        validation_errors.append("manifest status is not terminal")
    if manifest.get("status") == "completed":
        valid, errors = validate_completed_run(
            assignment.run_dir,
            expected_run_id=assignment.run_id,
            expected_task_id=assignment.task["task_id"],
        )
        if not valid:
            validation_errors.extend(errors)
    if validation_errors:
        supervisor_invalidate_manifest(
            assignment.run_dir,
            {"run_id": assignment.run_id, "task_id": assignment.task["task_id"]},
            list(dict.fromkeys(validation_errors)),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RunResult(
        assignment.task["task_id"],
        assignment.run_id,
        assignment.run_dir,
        manifest,
        process.returncode,
    )


def run_benchmark(
    options: RunnerOptions, worker_command: list[str] | None = None
) -> list[RunResult]:
    tasks = _load_tasks(options)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    assignments: list[RunAssignment] = []
    for task in tasks:
        prior = _manifests_for_task(options.output_dir, task["task_id"])
        if options.resume and prior and prior[-1][1].get("status") == "completed":
            run_dir, manifest = prior[-1]
            valid, errors = validate_completed_run(
                run_dir,
                expected_run_id=manifest.get("run_id"),
                expected_task_id=task["task_id"],
            )
            if valid:
                results.append(
                    RunResult(
                        task["task_id"],
                        manifest["run_id"],
                        run_dir,
                        manifest,
                        0,
                        True,
                    )
                )
                continue
            supervisor_invalidate_manifest(
                run_dir,
                {"run_id": manifest.get("run_id"), "task_id": task["task_id"]},
                errors,
            )
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            prior[-1] = (run_dir, manifest)
        retry_of = prior[-1][1].get("run_id") if prior else None
        assignments.append(allocate_run(task, options.output_dir, retry_of))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=options.workers
    ) as executor:
        futures = [
            executor.submit(
                launch_assignment,
                assignment,
                options.timeout_seconds,
                worker_command,
            )
            for assignment in assignments
        ]
        results.extend(future.result() for future in futures)
    order = {task["task_id"]: index for index, task in enumerate(tasks)}
    return sorted(results, key=lambda item: order[item.task_id])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated benchmark tasks")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/runs"))
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results = run_benchmark(
        RunnerOptions(
            args.dataset,
            args.output_dir,
            args.workers,
            args.timeout_seconds,
            args.limit,
            frozenset(args.task_id),
            args.resume,
        )
    )
    counts: dict[str, int] = {}
    for result in results:
        status = result.manifest.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"tasks": len(results), "statuses": counts}, sort_keys=True))
    return 0 if all(item.manifest.get("status") == "completed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
