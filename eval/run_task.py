from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from simple_cc.benchmark import BenchmarkOptions, build_benchmark_runtime
from simple_cc.config import Settings
from simple_cc.models import ChatProvider
from simple_cc.provider import SiliconFlowProvider
from simple_cc.trace import (
    TraceRecorder,
    TraceWriteError,
    configured_secrets_from_environment,
    supervisor_invalidate_manifest,
)
from simple_cc.web_research import PIT_MODE


@dataclass(frozen=True)
class TaskInput:
    run_id: str
    task_id: str
    question: str
    cutoff: str | None
    benchmark: str
    task_type: str
    retry_of_run_id: str | None = None


def load_task_input(path: Path | str) -> TaskInput:
    raw = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(raw, dict):
        raise ValueError("task input must be one JSON object")
    allowed = {
        "run_id",
        "task_id",
        "question",
        "cutoff",
        "benchmark",
        "task_type",
        "retry_of_run_id",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown task fields: {', '.join(unknown)}")
    required = ("run_id", "task_id", "question", "benchmark", "task_type")
    for key in required:
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    cutoff = raw.get("cutoff")
    if cutoff is not None:
        try:
            parsed = date.fromisoformat(cutoff)
        except (TypeError, ValueError) as error:
            raise ValueError("cutoff must use YYYY-MM-DD") from error
        if parsed.isoformat() != cutoff:
            raise ValueError("cutoff must use YYYY-MM-DD")
    retry_of = raw.get("retry_of_run_id")
    if retry_of is not None and (
        not isinstance(retry_of, str) or not retry_of.strip()
    ):
        raise ValueError("retry_of_run_id must be a non-empty string or null")
    return TaskInput(
        raw["run_id"],
        raw["task_id"],
        raw["question"],
        cutoff,
        raw["benchmark"],
        raw["task_type"],
        retry_of,
    )


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate task field: {key}")
        result[key] = value
    return result


def _hash_json(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _git_metadata() -> tuple[str | None, bool | None, str | None]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty, None
    except Exception as error:
        return None, None, f"git metadata unavailable: {type(error).__name__}"


def _atomic_text(path: Path, text: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def execute_task(
    task: TaskInput,
    run_dir: Path | str,
    workspace: Path | str,
    provider: ChatProvider,
    *,
    model: str,
    max_rounds: int = 40,
    secrets: tuple[str, ...] | list[str] = (),
) -> int:
    run_dir = Path(run_dir).resolve()
    workspace = Path(workspace).resolve()
    options = BenchmarkOptions(max_rounds=max_rounds)
    provider_secret = getattr(getattr(provider, "settings", None), "api_key", None)
    all_secrets = tuple(
        dict.fromkeys(
            str(item)
            for item in (*secrets, provider_secret)
            if item is not None and str(item)
        )
    )
    recorder = TraceRecorder(run_dir, run_id=task.run_id, secrets=all_secrets)
    commit, dirty, warning = _git_metadata()
    metadata = {
        "benchmark": task.benchmark,
        "task_type": task.task_type,
        "pit_mode": PIT_MODE,
        "provider": "siliconflow",
        "model": model,
        "worker_pid": os.getpid(),
        "agent_workspace": str(workspace),
        "isolation": {
            "one_task_per_process": True,
            "memory_enabled": False,
            "cron_enabled": False,
            "team_enabled": False,
            "subagent_enabled": False,
            "interactive_approval_enabled": False,
        },
        "max_rounds": max_rounds,
        "prompt_sha256": None,
        "tool_schema_sha256": _hash_json([]),
        "git_commit": commit,
        "git_dirty": dirty,
        "metadata_warning": warning,
        "retry_of_run_id": task.retry_of_run_id,
    }
    try:
        recorder.start_run(
            task_id=task.task_id,
            question=task.question,
            cutoff=task.cutoff,
            metadata=metadata,
        )
        session = build_benchmark_runtime(
            run_dir=run_dir,
            workspace=workspace,
            provider=provider,
            recorder=recorder,
            options=options,
        )
        try:
            answer = session.runtime.run_turn(
                task.question,
                task_id=task.task_id,
                cutoff=task.cutoff,
                run_metadata=metadata,
            )
            outcome = session.runtime.last_outcome
        finally:
            close_outcome = session.close()
        if outcome is None:
            recorder.finalize("failed", {"failure_class": "missing_outcome"})
            return 1
        if outcome.status != "completed":
            recorder.finalize(
                outcome.status,
                {
                    "failure_class": outcome.failure_class,
                    "failure_message": outcome.failure_message,
                },
            )
            return 1
        if not close_outcome.stopped:
            recorder.finalize(
                "failed",
                {
                    "failure_class": "resource_leak",
                    "live_resources": close_outcome.live_resources,
                },
            )
            return 1
        safe_answer = recorder.redact_text(answer)
        staged_answer = run_dir / ".final_answer.staged"
        recorder.record("run_completed", {"answer_chars": len(safe_answer)})
        _atomic_text(staged_answer, safe_answer)
        recorder.finalize("completed")
        try:
            os.replace(staged_answer, run_dir / "final_answer.txt")
        except Exception as error:
            supervisor_invalidate_manifest(
                run_dir,
                {"run_id": task.run_id, "task_id": task.task_id},
                [f"final answer publication failed: {type(error).__name__}"],
            )
            return 2
        return 0
    except TraceWriteError:
        for path in (run_dir / ".final_answer.staged", run_dir / "final_answer.txt"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return 2
    except Exception as error:
        try:
            recorder.record(
                "run_failed",
                {"failure_class": type(error).__name__, "message": str(error)},
            )
            recorder.finalize(
                "failed",
                {"failure_class": type(error).__name__, "failure_message": str(error)},
            )
        except Exception:
            pass
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exactly one benchmark task")
    parser.add_argument("--task-input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    task = load_task_input(args.task_input)
    settings = Settings.from_env(args.workspace, create_dirs=False)
    provider = SiliconFlowProvider(settings)
    return execute_task(
        task,
        args.run_dir,
        args.workspace,
        provider,
        model=settings.model,
        max_rounds=settings.max_rounds,
        secrets=configured_secrets_from_environment(settings.api_key),
    )


if __name__ == "__main__":
    raise SystemExit(main())
