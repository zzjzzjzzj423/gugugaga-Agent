from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simple_cc.trace import TraceRecorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--malformed-manifest", action="store_true")
    args = parser.parse_args()
    task = json.loads(args.task_input.read_text(encoding="utf-8"))
    state_names = (
        ".memory",
        ".transcripts",
        ".mailboxes",
        ".task_outputs",
        "final_answer.txt",
        "reference_answer.json",
        "judge.json",
    )
    preexisting_state = [
        name for name in state_names if (args.workspace / name).exists()
    ]
    args.workspace.mkdir(parents=True, exist_ok=True)
    other_marker = args.run_dir.parent / "shared-marker.txt"
    visible = other_marker.exists()
    (args.workspace / "own-marker.txt").write_text(task["task_id"], encoding="utf-8")
    time.sleep(args.sleep)
    recorder = TraceRecorder(args.run_dir, run_id=task["run_id"])
    metadata = {
        "worker_pid": os.getpid(),
        "agent_workspace": str(args.workspace.resolve()),
        "other_marker_visible": visible,
        "preexisting_state": preexisting_state,
    }
    recorder.start_run(
        task_id=task["task_id"],
        question=task["question"],
        cutoff=task.get("cutoff"),
        metadata=metadata,
    )
    answer = f"probe answer for {task['task_id']}"
    recorder.record("final_answer", {"text": answer})
    recorder.record("run_completed", {"answer_chars": len(answer)})
    recorder.finalize("completed")
    (args.run_dir / "final_answer.txt").write_text(answer, encoding="utf-8")
    if args.malformed_manifest:
        (args.run_dir / "manifest.json").write_text("{broken", encoding="utf-8")


if __name__ == "__main__":
    main()
