from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=0)
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
    manifest = {
        "schema_version": "1.0",
        "run_id": task["run_id"],
        "task_id": task["task_id"],
        "status": "completed",
        "worker_pid": os.getpid(),
        "agent_workspace": str(args.workspace.resolve()),
        "other_marker_visible": visible,
        "preexisting_state": preexisting_state,
    }
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
