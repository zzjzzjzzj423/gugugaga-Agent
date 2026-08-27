from __future__ import annotations

import glob as g
import subprocess
import threading
import time
from pathlib import Path

from . import config


def safe_path(p: str, cwd: Path = None) -> Path:
    # File tools stay inside the workspace. Bash remains powerful on purpose
    # and is controlled by the permission hook instead.
    base = cwd or config.WORKDIR
    path = (base / p).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(
    command: str,
    cwd: Path = None,
    run_in_background: bool = False,
    cancel_event: threading.Event | None = None,
) -> str:
    # run_in_background is consumed by the dispatcher; direct execution ignores it.
    process = None
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd or config.WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 120
        while True:
            if cancel_event is not None and cancel_event.is_set():
                # Terminating a shell cannot guarantee that every descendant has
                # exited on every platform; shutdown reports the worker live until
                # this process wait actually completes.
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
                return "Error: Cancelled"
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    process.kill()
                    process.wait(timeout=1)
                    return "Error: Timeout (120s)"
        output = (stdout + stderr).strip()
        return output[:50000] if output else "(no output)"
    finally:
        if process is not None and process.poll() is None:
            process.kill()


def run_read(
    path: str, limit: int | None = None, offset: int = 0, cwd: Path = None
) -> str:
    try:
        lines = safe_path(path, cwd).read_text(encoding="utf-8").splitlines()
        offset = max(int(offset or 0), 0)
        limit = int(limit) if limit is not None else None
        lines = lines[offset:]
        if limit is not None and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as error:
        return f"Error: {error}"

def run_write(path: str, content: str, cwd: Path = None) -> str:
    try:
        target = safe_path(path, cwd)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as error:
        return f"Error: {error}"


def run_edit(
    path: str, old_text: str, new_text: str, cwd: Path = None
) -> str:
    try:
        target = safe_path(path, cwd)
        text = target.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        target.write_text(
            text.replace(old_text, new_text, 1),
            encoding="utf-8",
            newline="\n",
        )
        return f"Edited {path}"
    except Exception as error:
        return f"Error: {error}"


def run_glob(pattern: str, cwd: Path = None) -> str:
    try:
        base = cwd or config.WORKDIR
        results = []
        for match in g.glob(pattern, root_dir=base):
            if (base / match).resolve().is_relative_to(base):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as error:
        return f"Error: {error}"
