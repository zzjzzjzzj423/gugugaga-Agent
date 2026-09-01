from __future__ import annotations

import glob as g
import hashlib
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

from . import config
from .mutations import MUTATIONS, emit_mutation_state, mutation_requires_hash


def safe_path(p: str, cwd: Path = None) -> Path:
    base = (cwd or config.WORKDIR).resolve()
    path = (base / p).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deadline_value(deadline: float | Callable[[], float] | None) -> float | None:
    return deadline() if callable(deadline) else deadline


def _cancelled(
    cancel_event: threading.Event | None,
    deadline: float | Callable[[], float] | None,
) -> bool:
    value = _deadline_value(deadline)
    return bool(
        (cancel_event is not None and cancel_event.is_set())
        or (value is not None and time.monotonic() >= value)
    )


def _atomic_commit(
    target: Path,
    content: str,
    *,
    cancel_event: threading.Event | None,
    deadline: float | Callable[[], float] | None,
) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.gugugaga-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for offset in range(0, len(content), 1024 * 1024):
                if _cancelled(cancel_event, deadline):
                    return False
                handle.write(content[offset : offset + 1024 * 1024])
            handle.flush()
            os.fsync(handle.fileno())
        if _cancelled(cancel_event, deadline):
            return False
        # This is the intentionally non-cancellable, short commit point.
        os.replace(temp_path, target)
        temp_path = None
        emit_mutation_state("committed", path=str(target))
        return True
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _workspace_snapshot(base: Path) -> dict[str, tuple[int, int]]:
    values: dict[str, tuple[int, int]] = {}
    try:
        for path in base.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            try:
                stat = path.stat()
                values[str(path.resolve()).casefold()] = (
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            except OSError:
                continue
    except OSError:
        pass
    return values


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _declared_bash_paths(base: Path, write_paths: list[str] | None) -> tuple[list[str], bool]:
    if not write_paths:
        return [], True
    values: list[str] = []
    for raw in write_paths:
        value = str(raw)
        if any(char in value for char in "*?[") or value.endswith(("/", "\\")):
            return [], True
        target = safe_path(value, base)
        if target.exists() and target.is_dir():
            return [], True
        values.append(value)
    return values, False


def run_bash(
    command: str,
    cwd: Path = None,
    run_in_background: bool = False,
    cancel_event: threading.Event | None = None,
    write_paths: list[str] | None = None,
    deadline: float | Callable[[], float] | None = None,
) -> str:
    del run_in_background
    base = (cwd or config.WORKDIR).resolve()
    process = None
    declared, global_write = _declared_bash_paths(base, write_paths)
    before: dict[str, tuple[int, int]] = {}
    try:
        with MUTATIONS.acquire(
            base,
            paths=declared,
            global_write=global_write,
            cancel_event=cancel_event,
            deadline=deadline,
        ):
            if _cancelled(cancel_event, deadline):
                return "Error: Cancelled"
            if declared:
                before = _workspace_snapshot(base)
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=base,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            command_deadline = min(
                value
                for value in (
                    time.monotonic() + 120,
                    _deadline_value(deadline) or float("inf"),
                )
            )
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    _terminate_process_tree(process)
                    return "Error: Cancelled"
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= command_deadline:
                        _terminate_process_tree(process)
                        return "Error: Timeout"
            output = (stdout + stderr).strip()
            if declared:
                after = _workspace_snapshot(base)
                declared_keys = set(MUTATIONS.normalize_paths(base, declared))
                changed = {
                    path
                    for path in set(before) | set(after)
                    if before.get(path) != after.get(path)
                }
                undeclared = sorted(changed - declared_keys)
                if undeclared:
                    relative = []
                    for value in undeclared[:100]:
                        try:
                            relative.append(str(Path(value).relative_to(base)))
                        except ValueError:
                            relative.append(value)
                    audit = "\nError: undeclared_write: " + ", ".join(relative)
                    output = (output + audit).strip()
            return output[:50000] if output else "(no output)"
    except Exception as error:
        return f"Error: {error}"
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)


def run_read(
    path: str,
    limit: int | None = None,
    offset: int = 0,
    cwd: Path = None,
    include_hash: bool = False,
) -> str:
    try:
        target = safe_path(path, cwd)
        lines = target.read_text(encoding="utf-8").splitlines()
        offset = max(int(offset or 0), 0)
        limit = int(limit) if limit is not None else None
        lines = lines[offset:]
        if limit is not None and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        rendered = "\n".join(lines)
        if include_hash:
            metadata = f"<file_metadata sha256=\"{file_sha256(target)}\" />"
            rendered = f"{rendered}\n{metadata}" if rendered else metadata
        return rendered
    except Exception as error:
        return f"Error: {error}"


def _check_expected_hash(target: Path, expected_sha256: str | None) -> str | None:
    if target.exists():
        actual = file_sha256(target)
        if not expected_sha256 and mutation_requires_hash():
            return f"Conflict: expected_sha256 is required for existing file {target.name}"
        if expected_sha256 and expected_sha256.lower() != actual:
            return f"Conflict: stale file {target.name}; current_sha256={actual}"
    elif expected_sha256:
        return f"Conflict: file {target.name} no longer exists"
    return None


def run_write(
    path: str,
    content: str,
    cwd: Path = None,
    expected_sha256: str | None = None,
    cancel_event: threading.Event | None = None,
    deadline: float | Callable[[], float] | None = None,
) -> str:
    try:
        base = (cwd or config.WORKDIR).resolve()
        target = safe_path(path, base)
        with MUTATIONS.acquire(
            base, paths=[path], cancel_event=cancel_event, deadline=deadline
        ):
            conflict = _check_expected_hash(target, expected_sha256)
            if conflict:
                return conflict
            if not _atomic_commit(
                target, content, cancel_event=cancel_event, deadline=deadline
            ):
                return "Error: Cancelled before commit"
            return f"Wrote {len(content)} bytes to {path}"
    except Exception as error:
        return f"Error: {error}"


def run_edit(
    path: str,
    old_text: str,
    new_text: str,
    cwd: Path = None,
    expected_sha256: str | None = None,
    cancel_event: threading.Event | None = None,
    deadline: float | Callable[[], float] | None = None,
) -> str:
    try:
        base = (cwd or config.WORKDIR).resolve()
        target = safe_path(path, base)
        with MUTATIONS.acquire(
            base, paths=[path], cancel_event=cancel_event, deadline=deadline
        ):
            conflict = _check_expected_hash(target, expected_sha256)
            if conflict:
                return conflict
            text = target.read_text(encoding="utf-8")
            if old_text not in text:
                return f"Error: text not found in {path}"
            content = text.replace(old_text, new_text, 1)
            if not _atomic_commit(
                target, content, cancel_event=cancel_event, deadline=deadline
            ):
                return "Error: Cancelled before commit"
            return f"Edited {path}"
    except Exception as error:
        return f"Error: {error}"


def run_glob(pattern: str, cwd: Path = None) -> str:
    try:
        base = (cwd or config.WORKDIR).resolve()
        results = []
        for match in g.glob(pattern, root_dir=base):
            if (base / match).resolve().is_relative_to(base):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as error:
        return f"Error: {error}"
