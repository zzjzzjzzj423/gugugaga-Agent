from __future__ import annotations

import glob as g
import subprocess
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
    command: str, cwd: Path = None, run_in_background: bool = False
) -> str:
    # run_in_background is consumed by the dispatcher; direct execution ignores it.
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or config.WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(
    path: str, limit: int | None = None, offset: int = 0, cwd: Path = None
) -> str:
    try:
        lines = safe_path(path, cwd).read_text().splitlines()
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
        target.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as error:
        return f"Error: {error}"


def run_edit(
    path: str, old_text: str, new_text: str, cwd: Path = None
) -> str:
    try:
        target = safe_path(path, cwd)
        text = target.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        target.write_text(text.replace(old_text, new_text, 1))
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
