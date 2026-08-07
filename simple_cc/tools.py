from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from .models import ToolSpec


class ToolRegistry:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, description: str, parameters: dict, handler: Callable[..., Any]) -> None:
        self._specs[name] = ToolSpec(name, description, parameters)
        self._handlers[name] = handler

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return str(handler(**arguments))
        except Exception as error:
            return f"Error: {type(error).__name__}: {error}"


class WorkspaceTools:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def _safe(self, value: str) -> Path:
        path = (self.workspace / value).resolve()
        try:
            path.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError(f"path escapes workspace: {value}") from error
        return path

    def bash(self, command: str, timeout: int = 120) -> str:
        result = subprocess.run(
            command, shell=True, cwd=self.workspace, capture_output=True,
            text=True, timeout=min(max(timeout, 1), 300),
        )
        output = (result.stdout + result.stderr).strip()
        return output[:100_000] or "(no output)"

    def read_file(self, path: str, limit: int | None = None) -> str:
        try:
            lines = self._safe(path).read_text(encoding="utf-8").splitlines()
            if limit is not None and limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
            return "\n".join(lines)
        except Exception as error:
            return f"Error: {error}"

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._safe(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} characters to {path}"
        except Exception as error:
            return f"Error: {error}"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            target = self._safe(path)
            content = target.read_text(encoding="utf-8")
            if old_text not in content:
                return f"Error: text not found in {path}"
            target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
            return f"Edited {path}"
        except Exception as error:
            return f"Error: {error}"

    def glob(self, pattern: str) -> str:
        try:
            matches = [p.relative_to(self.workspace).as_posix() for p in self.workspace.glob(pattern) if p.is_file()]
            return "\n".join(sorted(matches)[:1000]) or "No matches"
        except Exception as error:
            return f"Error: {error}"

    def register_into(self, registry: ToolRegistry) -> None:
        obj = lambda props, required=(): {"type": "object", "properties": props, "required": list(required)}
        registry.register("bash", "Run a shell command in the workspace", obj({"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]), self.bash)
        registry.register("read_file", "Read a UTF-8 file", obj({"path": {"type": "string"}, "limit": {"type": "integer"}}, ["path"]), self.read_file)
        registry.register("write_file", "Write a UTF-8 file", obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]), self.write_file)
        registry.register("edit_file", "Replace exact text once", obj({"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]), self.edit_file)
        registry.register("glob", "List files matching a glob", obj({"pattern": {"type": "string"}}, ["pattern"]), self.glob)

