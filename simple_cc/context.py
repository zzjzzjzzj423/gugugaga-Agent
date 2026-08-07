from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path


class SkillStore:
    def __init__(self, roots: list[Path]):
        self.roots = [Path(root) for root in roots]
        self._skills: dict[str, tuple[str, Path]] = {}
        self.discover()

    def discover(self) -> None:
        self._skills.clear()
        for root in self.roots:
            if not root.exists():
                continue
            for path in root.rglob("SKILL.md"):
                text = path.read_text(encoding="utf-8")
                name = re.search(r"(?m)^name:\s*(.+)$", text)
                desc = re.search(r"(?m)^description:\s*(.+)$", text)
                key = (name.group(1).strip() if name else path.parent.name)
                self._skills[key] = (desc.group(1).strip() if desc else "", path)

    def list_text(self) -> str:
        return "\n".join(f"- {name}: {desc}" for name, (desc, _) in sorted(self._skills.items())) or "No skills discovered."

    def load(self, name: str) -> str:
        item = self._skills.get(name)
        return item[1].read_text(encoding="utf-8") if item else f"Error: unknown skill '{name}'"


class MemoryStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def remember(self, title: str, content: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", title).strip("-") or uuid.uuid4().hex[:8]
        path = self.directory / f"{safe}.md"
        path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
        return f"Remembered {title}"

    def search(self, query: str, limit: int = 5) -> str:
        terms = query.lower().split()
        hits = []
        for path in self.directory.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            score = sum(term in text.lower() for term in terms)
            if score:
                hits.append((score, text))
        return "\n\n".join(text for _, text in sorted(hits, reverse=True)[:limit]) or "No matching memories."

    def index_text(self, limit: int = 20) -> str:
        return "\n".join(path.stem for path in sorted(self.directory.glob("*.md"))[:limit]) or "No memories."


class ContextManager:
    def __init__(self, outputs_dir: Path, transcripts_dir: Path, max_output_chars: int = 50_000, max_messages: int = 60):
        self.outputs_dir, self.transcripts_dir = Path(outputs_dir), Path(transcripts_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.max_output_chars, self.max_messages = max_output_chars, max_messages

    def apply_output_budget(self, messages: list[dict]) -> list[dict]:
        result = []
        for message in messages:
            item = dict(message)
            content = item.get("content")
            if item.get("role") == "tool" and isinstance(content, str) and len(content) > self.max_output_chars:
                path = self.outputs_dir / f"{item.get('tool_call_id', uuid.uuid4().hex)}.txt"
                path.write_text(content, encoding="utf-8")
                preview = content[: max(40, self.max_output_chars // 2)]
                item["content"] = f"{preview}\n...[full output: {path}]"
            result.append(item)
        return result

    def compact(self, messages: list[dict], summary: str | None = None) -> list[dict]:
        if len(messages) <= self.max_messages:
            return self.apply_output_budget(messages)
        stamp = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        (self.transcripts_dir / f"{stamp}.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        kept = messages[-self.max_messages :]
        prefix = {"role": "user", "content": f"<compacted>{summary or 'Earlier conversation archived.'}</compacted>"}
        return [prefix, *self.apply_output_budget(kept)]

    def prepare(self, messages: list[dict]) -> list[dict]:
        return self.compact(messages)

