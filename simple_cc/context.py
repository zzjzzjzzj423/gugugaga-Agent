from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from . import config
from .subagents import extract_text


client = None


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

    def index_text(self, limit: int = 20, max_chars: int = 4_000) -> str:
        entries = []
        for path in sorted(self.directory.glob("*.md"))[:limit]:
            body = path.read_text(encoding="utf-8").strip().replace("\n", " ")
            entries.append(f"- {path.stem}: {body[:240]}")
        return ("\n".join(entries)[:max_chars] if entries else "No memories.")


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

    def needs_compaction(self, messages: list[dict]) -> bool:
        if not messages or messages[0].get("role") != "user":
            return len(messages) > self.max_messages
        content = str(messages[0].get("content", ""))
        marker = re.match(r'^<compacted tail="(\d+)">', content)
        if not marker:
            return len(messages) > self.max_messages
        baseline_tail = int(marker.group(1))
        current_tail = len(messages) - 1
        return current_tail - baseline_tail > self.max_messages

    def _complete_tail(self, messages: list[dict]) -> list[dict]:
        start = max(0, len(messages) - self.max_messages)
        while start > 0 and messages[start].get("role") == "tool":
            start -= 1
        if (
            start > 0
            and messages[start].get("role") == "assistant"
            and messages[start].get("tool_calls")
            and messages[start - 1].get("role") == "user"
        ):
            start -= 1
        return messages[start:]

    def compact(self, messages: list[dict], summary: str | None = None, force: bool = False) -> list[dict]:
        if len(messages) <= self.max_messages and not force:
            return self.apply_output_budget(messages)
        stamp = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        (self.transcripts_dir / f"{stamp}.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        kept = self._complete_tail(messages)
        prefix = {
            "role": "user",
            "content": (
                f'<compacted tail="{len(kept)}">'
                f"{summary or 'Earlier conversation archived.'}</compacted>"
            ),
        }
        return [prefix, *self.apply_output_budget(kept)]

    def prepare(self, messages: list[dict], summarizer=None) -> list[dict]:
        if not self.needs_compaction(messages):
            return self.apply_output_budget(messages)
        summary = summarizer(messages) if summarizer else "Earlier conversation archived."
        return self.compact(messages, summary)


# Context compaction is layered: first shrink oversized tool results, then
# trim old message ranges, and only call the model for a summary when the
# context is still too large or the model explicitly asks for compact.
def estimate_size(messages: list) -> int:
    return len(json.dumps(messages, default=str))


def block_type(block):
    return (
        block.get("type")
        if isinstance(block, dict)
        else getattr(block, "type", None)
    )


def message_has_tool_use(message: dict) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(block_type(block) == "tool_use" for block in content)


def is_tool_result_message(message: dict) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def collect_tool_results(messages: list):
    found = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if msg.get("role") != "user" or not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
            ):
                found.append((mi, bi, block))
    return found


def persist_large_output(tool_use_id: str, output: str) -> str:
    if len(output) <= config.PERSIST_THRESHOLD:
        return output
    config.TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists():
        path.write_text(output)
    return (
        f"<persisted-output>\nFull output: {path}\n"
        f"Preview:\n{output[:2000]}\n</persisted-output>"
    )


def tool_result_budget(messages: list, max_bytes: int = 200_000) -> list:
    if not messages:
        return messages
    last = messages[-1]
    content = last.get("content")
    if last.get("role") != "user" or not isinstance(content, list):
        return messages
    blocks = [
        (i, block)
        for i, block in enumerate(content)
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    total = sum(len(str(block.get("content", ""))) for _, block in blocks)
    if total <= max_bytes:
        return messages
    for _, block in sorted(
        blocks,
        key=lambda pair: len(str(pair[1].get("content", ""))),
        reverse=True,
    ):
        if total <= max_bytes:
            break
        text = str(block.get("content", ""))
        block["content"] = persist_large_output(
            block.get("tool_use_id", "unknown"), text
        )
        total = sum(
            len(str(candidate.get("content", "")))
            for _, candidate in blocks
        )
    return messages


def snip_compact(messages: list, max_messages: int = 50) -> list:
    if len(messages) <= max_messages:
        return messages
    head_end, tail_start = 3, len(messages) - (max_messages - 3)
    if head_end > 0 and message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and is_tool_result_message(
            messages[head_end]
        ):
            head_end += 1
    if (
        tail_start > 0
        and tail_start < len(messages)
        and is_tool_result_message(messages[tail_start])
        and message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1
    if head_end >= tail_start:
        return messages
    snipped = tail_start - head_end
    return (
        messages[:head_end]
        + [{"role": "user", "content": f"[snipped {snipped} messages]"}]
        + messages[tail_start:]
    )


def micro_compact(messages: list) -> list:
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= config.KEEP_RECENT_TOOL_RESULTS:
        return messages
    for _, _, block in tool_results[: -config.KEEP_RECENT_TOOL_RESULTS]:
        if len(str(block.get("content", ""))) > 120:
            block["content"] = (
                "[Earlier tool result compacted. Re-run if needed.]"
            )
    return messages


def write_transcript(messages: list) -> Path:
    config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as file:
        for message in messages:
            file.write(json.dumps(message, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue. "
        "Preserve current goal, key findings, changed files, remaining work, "
        "and user constraints.\n\n"
        + conversation
    )
    if client is None:
        raise RuntimeError("Context provider is not configured")
    response = client.create(
        model=config.MODEL,
        system="",
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        max_tokens=2000,
    )
    return extract_text(response.content) or "(empty summary)"


def compact_history(messages: list) -> list:
    transcript = write_transcript(messages)
    print(f"  \033[36m[compact] transcript saved: {transcript}\033[0m")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


def reactive_compact(messages: list) -> list:
    transcript = write_transcript(messages)
    print(
        f"  \033[31m[reactive compact] transcript saved: {transcript}\033[0m"
    )
    tail_start = max(0, len(messages) - 5)
    if (
        tail_start > 0
        and tail_start < len(messages)
        and is_tool_result_message(messages[tail_start])
        and message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1
    try:
        summary = summarize_history(messages[:tail_start])
    except Exception:
        summary = (
            "Earlier conversation was trimmed after a prompt-too-long error."
        )
    return [
        {"role": "user", "content": f"[Reactive compact]\n\n{summary}"},
        *messages[tail_start:],
    ]
