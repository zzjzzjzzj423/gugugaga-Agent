from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

WORKDIR = Path.cwd()
MODEL = os.getenv("SILICONFLOW_MODEL", "")
PRIMARY_MODEL = MODEL
FALLBACK_MODEL = os.getenv("SILICONFLOW_FALLBACK_MODEL")

SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
TASKS_DIR = WORKDIR / ".tasks"
MAILBOX_DIR = WORKDIR / ".mailboxes"
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"

DEFAULT_MAX_TOKENS = 8000
ESCALATED_MAX_TOKENS = 16000
MAX_RETRIES = 3
MAX_CONSECUTIVE_529 = 2
MAX_RECOVERY_RETRIES = 2
BASE_DELAY_MS = 500
CONTEXT_LIMIT = 50000
KEEP_RECENT_TOOL_RESULTS = 3
PERSIST_THRESHOLD = 30000
MEMORY_ENABLED = os.getenv("SIMPLE_CC_MEMORY_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
MEMORY_MAX_SELECTED = int(os.getenv("SIMPLE_CC_MEMORY_MAX_SELECTED", "5"))
MEMORY_MAX_INJECTED_CHARS = int(
    os.getenv("SIMPLE_CC_MEMORY_MAX_INJECTED_CHARS", "12000")
)
MEMORY_INDEX_MAX_CHARS = int(
    os.getenv("SIMPLE_CC_MEMORY_INDEX_MAX_CHARS", "8000")
)
MEMORY_CONSOLIDATE_THRESHOLD = int(
    os.getenv("SIMPLE_CC_MEMORY_CONSOLIDATE_THRESHOLD", "30")
)
MEMORY_CONSOLIDATE_TARGET = int(
    os.getenv("SIMPLE_CC_MEMORY_CONSOLIDATE_TARGET", "24")
)
MEMORY_CONSOLIDATE_COOLDOWN_SECONDS = int(
    os.getenv("SIMPLE_CC_MEMORY_CONSOLIDATE_COOLDOWN", "86400")
)
CONTINUATION_PROMPT = (
    "Continue from the previous response. Do not repeat completed work."
)


def configure_workspace(workspace: Path | str) -> Path:
    """Select the S20 workspace before stateful modules are used."""
    global WORKDIR, SKILLS_DIR, TRANSCRIPT_DIR, TOOL_RESULTS_DIR
    global TASKS_DIR, MAILBOX_DIR, MEMORY_DIR, MEMORY_INDEX, DURABLE_PATH

    WORKDIR = Path(workspace).expanduser().resolve()
    SKILLS_DIR = WORKDIR / "skills"
    TRANSCRIPT_DIR = WORKDIR / ".transcripts"
    TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
    TASKS_DIR = WORKDIR / ".tasks"
    MAILBOX_DIR = WORKDIR / ".mailboxes"
    MEMORY_DIR = WORKDIR / ".memory"
    MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
    DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"
    return WORKDIR


@dataclass(frozen=True)
class Settings:
    workspace: Path
    state_dir: Path
    tasks_dir: Path
    memory_dir: Path
    mailboxes_dir: Path
    transcripts_dir: Path
    outputs_dir: Path
    skills_dir: Path
    api_key: str
    model: str
    base_url: str = "https://api.siliconflow.cn/v1"
    max_rounds: int = 40
    max_tokens: int = 8192
    idle_poll_seconds: float = 1.0
    idle_timeout_seconds: float = 30.0

    @classmethod
    def from_env(
        cls,
        workspace: Path | str,
        model_override: str | None = None,
        *,
        create_dirs: bool = True,
    ) -> "Settings":
        workspace_path = Path(workspace).expanduser().resolve()
        if create_dirs:
            workspace_path.mkdir(parents=True, exist_ok=True)
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        model = (model_override or os.getenv("SILICONFLOW_MODEL", "")).strip()
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is required")
        if not model:
            raise ValueError("SILICONFLOW_MODEL is required")
        state = workspace_path / ".simple_cc"
        paths = {
            "tasks_dir": state / "tasks",
            "memory_dir": state / "memory",
            "mailboxes_dir": state / "mailboxes",
            "transcripts_dir": state / "transcripts",
            "outputs_dir": state / "outputs",
            "skills_dir": state / "skills",
        }
        if create_dirs:
            state.mkdir(parents=True, exist_ok=True)
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
        return cls(
            workspace=workspace_path,
            state_dir=state,
            api_key=api_key,
            model=model,
            base_url=os.getenv(
                "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
            ).rstrip("/"),
            max_rounds=int(os.getenv("SIMPLE_CC_MAX_ROUNDS", "40")),
            max_tokens=int(os.getenv("SIMPLE_CC_MAX_TOKENS", "8192")),
            idle_poll_seconds=float(os.getenv("SIMPLE_CC_IDLE_POLL", "1")),
            idle_timeout_seconds=float(os.getenv("SIMPLE_CC_IDLE_TIMEOUT", "30")),
            **paths,
        )
