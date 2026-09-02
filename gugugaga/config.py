from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

WORKDIR = Path.cwd()
MODEL = os.getenv("SILICONFLOW_MODEL", "")
PRIMARY_MODEL = MODEL
FALLBACK_MODEL = os.getenv("SILICONFLOW_FALLBACK_MODEL")

SKILLS_DIR = WORKDIR / ".gugugaga" / "skills"
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
MEMORY_EXTRACTION_INTERVAL = max(
    1, int(os.getenv("GUGUGAGA_MEMORY_EXTRACTION_INTERVAL", "10"))
)
CONTINUATION_PROMPT = (
    "Continue from the previous response. Do not repeat completed work."
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def configure_workspace(workspace: Path | str) -> Path:
    """Select the S20 workspace before stateful modules are used."""
    global WORKDIR, SKILLS_DIR, TRANSCRIPT_DIR, TOOL_RESULTS_DIR
    global TASKS_DIR, MAILBOX_DIR, MEMORY_DIR, MEMORY_INDEX, DURABLE_PATH

    WORKDIR = Path(workspace).expanduser().resolve()
    SKILLS_DIR = WORKDIR / ".gugugaga" / "skills"
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
    context_mode: str = "cc"
    context_mode_source: str = "default"
    context_window_tokens: int = 131_072
    token_counter_id: str = "gugugaga_estimator_v1"
    token_counter_version: str = "v1"
    hermes_threshold_ratio: float = 0.50
    hermes_target_ratio: float = 0.20
    pi_reserve_tokens: int = 16_384
    pi_keep_recent_tokens: int = 20_000
    memory_enabled: bool = True
    memory_explicit_enabled: bool = True
    memory_consolidation_enabled: bool = True
    memory_consolidation_exchange_threshold: int = 6
    memory_consolidation_model: str | None = None
    memory_consolidation_timeout_seconds: int = 30
    memory_consolidation_lease_seconds: int = 600
    memory_consolidation_max_facts: int = 10
    memory_consolidation_min_importance: float = 0.8
    memory_recall_token_budget: int = 2000

    @classmethod
    def from_env(
        cls, workspace: Path | str, model_override: str | None = None
    ) -> "Settings":
        workspace_path = Path(workspace).expanduser().resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        model = (model_override or os.getenv("SILICONFLOW_MODEL", "")).strip()
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is required")
        if not model:
            raise ValueError("SILICONFLOW_MODEL is required")
        state = workspace_path / ".gugugaga"
        paths = {
            "tasks_dir": state / "tasks",
            "memory_dir": state / "memory",
            "mailboxes_dir": state / "mailboxes",
            "transcripts_dir": state / "transcripts",
            "outputs_dir": state / "outputs",
            "skills_dir": state / "skills",
        }
        state.mkdir(parents=True, exist_ok=True)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        threshold = int(os.getenv("GUGUGAGA_MEMORY_CONSOLIDATION_EXCHANGES", "6"))
        timeout_seconds = int(os.getenv("GUGUGAGA_MEMORY_CONSOLIDATION_TIMEOUT", "30"))
        lease_seconds = int(os.getenv("GUGUGAGA_MEMORY_CONSOLIDATION_LEASE", "600"))
        max_facts = int(os.getenv("GUGUGAGA_MEMORY_CONSOLIDATION_MAX_FACTS", "10"))
        min_importance = float(
            os.getenv("GUGUGAGA_MEMORY_CONSOLIDATION_MIN_IMPORTANCE", "0.8")
        )
        recall_budget = int(os.getenv("GUGUGAGA_MEMORY_RECALL_TOKENS", "2000"))
        if not 1 <= threshold <= 100:
            raise ValueError("GUGUGAGA_MEMORY_CONSOLIDATION_EXCHANGES must be 1-100")
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("GUGUGAGA_MEMORY_CONSOLIDATION_TIMEOUT must be 1-120")
        if lease_seconds <= timeout_seconds:
            raise ValueError("GUGUGAGA_MEMORY_CONSOLIDATION_LEASE must exceed timeout")
        if not 0 <= max_facts <= 20:
            raise ValueError("GUGUGAGA_MEMORY_CONSOLIDATION_MAX_FACTS must be 0-20")
        if not 0 <= min_importance <= 1:
            raise ValueError(
                "GUGUGAGA_MEMORY_CONSOLIDATION_MIN_IMPORTANCE must be 0-1"
            )
        if not 0 <= recall_budget <= 8000:
            raise ValueError("GUGUGAGA_MEMORY_RECALL_TOKENS must be 0-8000")
        return cls(
            workspace=workspace_path,
            state_dir=state,
            api_key=api_key,
            model=model,
            base_url=os.getenv(
                "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
            ).rstrip("/"),
            max_rounds=int(os.getenv("GUGUGAGA_MAX_ROUNDS", "40")),
            max_tokens=int(os.getenv("GUGUGAGA_MAX_TOKENS", "8192")),
            idle_poll_seconds=float(os.getenv("GUGUGAGA_IDLE_POLL", "1")),
            idle_timeout_seconds=float(os.getenv("GUGUGAGA_IDLE_TIMEOUT", "30")),
            memory_enabled=_env_bool("GUGUGAGA_MEMORY_ENABLED", True),
            memory_explicit_enabled=_env_bool(
                "GUGUGAGA_MEMORY_EXPLICIT_ENABLED", True
            ),
            memory_consolidation_enabled=_env_bool(
                "GUGUGAGA_MEMORY_CONSOLIDATION_ENABLED", True
            ),
            memory_consolidation_exchange_threshold=threshold,
            memory_consolidation_model=(
                os.getenv("GUGUGAGA_MEMORY_CONSOLIDATION_MODEL", "").strip()
                or None
            ),
            memory_consolidation_timeout_seconds=timeout_seconds,
            memory_consolidation_lease_seconds=lease_seconds,
            memory_consolidation_max_facts=max_facts,
            memory_consolidation_min_importance=min_importance,
            memory_recall_token_budget=recall_budget,
            **paths,
        )
