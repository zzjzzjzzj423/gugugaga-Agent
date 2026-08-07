from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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
        cls, workspace: Path | str, model_override: str | None = None
    ) -> "Settings":
        load_dotenv(override=False)
        workspace_path = Path(workspace).expanduser().resolve()
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

