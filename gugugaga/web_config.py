from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


_ALLOWED_FIELDS = {
    "model",
    "consolidation_model",
    "siliconflow_api_key",
    "tavily_api_key",
}
_ENV_NAMES = {
    "model": "SILICONFLOW_MODEL",
    "consolidation_model": "GUGUGAGA_MEMORY_CONSOLIDATION_MODEL",
    "siliconflow_api_key": "SILICONFLOW_API_KEY",
    "tavily_api_key": "TAVILY_API_KEY",
}


def _clean(value: Any, field: str, *, required: bool = False, limit: int = 1000) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    cleaned = value.strip()
    if required and not cleaned:
        raise ValueError(f"{field} is required")
    if len(cleaned) > limit or any(character in cleaned for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{field} is invalid")
    return cleaned


def _hint(value: str) -> str | None:
    return f"••••{value[-4:]}" if value else None


class WebConfiguration:
    """Workspace-local settings whose secrets are never returned to the browser."""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace).expanduser().resolve()
        self.path = self.workspace / ".gugugaga" / "web_config.json"
        self._lock = threading.RLock()
        self._base_environment = {
            field: os.getenv(environment, "").strip()
            for field, environment in _ENV_NAMES.items()
        }

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            key: item
            for key, item in value.items()
            if key in _ALLOWED_FIELDS and isinstance(item, str) and item
        }

    def apply_environment(self) -> None:
        with self._lock:
            stored = self._load()
            for field, environment in _ENV_NAMES.items():
                value = stored.get(field) or self._base_environment[field]
                if value:
                    os.environ[environment] = value
                else:
                    os.environ.pop(environment, None)

    def effective(self, model_override: str | None = None) -> dict[str, str]:
        with self._lock:
            stored = self._load()
            return {
                "model": stored.get("model")
                or (model_override or "").strip()
                or self._base_environment["model"],
                "consolidation_model": stored.get("consolidation_model")
                or self._base_environment["consolidation_model"],
                "siliconflow_api_key": stored.get("siliconflow_api_key")
                or self._base_environment["siliconflow_api_key"],
                "tavily_api_key": stored.get("tavily_api_key")
                or self._base_environment["tavily_api_key"],
            }

    def public(self, model_override: str | None = None) -> dict[str, Any]:
        value = self.effective(model_override)
        return {
            "model": value["model"],
            "consolidation_model": value["consolidation_model"],
            "siliconflow_api_key_configured": bool(value["siliconflow_api_key"]),
            "siliconflow_api_key_hint": _hint(value["siliconflow_api_key"]),
            "tavily_api_key_configured": bool(value["tavily_api_key"]),
            "tavily_api_key_hint": _hint(value["tavily_api_key"]),
        }

    def update(self, payload: dict[str, Any], model_override: str | None = None) -> dict[str, Any]:
        unexpected = set(payload) - _ALLOWED_FIELDS
        if unexpected:
            raise ValueError(f"unsupported configuration fields: {', '.join(sorted(unexpected))}")
        with self._lock:
            stored = self._load()
            stored["model"] = _clean(
                payload.get("model"), "model", required=True, limit=300
            )
            consolidation_model = _clean(
                payload.get("consolidation_model"),
                "consolidation_model",
                limit=300,
            )
            if consolidation_model:
                stored["consolidation_model"] = consolidation_model
            else:
                stored.pop("consolidation_model", None)
            for field in ("siliconflow_api_key", "tavily_api_key"):
                secret = _clean(payload.get(field), field)
                if secret:
                    stored[field] = secret

            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(stored, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
            temporary.replace(self.path)
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
            self.apply_environment()
            return self.public(model_override)
