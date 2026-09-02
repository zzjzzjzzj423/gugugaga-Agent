import json
from types import SimpleNamespace

import pytest

from gugugaga.config import Settings
from gugugaga.models import ToolCall, ToolSpec
from gugugaga.provider import normalize_tool_call, to_openai_tool


def test_settings_builds_state_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    monkeypatch.delenv("GUGUGAGA_MEMORY_CONSOLIDATION_TIMEOUT", raising=False)

    settings = Settings.from_env(tmp_path)

    assert settings.base_url == "https://api.siliconflow.cn/v1"
    assert settings.state_dir == tmp_path / ".gugugaga"
    assert settings.tasks_dir.exists()
    assert settings.mailboxes_dir.exists()
    assert settings.memory_consolidation_timeout_seconds == 90
    assert settings.memory_evidence_hot_exchanges == 30
    assert settings.memory_intent_gate_enabled is True
    assert settings.memory_intent_gate_model is None
    assert settings.memory_intent_gate_timeout_seconds == 5


def test_settings_reads_memory_intent_gate_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    monkeypatch.setenv("GUGUGAGA_MEMORY_INTENT_GATE_ENABLED", "false")
    monkeypatch.setenv("GUGUGAGA_MEMORY_INTENT_GATE_MODEL", "gate-model")
    monkeypatch.setenv("GUGUGAGA_MEMORY_INTENT_GATE_TIMEOUT", "7")

    settings = Settings.from_env(tmp_path)

    assert settings.memory_intent_gate_enabled is False
    assert settings.memory_intent_gate_model == "gate-model"
    assert settings.memory_intent_gate_timeout_seconds == 7


def test_settings_reads_evidence_hot_exchange_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    monkeypatch.setenv("GUGUGAGA_MEMORY_EVIDENCE_HOT_EXCHANGES", "12")

    settings = Settings.from_env(tmp_path)

    assert settings.memory_evidence_hot_exchanges == 12


def test_settings_loads_workspace_dotenv_without_overriding_process(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "SILICONFLOW_API_KEY=dotenv-key\n"
        "SILICONFLOW_MODEL=dotenv-model\n"
        "GUGUGAGA_MEMORY_EMBEDDING_MODEL=BAAI/bge-m3\n",
        encoding="utf-8",
    )
    for name in (
        "SILICONFLOW_API_KEY",
        "SILICONFLOW_MODEL",
        "GUGUGAGA_MEMORY_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env(tmp_path)

    assert settings.api_key == "dotenv-key"
    assert settings.model == "dotenv-model"
    assert settings.memory_embedding_model == "BAAI/bge-m3"

    monkeypatch.setenv("GUGUGAGA_MEMORY_EMBEDDING_MODEL", "process-embedding")
    assert Settings.from_env(tmp_path).memory_embedding_model == "process-embedding"


def test_settings_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.delenv("SILICONFLOW_MODEL", raising=False)

    with pytest.raises(ValueError, match="SILICONFLOW_MODEL"):
        Settings.from_env(tmp_path)


def test_normalize_tool_call_decodes_json_arguments():
    raw = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="read_file", arguments=json.dumps({"path": "README.md"})
        ),
    )

    assert normalize_tool_call(raw) == ToolCall(
        id="call_1", name="read_file", arguments={"path": "README.md"}
    )


def test_to_openai_tool_wraps_function_schema():
    spec = ToolSpec(
        name="read_file",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )

    converted = to_openai_tool(spec)

    assert converted["type"] == "function"
    assert converted["function"]["name"] == "read_file"
    assert converted["function"]["parameters"] == spec.parameters
