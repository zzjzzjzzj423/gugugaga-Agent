import json
from types import SimpleNamespace

import pytest

from simple_cc.config import Settings
from simple_cc.models import ToolCall, ToolSpec
from simple_cc.provider import normalize_tool_call, to_openai_tool


def test_settings_builds_state_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")

    settings = Settings.from_env(tmp_path)

    assert settings.base_url == "https://api.siliconflow.cn/v1"
    assert settings.state_dir == tmp_path / ".simple_cc"
    assert settings.tasks_dir.exists()
    assert settings.mailboxes_dir.exists()


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
