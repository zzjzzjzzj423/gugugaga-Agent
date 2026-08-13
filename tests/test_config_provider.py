import json
from types import SimpleNamespace

import pytest

from simple_cc.config import Settings
from simple_cc.models import ToolCall, ToolSpec
from simple_cc.provider import (
    ProviderUsage,
    SiliconFlowProvider,
    normalize_tool_call,
    to_openai_tool,
)


class _Client:
    def __init__(self, response):
        self.response = response
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: self.response)
        )


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


def test_provider_preserves_usage_request_id_and_attempt_count(tmp_path):
    monkey_settings = Settings(
        workspace=tmp_path,
        state_dir=tmp_path / ".simple_cc",
        tasks_dir=tmp_path / ".simple_cc/tasks",
        memory_dir=tmp_path / ".simple_cc/memory",
        mailboxes_dir=tmp_path / ".simple_cc/mailboxes",
        transcripts_dir=tmp_path / ".simple_cc/transcripts",
        outputs_dir=tmp_path / ".simple_cc/outputs",
        skills_dir=tmp_path / ".simple_cc/skills",
        api_key="key",
        model="model",
    )
    response = SimpleNamespace(
        id="req-1",
        usage=SimpleNamespace(
            prompt_tokens=11, completion_tokens=7, total_tokens=18
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", tool_calls=[]),
            )
        ],
    )
    result = SiliconFlowProvider(monkey_settings, client=_Client(response)).create(
        messages=[], system="", tools=[], max_tokens=10
    )
    assert result.usage == ProviderUsage(11, 7, 18)
    assert result.request_id == "req-1"
    assert result.attempts == 1


def test_provider_reports_missing_usage_as_unknown(tmp_path):
    settings = Settings(
        workspace=tmp_path,
        state_dir=tmp_path / ".simple_cc",
        tasks_dir=tmp_path / ".simple_cc/tasks",
        memory_dir=tmp_path / ".simple_cc/memory",
        mailboxes_dir=tmp_path / ".simple_cc/mailboxes",
        transcripts_dir=tmp_path / ".simple_cc/transcripts",
        outputs_dir=tmp_path / ".simple_cc/outputs",
        skills_dir=tmp_path / ".simple_cc/skills",
        api_key="key",
        model="model",
    )
    response = SimpleNamespace(
        id=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", tool_calls=[]),
            )
        ],
    )
    result = SiliconFlowProvider(settings, client=_Client(response)).create(
        messages=[], system="", tools=[], max_tokens=10
    )
    assert result.usage == ProviderUsage(None, None, None)
