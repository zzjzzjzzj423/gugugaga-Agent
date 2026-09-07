import json
from types import SimpleNamespace

from gugugaga.config import Settings
from gugugaga.provider import SiliconFlowProvider


class ScriptedOpenAIClient:
    def __init__(self, completion):
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.completion = completion

    def _create(self, **request):
        self.requests.append(request)
        return self.completion


def settings(tmp_path):
    return Settings(
        workspace=tmp_path,
        state_dir=tmp_path / ".gugugaga",
        tasks_dir=tmp_path / ".gugugaga" / "tasks",
        memory_dir=tmp_path / ".gugugaga" / "memory",
        mailboxes_dir=tmp_path / ".gugugaga" / "mailboxes",
        transcripts_dir=tmp_path / ".gugugaga" / "transcripts",
        outputs_dir=tmp_path / ".gugugaga" / "outputs",
        skills_dir=tmp_path / ".gugugaga" / "skills",
        api_key="test-key",
        model="default-model",
    )


def test_create_returns_ordered_anthropic_content_blocks_for_text_and_function_calls(tmp_path):
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="I will inspect both files.",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_read",
                            function=SimpleNamespace(
                                name="read_file",
                                arguments=json.dumps({"path": "README.md"}),
                            ),
                        ),
                        SimpleNamespace(
                            id="call_glob",
                            function=SimpleNamespace(
                                name="glob",
                                arguments=json.dumps({"pattern": "*.py"}),
                            ),
                        ),
                    ],
                ),
            )
        ]
    )
    client = ScriptedOpenAIClient(completion)
    provider = SiliconFlowProvider(settings(tmp_path), client=client)

    response = provider.create(
        messages=[{"role": "user", "content": "inspect the repository"}],
        system="You are a coding agent.",
        tools=[
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        max_tokens=321,
        model="chosen-model",
    )

    assert response.stop_reason == "tool_use"
    assert [block.type for block in response.content] == ["text", "tool_use", "tool_use"]
    assert response.content[0].text == "I will inspect both files."
    assert [(block.id, block.name, block.input) for block in response.content[1:]] == [
        ("call_read", "read_file", {"path": "README.md"}),
        ("call_glob", "glob", {"pattern": "*.py"}),
    ]
    assert client.requests[0]["model"] == "chosen-model"
    assert client.requests[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_create_renders_s20_tool_history_as_openai_assistant_and_tool_messages(tmp_path):
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="Both results received.", tool_calls=[]),
            )
        ]
    )
    client = ScriptedOpenAIClient(completion)
    provider = SiliconFlowProvider(settings(tmp_path), client=client)

    provider.create(
        system="You are a coding agent.",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect these files."},
                    {
                        "type": "tool_use",
                        "id": "call_read",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                    {
                        "type": "tool_use",
                        "id": "call_glob",
                        "name": "glob",
                        "input": {"pattern": "*.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_read", "content": "contents"},
                    {"type": "tool_result", "tool_use_id": "call_glob", "content": "a.py\nb.py"},
                ],
            },
        ],
        tools=[],
        max_tokens=321,
    )

    request_messages = client.requests[0]["messages"]
    assert request_messages[0] == {"role": "system", "content": "You are a coding agent."}
    assert request_messages[1] == {
        "role": "assistant",
        "content": "I will inspect these files.",
        "tool_calls": [
            {
                "id": "call_read",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "README.md"}',
                },
            },
            {
                "id": "call_glob",
                "type": "function",
                "function": {
                    "name": "glob",
                    "arguments": '{"pattern": "*.py"}',
                },
            },
        ],
    }
    assert request_messages[2:] == [
        {"role": "tool", "tool_call_id": "call_read", "content": "contents"},
        {"role": "tool", "tool_call_id": "call_glob", "content": "a.py\nb.py"},
    ]


def test_internal_context_metadata_is_not_sent_to_openai(tmp_path):
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", tool_calls=[]),
            )
        ]
    )
    client = ScriptedOpenAIClient(completion)
    provider = SiliconFlowProvider(settings(tmp_path), client=client)

    provider.create(
        system="system",
        messages=[
            {
                "role": "user",
                "content": "hello",
                "message_id": "msg_internal",
                "_context_meta": {"synthetic": False},
            }
        ],
        tools=[],
        max_tokens=20,
    )

    assert client.requests[0]["messages"][1] == {
        "role": "user",
        "content": "hello",
    }


def test_create_can_disable_provider_thinking_mode(tmp_path):
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content='{"ok":true}', tool_calls=[]),
            )
        ]
    )
    client = ScriptedOpenAIClient(completion)
    provider = SiliconFlowProvider(
        settings(tmp_path),
        client=client,
        enable_thinking=False,
        temperature=0,
    )

    provider.create(
        system="Return JSON.",
        messages=[{"role": "user", "content": "status"}],
        tools=[],
        max_tokens=20,
    )

    assert client.requests[0]["extra_body"] == {"enable_thinking": False}
    assert client.requests[0]["temperature"] == 0
