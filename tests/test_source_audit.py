from simple_cc.provider import ProviderResponse, TextBlock, ToolUseBlock


def test_provider_boundary_uses_s20_content_blocks_instead_of_openai_response_objects():
    response = ProviderResponse(
        content=[
            TextBlock(text="I will inspect the file."),
            ToolUseBlock(id="call_1", name="read_file", input={"path": "README.md"}),
        ],
        stop_reason="tool_use",
    )

    assert response.stop_reason == "tool_use"
    assert [(block.type, getattr(block, "id", None)) for block in response.content] == [
        ("text", None),
        ("tool_use", "call_1"),
    ]
