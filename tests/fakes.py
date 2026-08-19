import copy

from simple_cc.models import ModelResponse
from simple_cc.provider import ProviderResponse, TextBlock, ToolUseBlock


class ScriptedProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.requests = []

    def queue(self, response: ModelResponse):
        self.responses.append(response)

    def create(self, messages, system, tools, max_tokens=8192, model=None):
        self.requests.append({
            "system": system,
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "max_tokens": max_tokens,
            "model": model,
        })
        if not self.responses:
            raise AssertionError("ScriptedProvider has no queued response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, ProviderResponse):
            return response
        assert isinstance(response, ModelResponse)
        content = []
        if response.content:
            content.append(TextBlock(text=response.content))
        content.extend(
            ToolUseBlock(call.id, call.name, call.arguments)
            for call in response.tool_calls
        )
        stop_reason = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }.get(response.finish_reason, response.finish_reason)
        return ProviderResponse(content=content, stop_reason=stop_reason)
