from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from .config import Settings
from .models import ChatProvider, ModelResponse, ToolCall, ToolSpec


class ContextLengthError(RuntimeError):
    pass


def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def normalize_tool_call(raw: Any) -> ToolCall:
    try:
        arguments = json.loads(raw.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {"_raw": raw.function.arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return ToolCall(id=raw.id, name=raw.function.name, arguments=arguments)


def _status_code(error: Exception) -> int | None:
    return getattr(error, "status_code", None) or getattr(
        getattr(error, "response", None), "status_code", None
    )


def _is_context_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(token in text for token in ("context length", "too long", "maximum context"))


class SiliconFlowProvider(ChatProvider):
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self.client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=120,
        )

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_tokens: int = 8192,
    ) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=[{"role": "system", "content": system}, *messages],
                    tools=[to_openai_tool(spec) for spec in tools] or None,
                    max_tokens=max_tokens,
                    stream=False,
                )
                choice = response.choices[0]
                message = choice.message
                return ModelResponse(
                    content=message.content or "",
                    tool_calls=[normalize_tool_call(c) for c in (message.tool_calls or [])],
                    finish_reason=choice.finish_reason or "stop",
                )
            except Exception as error:
                if _is_context_error(error):
                    raise ContextLengthError(str(error)) from error
                last_error = error
                status = _status_code(error)
                if status != 429 and (status is None or status < 500):
                    raise
                if attempt == 3:
                    break
                time.sleep(min(2**attempt, 4))
        assert last_error is not None
        raise last_error

