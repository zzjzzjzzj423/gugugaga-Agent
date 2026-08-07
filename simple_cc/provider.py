from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .config import Settings
from .models import ChatProvider, ToolCall, ToolSpec


class ContextLengthError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: str = field(default="text", init=False)


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = field(default="tool_use", init=False)


@dataclass(frozen=True)
class ProviderResponse:
    content: list[TextBlock | ToolUseBlock]
    stop_reason: str


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    return "".join(
        str(_value(block, "text", ""))
        for block in content
        if _value(block, "type") == "text"
    )


def to_openai_tool(spec: ToolSpec | dict[str, Any]) -> dict[str, Any]:
    name = _value(spec, "name")
    description = _value(spec, "description", "")
    parameters = _value(spec, "input_schema", _value(spec, "parameters", {}))
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message["content"]
    result: dict[str, Any] = {
        "role": "assistant",
        "content": _text_from_content(content),
    }
    tool_calls = []
    for block in content:
        if _value(block, "type") != "tool_use":
            continue
        tool_calls.append({
            "id": _value(block, "id"),
            "type": "function",
            "function": {
                "name": _value(block, "name"),
                "arguments": json.dumps(_value(block, "input", {})),
            },
        })
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def to_openai_messages(messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    """Render S20's Anthropic-shaped history for Chat Completions."""
    rendered: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        content = message.get("content")
        role = message.get("role")
        if role == "assistant" and isinstance(content, list):
            rendered.append(_assistant_message(message))
            continue
        if role == "user" and isinstance(content, list):
            text_blocks: list[Any] = []
            for block in content:
                if _value(block, "type") != "tool_result":
                    text_blocks.append(block)
                    continue
                if text_blocks:
                    rendered.append({"role": "user", "content": _text_from_content(text_blocks)})
                    text_blocks = []
                rendered.append({
                    "role": "tool",
                    "tool_call_id": _value(block, "tool_use_id"),
                    "content": _text_from_content(_value(block, "content", "")),
                })
            if text_blocks:
                rendered.append({"role": "user", "content": _text_from_content(text_blocks)})
            continue
        rendered.append(dict(message))
    return rendered


def normalize_tool_call(raw: Any) -> ToolCall:
    try:
        arguments = json.loads(raw.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {"_raw": raw.function.arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return ToolCall(id=raw.id, name=raw.function.name, arguments=arguments)


def _tool_use_block(raw: Any) -> ToolUseBlock:
    call = normalize_tool_call(raw)
    return ToolUseBlock(id=call.id, name=call.name, input=call.arguments)


def _stop_reason(finish_reason: str | None) -> str:
    return {
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "stop": "end_turn",
    }.get(finish_reason or "", finish_reason or "end_turn")


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

    def create(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolSpec | dict[str, Any]],
        max_tokens: int,
        model: str | None = None,
    ) -> ProviderResponse:
        """Call SiliconFlow and return the S20 content-block response shape."""
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=model or self.settings.model,
                    messages=to_openai_messages(messages, system),
                    tools=[to_openai_tool(spec) for spec in tools] or None,
                    max_tokens=max_tokens,
                    stream=False,
                )
                choice = response.choices[0]
                message = choice.message
                content: list[TextBlock | ToolUseBlock] = []
                if message.content:
                    content.append(TextBlock(text=message.content))
                content.extend(_tool_use_block(call) for call in (message.tool_calls or []))
                return ProviderResponse(content=content, stop_reason=_stop_reason(choice.finish_reason))
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
