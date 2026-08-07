from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

    def as_assistant_message(self) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        if self.content:
            blocks.append({"type": "text", "text": self.content})
        blocks.extend(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
            for call in self.tool_calls
        )
        return {"role": "assistant", "content": blocks}


class ChatProvider(Protocol):
    def create(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolSpec | dict[str, Any]],
        max_tokens: int = 8192,
        model: str | None = None,
    ) -> Any: ...
