from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from .models import ChatProvider
from .trace import ArtifactRef, TraceRecorder, current_run_context


@dataclass(frozen=True)
class ModelCallContext:
    kind: str = "agent"
    parent_span_id: str | None = None


_MODEL_CALL: contextvars.ContextVar[ModelCallContext] = contextvars.ContextVar(
    "simple_cc_model_call", default=ModelCallContext()
)


@contextlib.contextmanager
def model_call_scope(
    kind: str, parent_span_id: str | None = None
) -> Iterator[None]:
    token = _MODEL_CALL.set(ModelCallContext(kind, parent_span_id))
    try:
        yield
    finally:
        _MODEL_CALL.reset(token)


@dataclass
class ToolCapture:
    recorder: TraceRecorder
    span_id: str
    artifacts: list[ArtifactRef] = field(default_factory=list)

    def store(
        self,
        content: Any,
        *,
        media_type: str,
        source: str,
        suffix: str,
    ) -> ArtifactRef:
        ref = self.recorder.store_artifact(
            content, media_type=media_type, source=source, suffix=suffix
        )
        self.artifacts.append(ref)
        return ref


_TOOL_CAPTURE: contextvars.ContextVar[ToolCapture | None] = contextvars.ContextVar(
    "simple_cc_tool_capture", default=None
)


@contextlib.contextmanager
def bind_tool_capture(capture: ToolCapture | None) -> Iterator[None]:
    token = _TOOL_CAPTURE.set(capture)
    try:
        yield
    finally:
        _TOOL_CAPTURE.reset(token)


def capture_tool_artifact(
    content: Any, *, media_type: str, source: str, suffix: str
) -> ArtifactRef | None:
    capture = _TOOL_CAPTURE.get()
    if capture is None:
        return None
    return capture.store(
        content, media_type=media_type, source=source, suffix=suffix
    )


class TracingProvider:
    def __init__(self, delegate: ChatProvider):
        self.delegate = delegate

    def create(
        self,
        messages,
        system,
        tools,
        max_tokens=8192,
        model=None,
    ):
        run = current_run_context()
        if run is None:
            return self.delegate.create(messages, system, tools, max_tokens, model)
        call = _MODEL_CALL.get()
        span_id = f"llm_{uuid.uuid4().hex}"
        request_ref = run.recorder.store_artifact(
            {
                "system": system,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
                "model": model,
            },
            media_type="application/json",
            source="llm_request",
            suffix=".json",
        )
        prompt_sha256 = hashlib.sha256(system.encode("utf-8")).hexdigest()
        tool_schema_sha256 = hashlib.sha256(
            json.dumps(
                tools,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        run.recorder.set_initial_request_hashes(
            prompt_sha256=prompt_sha256,
            tool_schema_sha256=tool_schema_sha256,
        )
        run.recorder.record(
            "llm_request_started",
            {
                "call_kind": call.kind,
                "model": model,
                "max_tokens": max_tokens,
                "request_artifact": request_ref.as_dict(),
                "prompt_sha256": prompt_sha256,
                "tool_schema_sha256": tool_schema_sha256,
            },
            span_id=span_id,
            parent_span_id=call.parent_span_id or run.parent_span_id,
            agent_id=run.agent_id,
        )
        started = time.monotonic()
        try:
            response = self.delegate.create(
                messages, system, tools, max_tokens, model
            )
        except Exception as error:
            run.recorder.record(
                "llm_error",
                {
                    "exception_class": type(error).__name__,
                    "message": str(error),
                    "attempts": getattr(error, "attempts", None),
                    "latency_ms": round(
                        (time.monotonic() - started) * 1000, 3
                    ),
                },
                span_id=span_id,
                parent_span_id=call.parent_span_id or run.parent_span_id,
                agent_id=run.agent_id,
            )
            raise
        response_ref = run.recorder.store_artifact(
            response.content,
            media_type="application/json",
            source="llm_response",
            suffix=".json",
        )
        run.recorder.record(
            "llm_response",
            {
                "call_kind": call.kind,
                "stop_reason": response.stop_reason,
                "usage": asdict(response.usage),
                "request_id": response.request_id,
                "attempts": response.attempts,
                "latency_ms": round(
                    (time.monotonic() - started) * 1000, 3
                ),
                "response_artifact": response_ref.as_dict(),
            },
            span_id=span_id,
            parent_span_id=call.parent_span_id or run.parent_span_id,
            agent_id=run.agent_id,
        )
        return response
