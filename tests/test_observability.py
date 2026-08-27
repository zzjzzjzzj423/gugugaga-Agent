from __future__ import annotations

import json
import sqlite3

from gugugaga.__main__ import build_runtime
from gugugaga.config import Settings
from gugugaga.observability import Observer, RecordingSystem, event_scope
from gugugaga.provider import ProviderResponse, TextBlock, ToolUseBlock
from gugugaga import subagents
from tests.fakes import ScriptedProvider


def make_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    return Settings.from_env(tmp_path)


def test_observer_subscribe_unsubscribe_redacts_and_isolates_failures():
    observer = Observer()
    received = []
    observer.subscribe("*", lambda event: (_ for _ in ()).throw(RuntimeError("boom")))
    unsubscribe = observer.subscribe("tool", received.append)

    with event_scope(observer, session_id="s1", turn_id="t1"):
        observer.notify(
            "tool",
            {
                "tool": "demo",
                "args": {
                    "password": "should-not-appear",
                    "nested": {"api_key": "also-secret"},
                },
            },
        )

    assert received[0]["session_id"] == "s1"
    assert received[0]["args"]["password"] == "[REDACTED]"
    assert received[0]["args"]["nested"]["api_key"] == "[REDACTED]"
    unsubscribe()
    observer.notify("tool", {"tool": "ignored"})
    assert len(received) == 1


def test_runtime_records_trace_usage_chat_log_and_live_events(tmp_path, monkeypatch):
    responses = [
        ProviderResponse(
            content=[
                ToolUseBlock(
                    id="toolu_glob",
                    name="glob",
                    input={"pattern": "*.md"},
                )
            ],
            stop_reason="tool_use",
            usage={"input_tokens": 12, "output_tokens": 3},
            model="recording-model",
            provider="test-provider",
        ),
        ProviderResponse(
            content=[TextBlock(text="Done.")],
            stop_reason="end_turn",
            usage={"input_tokens": 20, "output_tokens": 2},
            model="recording-model",
            provider="test-provider",
        ),
    ]
    app = build_runtime(
        make_settings(tmp_path, monkeypatch),
        provider=ScriptedProvider(responses),
    )
    live_events = []
    unsubscribe = app.runtime.recording.observer.subscribe("*", live_events.append)
    try:
        assert app.runtime.run_turn("Find markdown files.", source="dashboard") == "Done."
    finally:
        unsubscribe()
        app.close()

    assert [event["type"] for event in live_events] == [
        "turn_start",
        "context",
        "llm",
        "tool",
        "context",
        "llm",
        "turn_end",
    ]
    assert all("system" not in event for event in live_events)

    trace_files = list((tmp_path / ".gugugaga" / "traces").glob("*.jsonl"))
    trace = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in trace] == [
        "turn_start",
        "context",
        "llm",
        "tool",
        "context",
        "llm",
        "turn_end",
    ]

    usage = [
        json.loads(line)
        for line in (tmp_path / ".gugugaga" / "usage.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(item["input_tokens"], item["output_tokens"]) for item in usage] == [
        (12, 3),
        (20, 2),
    ]
    assert {item["call_type"] for item in usage} == {"agent"}

    with sqlite3.connect(tmp_path / ".gugugaga" / "state.db") as connection:
        rows = connection.execute(
            "SELECT role, content, source, meta FROM chat_log ORDER BY id"
        ).fetchall()
    assert [(role, content, source) for role, content, source, _ in rows] == [
        ("user", "Find markdown files.", "dashboard"),
        ("assistant", "Done.", "dashboard"),
    ]
    meta = json.loads(rows[1][3])
    assert meta["iterations"] == 2
    assert meta["model"] == "recording-model"
    assert meta["provider"] == "test-provider"
    assert meta["tools"][0]["tool"] == "glob"
    assert meta["context"]["mode"] == "cc"


def test_turn_error_is_traced_without_masking_original_error(tmp_path):
    recording = RecordingSystem(tmp_path / ".gugugaga")
    try:
        with recording.start_turn(
            session_id="session-error",
            user_message="fail",
            source="test",
        ):
            raise ValueError("original failure")
    except ValueError as error:
        assert str(error) == "original failure"
    else:
        raise AssertionError("the original exception must propagate")

    trace_file = next((tmp_path / ".gugugaga" / "traces").glob("*.jsonl"))
    events = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["turn_start", "turn_error"]


def test_subagent_events_keep_parent_turn_context(tmp_path, monkeypatch):
    recording = RecordingSystem(tmp_path / ".gugugaga")
    provider = ScriptedProvider(
        [
            ProviderResponse(
                content=[TextBlock(text="Subtask done.")],
                stop_reason="end_turn",
                usage={"input_tokens": 7, "output_tokens": 4},
                model="sub-model",
                provider="test-provider",
            )
        ]
    )
    monkeypatch.setattr(subagents, "client", provider)
    events = []
    recording.observer.subscribe("*", events.append)

    with event_scope(
        recording.observer,
        session_id="parent-session",
        turn_id="parent-turn",
    ):
        assert subagents.spawn_subagent("Do the subtask") == "Subtask done."

    assert [event["type"] for event in events] == [
        "subagent_start",
        "llm",
        "subagent_end",
    ]
    assert {event["session_id"] for event in events} == {"parent-session"}
    assert {event["turn_id"] for event in events} == {"parent-turn"}
    assert {event["agent_type"] for event in events} == {"subagent"}
    usage = json.loads(
        (tmp_path / ".gugugaga" / "usage.jsonl").read_text(encoding="utf-8").strip()
    )
    assert usage["call_type"] == "subagent"
