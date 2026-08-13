from __future__ import annotations

import json

import pytest

from simple_cc import agent, config
from simple_cc.evidence import (
    CutoffMismatch,
    canonicalize_url,
    link_final_answer_sources,
    prepare_research_arguments,
)
from simple_cc.models import ModelResponse, ToolCall
from simple_cc.trace import TraceRecorder
from tests.fakes import ScriptedProvider


def test_cutoff_is_injected_and_mismatch_is_rejected():
    prepared = prepare_research_arguments(
        "web_search", {"query": "rates"}, required_cutoff="2025-05-01"
    )
    assert prepared.arguments["cutoff"] == "2025-05-01"
    assert prepared.decision == "injected"

    with pytest.raises(CutoffMismatch):
        prepare_research_arguments(
            "web_fetch",
            {"url": "https://example.com", "cutoff": "2025-05-02"},
            required_cutoff="2025-05-01",
        )


def test_canonical_url_and_citation_linkage_are_deterministic():
    canonical = canonicalize_url(
        "HTTPS://Example.COM:443/Report?b=2&a=1#section"
    )
    assert canonical == "https://example.com/Report?a=1&b=2"
    linked = link_final_answer_sources(
        "See [report](https://example.com/Report?a=1&b=2).",
        {canonical: "src_123"},
    )
    assert linked["matched_source_ids"] == ["src_123"]
    assert linked["unmatched_citations"] == []


def test_source_runtime_traces_tool_and_injects_cutoff(tmp_path, monkeypatch):
    old_workspace = config.WORKDIR
    config.configure_workspace(tmp_path / "workspace")
    monkeypatch.setattr(config, "MEMORY_ENABLED", False)
    seen = []

    def search(**arguments):
        seen.append(arguments)
        return json.dumps(
            {
                "ok": True,
                "operation": "search",
                "query": arguments["query"],
                "cutoff": arguments["cutoff"],
                "results": [],
            }
        )

    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                [ToolCall("search-1", "web_search", {"query": "rates"})],
                "tool_calls",
            ),
            ModelResponse("Research complete", [], "stop"),
        ]
    )
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(
        task_id="task-1", question="Research rates", cutoff="2025-05-01", metadata={}
    )
    runtime = agent.SourceRuntime(
        provider,
        recorder=recorder,
        tool_definitions=[
            {
                "name": "web_search",
                "description": "search",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        tool_handlers={"web_search": search},
        memory_enabled=False,
    )
    try:
        answer = runtime.run_turn(
            "Research rates", task_id="task-1", cutoff="2025-05-01"
        )
    finally:
        config.configure_workspace(old_workspace)

    assert answer == "Research complete"
    assert runtime.last_outcome.status == "completed"
    assert seen == [{"query": "rates", "cutoff": "2025-05-01"}]
    rows = [
        json.loads(line)
        for line in recorder.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [row["event_type"] for row in rows]
    assert "tool_requested" in event_types
    assert "cutoff_validation" in event_types
    assert "permission_decision" in event_types
    assert "tool_started" in event_types
    assert "tool_result" in event_types
    assert event_types[-1] == "final_answer"
