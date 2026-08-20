from __future__ import annotations

import json

import pytest

from simple_cc import agent, config
from simple_cc.evidence import (
    CutoffMismatch,
    canonicalize_url,
    evidence_record_from_result,
    link_final_answer_sources,
    prepare_research_arguments,
    validate_research_final,
)
from simple_cc.models import ModelResponse, ToolCall
from simple_cc.research_models import (
    EVIDENCE_PDF_FRAGMENT_CHARS_MAX,
    EvidenceRegistry,
    ResearchPlan,
    ResearchRank,
)
from simple_cc.telemetry import capture_tool_artifact
from simple_cc.trace import RunContext, TraceRecorder, read_trace_lines
from tests.fakes import ScriptedProvider


def _light_plan_response() -> ModelResponse:
    return ModelResponse(json.dumps({
        "rank": "light",
        "directions": ["primary filings"],
        "reason": "bounded trace test",
    }))


def _failed_gate_response() -> ModelResponse:
    return ModelResponse(json.dumps({
        "directions": [{
            "direction": "primary filings",
            "covered": False,
            "source_ids": [],
            "reason": "no fetched evidence",
        }],
        "authorities": [],
        "gaps": ["no fetched evidence"],
    }))


def _unsupported_research_script(
    initial_research_responses: list[ModelResponse],
) -> list[ModelResponse]:
    return [
        _light_plan_response(),
        *initial_research_responses,
        _failed_gate_response(),
        ModelResponse("supplemental research notes"),
        _failed_gate_response(),
        ModelResponse("unsupported draft"),
        ModelResponse("unsupported rewrite"),
    ]


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
    assert canonicalize_url("https://例子.测试/报告") == (
        "https://xn--fsqu00a.xn--0zwm56d/报告"
    )
    assert canonicalize_url("https://[2001:db8::1]:443/report") == (
        "https://[2001:db8::1]/report"
    )


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
        _unsupported_research_script([
            ModelResponse(
                "",
                [ToolCall("search-1", "web_search", {"query": "rates"})],
                "tool_calls",
            ),
            ModelResponse("initial research notes", [], "stop"),
        ])
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
            "Research rates",
            task_id="task-1",
            cutoff="2025-05-01",
            run_metadata={"task_type": "research"},
        )
    finally:
        config.configure_workspace(old_workspace)

    assert answer.startswith("INSUFFICIENT_EVIDENCE")
    assert len(provider.requests) == 8
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


def test_tool_exception_has_exactly_one_terminal_trace_event(tmp_path):
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                [ToolCall("tool-1", "explode", {})],
                "tool_calls",
            ),
            ModelResponse("done", [], "stop"),
            ModelResponse("done", [], "stop"),
        ]
    )

    def explode():
        raise RuntimeError("boom")

    recorder = TraceRecorder(tmp_path / "run", run_id="run-error")
    recorder.start_run(
        task_id="task-error", question="test", cutoff=None, metadata={}
    )
    runtime = agent.SourceRuntime(
        provider,
        recorder=recorder,
        tool_definitions=[
            {
                "name": "explode",
                "description": "fail",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        tool_handlers={"explode": explode},
        memory_enabled=False,
    )
    runtime.run_turn("test", task_id="task-error", cutoff=None)

    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    terminal = [
        row["event_type"]
        for row in rows
        if row.get("span_id")
        and row["event_type"] in {"tool_error", "tool_result"}
    ]
    assert incomplete is False
    assert terminal == ["tool_error"]


def test_cutoff_mismatch_is_a_failed_terminal_tool_result(tmp_path):
    called = False

    def fetch(**arguments):
        nonlocal called
        called = True
        return "should not run"

    provider = ScriptedProvider(
        _unsupported_research_script([
            ModelResponse(
                "",
                [
                    ToolCall(
                        "fetch-1",
                        "web_fetch",
                        {
                            "url": "https://example.com/report",
                            "cutoff": "2025-05-02",
                        },
                    )
                ],
                "tool_calls",
            ),
            ModelResponse("initial research notes", [], "stop"),
        ])
    )
    recorder = TraceRecorder(tmp_path / "run", run_id="run-cutoff")
    recorder.start_run(
        task_id="task-cutoff", question="test", cutoff="2025-05-01", metadata={}
    )
    runtime = agent.SourceRuntime(
        provider,
        recorder=recorder,
        tool_definitions=[
            {"name": "web_fetch", "description": "fetch", "input_schema": {}}
        ],
        tool_handlers={"web_fetch": fetch},
        memory_enabled=False,
    )

    runtime.run_turn(
        "test",
        task_id="task-cutoff",
        cutoff="2025-05-01",
        run_metadata={"task_type": "research"},
    )
    rows, _ = read_trace_lines(recorder.trajectory_path)
    results = [row for row in rows if row["event_type"] == "tool_result"]

    assert called is False
    assert len(results) == 1
    assert results[0]["payload"]["success"] is False
    assert results[0]["payload"]["error_code"] == "cutoff_mismatch"

def _register(registry, url):
    record = evidence_record_from_result(
        "web_fetch",
        json.dumps(
            {
                "ok": True,
                "operation": "fetch",
                "url": url,
                "title": "Source",
                "content": f"evidence from {url}",
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }
        ),
    )
    assert record is not None
    registry.register(record)
    return record


def test_evidence_record_ignores_search_and_failed_fetch():
    assert evidence_record_from_result("web_search", '{"ok": true}') is None
    assert evidence_record_from_result("web_fetch", '{"ok": false}') is None


def test_evidence_record_is_bounded_and_canonicalized():
    record = evidence_record_from_result(
        "web_fetch",
        json.dumps(
            {
                "ok": True,
                "operation": "fetch",
                "url": "HTTPS://Example.COM:443/report?b=2&a=1#part",
                "title": "Report",
                "content": "x" * 7000,
                "published_at": None,
                "date_status": "unknown",
                "cutoff": None,
            }
        ),
    )
    assert record is not None
    assert record.canonical_url == "https://example.com/report?a=1&b=2"
    assert record.domain == "example.com"
    assert len(record.content_excerpt) == 6000


def _direct_evidence_run(tmp_path, run_id, *, cutoff="2025-05-01"):
    recorder = TraceRecorder(tmp_path / run_id, run_id=run_id)
    recorder.start_run(
        task_id=f"{run_id}-task",
        question="research question",
        cutoff=cutoff,
        metadata={},
    )
    return recorder, RunContext(
        recorder,
        run_id,
        f"{run_id}-task",
        cutoff,
    )


def test_pdf_pages_are_flattened_and_repeated_ranges_merge_once(tmp_path):
    url = "https://example.com/report.pdf"
    provider = ScriptedProvider([
        ModelResponse(
            "",
            [ToolCall(
                "pdf-1",
                "pdf_fetch",
                {"url": url, "start_page": 11, "page_count": 1},
            )],
            "tool_calls",
        ),
        ModelResponse(
            "",
            [ToolCall(
                "pdf-2",
                "pdf_fetch",
                {"url": url, "start_page": 13, "page_count": 1},
            )],
            "tool_calls",
        ),
        ModelResponse("research notes", [], "stop"),
    ])

    def pdf_fetch(**arguments):
        page_number = arguments["start_page"]
        return json.dumps({
            "ok": True,
            "operation": "pdf_fetch",
            "url": url,
            "total_pages": 20,
            "start_page": page_number,
            "end_page": page_number,
            "has_more": page_number < 20,
            "published_at": "2025-01-02",
            "date_status": "verified",
            "cutoff": arguments["cutoff"],
            "pages": [{
                "page_number": page_number,
                "content": (
                    f"--- PAGE {page_number} START ---\n"
                    f"facts from page {page_number}\n"
                    f"--- PAGE {page_number} END ---"
                ),
            }],
        })

    recorder, run = _direct_evidence_run(tmp_path, "pdf-merge")
    registry = EvidenceRegistry()
    outcome = agent.agent_loop(
        [{"role": "user", "content": "read two PDF ranges"}],
        {},
        provider=provider,
        tools=[{"name": "pdf_fetch", "description": "pdf", "input_schema": {}}],
        handlers={"pdf_fetch": pdf_fetch},
        memory_enabled=False,
        run_context=run,
        evidence_registry=registry,
        research_cutoff="2025-05-01",
        finalize_user_turn=False,
    )

    assert outcome.final_text == "research notes"
    assert len(registry.records) == 1
    record = registry.records[0]
    assert record.content_excerpt == (
        "--- PAGE 11 START ---\nfacts from page 11\n--- PAGE 11 END ---\n\n"
        "--- PAGE 13 START ---\nfacts from page 13\n--- PAGE 13 END ---"
    )
    assert len(record.content_excerpt) <= 6000
    assert len(record.artifact_references) == 2
    assert len({item.sha256 for item in record.artifact_references}) == 2

    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert incomplete is False
    assert len([row for row in rows if row["event_type"] == "source_registered"]) == 2


def test_repeated_pdf_fetches_bound_merged_fragments_and_artifacts(tmp_path):
    url = "https://example.com/report.pdf"
    provider = ScriptedProvider([
        *[
            ModelResponse(
                "",
                [ToolCall(
                    f"pdf-{page_number}",
                    "pdf_fetch",
                    {"url": url, "start_page": page_number, "page_count": 1},
                )],
                "tool_calls",
            )
            for page_number in range(1, 11)
        ],
        ModelResponse("research notes", [], "stop"),
    ])

    def pdf_fetch(**arguments):
        page_number = arguments["start_page"]
        capture_tool_artifact(
            f"raw PDF {page_number}".encode(),
            media_type="application/pdf",
            source=url,
            suffix=".pdf",
        )
        capture_tool_artifact(
            [{"page_number": page_number, "content": f"page {page_number}"}],
            media_type="application/json",
            source=f"{url}#page={page_number}",
            suffix=".json",
        )
        return json.dumps({
            "ok": True,
            "operation": "pdf_fetch",
            "url": url,
            "start_page": page_number,
            "end_page": page_number,
            "published_at": "2025-01-02",
            "date_status": "verified",
            "cutoff": arguments["cutoff"],
            "pages": [{
                "page_number": page_number,
                "content": f"page {page_number}",
            }],
        })

    _, run = _direct_evidence_run(tmp_path, "pdf-bounds")
    registry = EvidenceRegistry()
    outcome = agent.agent_loop(
        [{"role": "user", "content": "read many PDF ranges"}],
        {},
        provider=provider,
        tools=[{"name": "pdf_fetch", "description": "pdf", "input_schema": {}}],
        handlers={"pdf_fetch": pdf_fetch},
        memory_enabled=False,
        run_context=run,
        evidence_registry=registry,
        research_cutoff="2025-05-01",
        finalize_user_turn=False,
    )

    assert outcome.final_text == "research notes"
    assert len(registry.records) == 1
    record = registry.records[0]
    assert len(record.content_excerpt) <= 6000
    assert len(record.content_fragments) == 8
    assert "--- PAGE 8 START ---" in record.content_excerpt
    assert "--- PAGE 9 START ---" not in record.content_excerpt
    assert len(record.artifact_references) == 16


def _run_single_fetch_result(tmp_path, run_id, tool_name, output):
    provider = ScriptedProvider([
        ModelResponse(
            "",
            [ToolCall("fetch-1", tool_name, {"url": "https://example.com"})],
            "tool_calls",
        ),
        ModelResponse("research notes", [], "stop"),
    ])
    recorder, run = _direct_evidence_run(tmp_path, run_id)
    registry = EvidenceRegistry()
    outcome = agent.agent_loop(
        [{"role": "user", "content": "fetch"}],
        {},
        provider=provider,
        tools=[{"name": tool_name, "description": "fetch", "input_schema": {}}],
        handlers={tool_name: lambda **_: output},
        memory_enabled=False,
        run_context=run,
        evidence_registry=registry,
        research_cutoff="2025-05-01",
        finalize_user_turn=False,
    )
    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    return recorder, outcome, registry, rows, incomplete


def test_pdf_range_keeps_usable_pages_and_accepts_blank_neighbors(tmp_path):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/report.pdf",
        "start_page": 1,
        "end_page": 3,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [
            {"page_number": 1, "content": "  \n"},
            {"page_number": 2, "content": "facts from page two"},
            {
                "page_number": 3,
                "content": "--- PAGE 3 START ---\n \n--- PAGE 3 END ---",
            },
        ],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-partial-blank",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert incomplete is False
    assert len(registry.records) == 1
    record = registry.records[0]
    assert [item.key for item in record.content_fragments] == [
        "pdf_page:0000000002"
    ]
    assert record.content_excerpt == (
        "--- PAGE 2 START ---\nfacts from page two\n--- PAGE 2 END ---"
    )
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    registered = next(
        row for row in rows if row["event_type"] == "source_registered"
    )
    assert terminal["sequence"] < registered["sequence"]


def test_pdf_fragment_preserves_meaningful_normalized_unicode(tmp_path):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/unicode.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{
            "page_number": 1,
            "content": (
                "--- PAGE 1 START ---\r\n"
                "研究结论 📈\r\ncafe\u0301\r\n"
                "--- PAGE 1 END ---"
            ),
        }],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-meaningful-unicode",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert incomplete is False
    assert len(registry.records) == 1
    expected = "--- PAGE 1 START ---\n研究结论 📈\ncafé\n--- PAGE 1 END ---"
    assert registry.records[0].content_excerpt == expected
    assert registry.records[0].content_fragments[0].content == expected
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    registered = next(
        row for row in rows if row["event_type"] == "source_registered"
    )
    assert terminal["sequence"] < registered["sequence"]


@pytest.mark.parametrize(
    "content",
    (
        "--- PAGE 1 START ---\nfacts",
        "facts\n--- PAGE 1 END ---",
        "--- PAGE 2 START ---\nfacts\n--- PAGE 2 END ---",
        (
            "--- PAGE 1 START ---\nfacts\n"
            "--- PAGE 2 START ---\n--- PAGE 1 END ---"
        ),
    ),
)
def test_pdf_rejects_lone_mismatched_or_extra_page_markers(tmp_path, content):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/malformed-markers.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{"page_number": 1, "content": content}],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-malformed-page-markers",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


@pytest.mark.parametrize(
    "body",
    (
        "--- TABLE 1 START ---\n\t \n--- TABLE 1 END ---",
        "--- !!! ... ---",
    ),
)
def test_pdf_rejects_empty_table_or_punctuation_scaffolding(tmp_path, body):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/empty-scaffolding.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{
            "page_number": 1,
            "content": (
                f"--- PAGE 1 START ---\n{body}\n--- PAGE 1 END ---"
            ),
        }],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-empty-scaffolding",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


@pytest.mark.parametrize(
    "body",
    (
        "--- PAGE 1 BEGIN ---",
        "--- TABLE 1 BEGIN ---",
        "--- TABLE 1 START ---\nfacts",
        "facts\n--- TABLE 1 END ---",
        "--- TABLE 1 START ---\nfacts\n--- TABLE 2 END ---",
        (
            "--- TABLE 1 START ---\n--- TABLE 2 START ---\nfacts\n"
            "--- TABLE 2 END ---\n--- TABLE 1 END ---"
        ),
        (
            "--- TABLE 1 START ---\nfacts\n--- TABLE 1 END ---\n"
            "--- TABLE 1 START ---\nmore facts\n--- TABLE 1 END ---"
        ),
        "--- TABLE 1 STA",
    ),
)
def test_pdf_rejects_marker_like_or_malformed_table_structure(tmp_path, body):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/malformed-structure.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{
            "page_number": 1,
            "content": (
                f"--- PAGE 1 START ---\n{body}\n--- PAGE 1 END ---"
            ),
        }],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-malformed-structure",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


@pytest.mark.parametrize(
    "content",
    (
        " --- PAGE 1 START ---\nfacts\n--- PAGE 1 END ---",
        "--- PAGE 1 START ---\nfacts\n--- PAGE 1 END --- ",
        "\n--- PAGE 1 START ---\nfacts\n--- PAGE 1 END ---",
        "--- PAGE 1 START ---\nfacts\n--- PAGE 1 END ---\n",
    ),
)
def test_pdf_rejects_padded_or_outer_blank_page_wrappers(tmp_path, content):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/padded-wrapper.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{"page_number": 1, "content": content}],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-padded-wrapper",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


@pytest.mark.parametrize(
    "reserved_line",
    (
        "--- PAGE1 START ---",
        "--- PAG",
        "--- PA",
        "--- TABLE1 START ---",
        "--- TABL",
        "--- TAB",
    ),
)
def test_pdf_rejects_compact_or_truncated_reserved_marker_prefixes(
    tmp_path, reserved_line
):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/reserved-prefix.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{
            "page_number": 1,
            "content": (
                f"--- PAGE 1 START ---\n{reserved_line}\n--- PAGE 1 END ---"
            ),
        }],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-reserved-prefix",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


@pytest.mark.parametrize(
    ("content", "expected_body"),
    (
        (
            "--- PAGE 1 START ---\nexact facts\n--- PAGE 1 END ---",
            "exact facts",
        ),
        (
            "--- DATA 1 BEGIN ---\nOrdinary page/table prose: π ≥ 3.14",
            "--- DATA 1 BEGIN ---\nOrdinary page/table prose: π ≥ 3.14",
        ),
    ),
)
def test_pdf_accepts_exact_wrapper_and_nonreserved_delimiter_controls(
    tmp_path, content, expected_body
):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/marker-controls.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{"page_number": 1, "content": content}],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-marker-controls",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert incomplete is False
    assert len(registry.records) == 1
    fragment = registry.records[0].content_fragments[0].content
    assert fragment == (
        f"--- PAGE 1 START ---\n{expected_body}\n--- PAGE 1 END ---"
    )
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    registered = next(
        row for row in rows if row["event_type"] == "source_registered"
    )
    assert terminal["sequence"] < registered["sequence"]


def test_pdf_strips_valid_table_delimiters_before_fragment_bounding(tmp_path):
    page_start = "--- PAGE 1 START ---"
    page_end = "--- PAGE 1 END ---"
    available_body_chars = (
        EVIDENCE_PDF_FRAGMENT_CHARS_MAX
        - len(page_start)
        - len(page_end)
        - 2
    )
    table_row = "Revenue\t120"
    prefix = "." * (available_body_chars - len(table_row) - 1)
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/boundary-table.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{
            "page_number": 1,
            "content": (
                f"{page_start}\n{prefix}\n"
                f"--- TABLE 1 START ---\n{table_row}\n"
                f"--- TABLE 1 END ---\n{page_end}"
            ),
        }],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-boundary-table",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert incomplete is False
    assert len(registry.records) == 1
    fragment = registry.records[0].content_fragments[0].content
    assert table_row in fragment
    assert "--- TABLE" not in fragment
    assert len(fragment) == EVIDENCE_PDF_FRAGMENT_CHARS_MAX
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    registered = next(
        row for row in rows if row["event_type"] == "source_registered"
    )
    assert terminal["sequence"] < registered["sequence"]


def test_pdf_preserves_unmarked_page_table_prose_math_and_unicode(tmp_path):
    content = "This page compares table values: π ≥ 3.14；研究完成。"
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/ordinary-prose.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{"page_number": 1, "content": content}],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-ordinary-prose",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert incomplete is False
    assert len(registry.records) == 1
    assert content in registry.records[0].content_fragments[0].content
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    registered = next(
        row for row in rows if row["event_type"] == "source_registered"
    )
    assert terminal["sequence"] < registered["sequence"]


def test_pdf_fragment_merge_orders_cross_width_page_keys_numerically():
    page_numbers = (
        10_000_000_004,
        9_999_999_998,
        10_000_000_001,
        9_999_999_996,
        10_000_000_003,
        9_999_999_999,
        10_000_000_000,
        9_999_999_995,
        10_000_000_002,
        9_999_999_997,
    )
    registry = EvidenceRegistry()
    for page_number in page_numbers:
        record = evidence_record_from_result(
            "pdf_fetch",
            json.dumps({
                "ok": True,
                "operation": "pdf_fetch",
                "url": "https://example.com/cross-width.pdf",
                "start_page": page_number,
                "end_page": page_number,
                "cutoff": "2025-05-01",
                "published_at": None,
                "date_status": "unknown",
                "pages": [{
                    "page_number": page_number,
                    "content": f"facts for page {page_number}",
                }],
            }),
            required_cutoff="2025-05-01",
        )
        assert record is not None
        registry.register(record)

    retained_keys = [
        fragment.key for fragment in registry.records[0].content_fragments
    ]
    retained_pages = [int(key.removeprefix("pdf_page:")) for key in retained_keys]
    assert retained_pages == sorted(page_numbers)[:8]
    assert any(len(key.removeprefix("pdf_page:")) == 10 for key in retained_keys)
    assert any(len(key.removeprefix("pdf_page:")) == 11 for key in retained_keys)


@pytest.mark.parametrize(
    ("start_page", "end_page", "pages"),
    (
        (
            1,
            3,
            [
                {"page_number": 1, "content": "page one"},
                {"page_number": 3, "content": "page three"},
            ],
        ),
        (
            1,
            2,
            [
                {"page_number": 2, "content": "page two"},
                {"page_number": 1, "content": "page one"},
            ],
        ),
    ),
)
def test_pdf_success_rejects_noncontiguous_or_out_of_order_pages(
    tmp_path, start_page, end_page, pages
):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/report.pdf",
        "start_page": start_page,
        "end_page": end_page,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": pages,
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-invalid-sequence",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


def test_web_content_is_trimmed_before_the_stored_excerpt_is_bounded(tmp_path):
    output = json.dumps({
        "ok": True,
        "operation": "fetch",
        "url": "https://example.com/report",
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "content": " " * 6000 + "facts",
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "web-trim-before-bound",
        "web_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert incomplete is False
    assert len(registry.records) == 1
    assert registry.records[0].content_excerpt == "facts"
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    registered = next(
        row for row in rows if row["event_type"] == "source_registered"
    )
    assert terminal["sequence"] < registered["sequence"]


def test_web_content_normalizes_nonprinting_prefix_before_bounding(tmp_path):
    output = json.dumps({
        "ok": True,
        "operation": "fetch",
        "url": "https://example.com/unicode-report",
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "content": "\x00" * 6000 + "研究事实\r\ncafe\u0301 📈",
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "web-normalize-before-bound",
        "web_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert incomplete is False
    assert len(registry.records) == 1
    assert registry.records[0].content_excerpt == "研究事实\ncafé 📈"
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    registered = next(
        row for row in rows if row["event_type"] == "source_registered"
    )
    assert terminal["sequence"] < registered["sequence"]


def test_web_content_is_nfc_after_format_characters_are_filtered(tmp_path):
    output = json.dumps({
        "ok": True,
        "operation": "fetch",
        "url": "https://example.com/post-filter-nfc",
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "content": "cafe\u200b\u0301 findings",
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "web-post-filter-nfc",
        "web_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert incomplete is False
    assert len(registry.records) == 1
    assert registry.records[0].content_excerpt == "café findings"
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    registered = next(
        row for row in rows if row["event_type"] == "source_registered"
    )
    assert terminal["sequence"] < registered["sequence"]


@pytest.mark.parametrize("content", ("\x00\x01", "\u200b\u2060"))
def test_control_or_format_only_web_content_is_rejected(tmp_path, content):
    output = json.dumps({
        "ok": True,
        "operation": "fetch",
        "url": "https://example.com/nonprinting-report",
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "content": content,
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "web-nonprinting-only",
        "web_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_content"


def test_control_or_format_only_pdf_page_is_rejected(tmp_path):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/nonprinting.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{
            "page_number": 1,
            "content": (
                "--- PAGE 1 START ---\n\x00\u200b\u2060\n"
                "--- PAGE 1 END ---"
            ),
        }],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-nonprinting-only",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


def test_pdf_page_number_cannot_overflow_the_fragment_cap(tmp_path):
    page_number = int("9" * 400)
    assert len(
        f"--- PAGE {page_number} START ---\nfacts\n"
        f"--- PAGE {page_number} END ---"
    ) > EVIDENCE_PDF_FRAGMENT_CHARS_MAX
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/oversized-marker.pdf",
        "start_page": page_number,
        "end_page": page_number,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{"page_number": page_number, "content": "facts"}],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-oversized-marker",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


def test_pdf_success_cannot_exceed_the_handler_page_count_limit(tmp_path):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/too-many-pages.pdf",
        "start_page": 1,
        "end_page": 21,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [
            {"page_number": page_number, "content": f"facts {page_number}"}
            for page_number in range(1, 22)
        ],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "pdf-handler-page-limit",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_pdf_pages"


@pytest.mark.parametrize("tool_name", ("web_fetch", "pdf_fetch"))
def test_fetch_parse_failure_is_auditable_source_rejection(tmp_path, tool_name):
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        f"invalid-json-{tool_name}",
        tool_name,
        "{not valid JSON",
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    rejected = next(row for row in rows if row["event_type"] == "source_rejected")
    assert terminal["payload"]["success"] is True
    assert terminal["sequence"] < rejected["sequence"]
    assert rejected["payload"]["reason_code"] == "invalid_json"
    assert not any(
        row["event_type"] == "research_result_unparseable" for row in rows
    )


def test_search_parse_failure_keeps_legacy_unparseable_trace(tmp_path):
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "invalid-json-web-search",
        "web_search",
        "{not valid JSON",
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = next(row for row in rows if row["event_type"] == "tool_result")
    unparseable = next(
        row
        for row in rows
        if row["event_type"] == "research_result_unparseable"
    )
    assert terminal["payload"]["success"] is True
    assert terminal["sequence"] < unparseable["sequence"]
    assert not any(row["event_type"] == "source_rejected" for row in rows)


@pytest.mark.parametrize(
    "malformed_url",
    (
        {"unexpected": "object"},
        "https://user:password@example.com/report",
        "https://exa mple.com/report",
        "https://exa\tmple.com/report",
        "https://-bad.example/report",
        "https://example..com/report",
        "https://example.com../report",
        "https://example.com/\u00a0report",
        "https://example.com/\u0085report",
        "https://[fe80::1%25eth0]/report",
        "https://999.999.999.999/report",
        "https://example.com:0/report",
        "https://example.com:99999/report",
    ),
)
def test_invalid_fetch_url_is_rejected_without_losing_terminal_result(
    tmp_path, malformed_url
):
    malformed_output = json.dumps({
        "ok": True,
        "operation": "fetch",
        "url": malformed_url,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "content": "must not become registered evidence",
    })
    recorder, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "invalid-url",
        "web_fetch",
        malformed_output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    terminal = [row for row in rows if row["event_type"] == "tool_result"]
    rejected = [row for row in rows if row["event_type"] == "source_rejected"]
    assert incomplete is False
    assert len(terminal) == 1
    assert terminal[0]["payload"]["success"] is True
    artifact_path = recorder.run_dir / terminal[0]["payload"]["output_artifact"]["path"]
    assert artifact_path.read_text(encoding="utf-8") == malformed_output
    assert len(rejected) == 1
    assert rejected[0]["payload"]["reason_code"] == "invalid_url"
    assert terminal[0]["sequence"] < rejected[0]["sequence"]


@pytest.mark.parametrize(
    ("payload", "reason_code"),
    (
        (
            {
                "ok": True,
                "url": "https://example.com/report",
                "cutoff": "2025-05-01",
                "published_at": None,
                "date_status": "unknown",
                "content": "facts",
            },
            "invalid_operation",
        ),
        (
            {
                "ok": True,
                "operation": "web_fetch",
                "url": "https://example.com/report",
                "cutoff": "2025-05-01",
                "published_at": None,
                "date_status": "unknown",
                "content": "facts",
            },
            "invalid_operation",
        ),
        (
            {
                "ok": True,
                "operation": "fetch",
                "url": "https://example.com/report",
                "published_at": None,
                "date_status": "unknown",
                "content": "facts",
            },
            "invalid_cutoff",
        ),
        (
            {
                "ok": True,
                "operation": "fetch",
                "url": "https://example.com/report",
                "cutoff": "2025-05-01",
                "date_status": "unknown",
                "content": "facts",
            },
            "invalid_published_at",
        ),
        (
            {
                "ok": True,
                "operation": "fetch",
                "url": "https://example.com/report",
                "cutoff": "2025-05-01",
                "published_at": None,
                "date_status": "unknown",
            },
            "invalid_content",
        ),
        (
            {
                "ok": True,
                "operation": "fetch",
                "url": "https://example.com/report",
                "cutoff": "2025-05-01",
                "published_at": None,
                "date_status": "unknown",
                "content": "   \n\t",
            },
            "invalid_content",
        ),
        (
            {
                "ok": True,
                "operation": "fetch",
                "url": "https://example.com/report",
                "cutoff": "2025-05-01",
                "published_at": None,
                "content": "facts",
            },
            "invalid_date_status",
        ),
        (
            {
                "ok": True,
                "operation": "fetch",
                "url": "https://example.com/report",
                "cutoff": "2025-05-01",
                "published_at": None,
                "date_status": "nonsense",
                "content": "facts",
            },
            "invalid_date_status",
        ),
    ),
)
def test_malformed_web_success_schema_does_not_count_as_evidence(
    tmp_path, payload, reason_code
):
    output = json.dumps(payload)
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        f"invalid-web-{reason_code}",
        "web_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    terminal = [row for row in rows if row["event_type"] == "tool_result"]
    rejected = [row for row in rows if row["event_type"] == "source_rejected"]
    assert len(terminal) == 1
    assert terminal[0]["payload"]["success"] is True
    assert len(rejected) == 1
    assert rejected[0]["payload"]["reason_code"] == reason_code


def test_pdf_success_with_empty_page_text_does_not_count_as_evidence(tmp_path):
    output = json.dumps({
        "ok": True,
        "operation": "pdf_fetch",
        "url": "https://example.com/report.pdf",
        "start_page": 1,
        "end_page": 1,
        "cutoff": "2025-05-01",
        "published_at": None,
        "date_status": "unknown",
        "pages": [{"page_number": 1, "content": " \n\t "}],
    })
    _, outcome, registry, rows, incomplete = _run_single_fetch_result(
        tmp_path,
        "invalid-pdf-empty",
        "pdf_fetch",
        output,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    assert incomplete is False
    assert len([row for row in rows if row["event_type"] == "tool_result"]) == 1
    rejected = [row for row in rows if row["event_type"] == "source_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["payload"]["reason_code"] == "invalid_pdf_pages"


def test_repeated_source_with_conflicting_date_metadata_is_rejected(tmp_path):
    url = "https://example.com/report"
    provider = ScriptedProvider([
        ModelResponse(
            "",
            [ToolCall("fetch-1", "web_fetch", {"url": url})],
            "tool_calls",
        ),
        ModelResponse(
            "",
            [ToolCall("fetch-2", "web_fetch", {"url": url})],
            "tool_calls",
        ),
        ModelResponse("research notes", [], "stop"),
    ])
    calls = 0

    def fetch(**arguments):
        nonlocal calls
        calls += 1
        return json.dumps({
            "ok": True,
            "operation": "fetch",
            "url": url,
            "cutoff": arguments["cutoff"],
            "published_at": None if calls == 1 else "2025-01-02",
            "date_status": "unknown" if calls == 1 else "verified",
            "content": f"fetch {calls}",
        })

    recorder, run = _direct_evidence_run(tmp_path, "conflicting-source-date")
    registry = EvidenceRegistry()
    outcome = agent.agent_loop(
        [{"role": "user", "content": "fetch twice"}],
        {},
        provider=provider,
        tools=[{"name": "web_fetch", "description": "fetch", "input_schema": {}}],
        handlers={"web_fetch": fetch},
        memory_enabled=False,
        run_context=run,
        evidence_registry=registry,
        research_cutoff="2025-05-01",
        finalize_user_turn=False,
    )

    assert outcome.final_text == "research notes"
    assert len(registry.records) == 1
    assert registry.records[0].published_at is None
    assert registry.records[0].date_status == "unknown"
    rows, _ = read_trace_lines(recorder.trajectory_path)
    assert len([row for row in rows if row["event_type"] == "source_registered"]) == 1
    rejected = [row for row in rows if row["event_type"] == "source_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["payload"]["reason_code"] == "conflicting_date_status"


@pytest.mark.parametrize(
    ("metadata", "reason_code"),
    (
        (
            {
                "cutoff": "2025-05-01",
                "published_at": "2025-06-01",
                "date_status": "verified",
            },
            "published_after_cutoff",
        ),
        (
            {
                "cutoff": "2025-05-02",
                "published_at": "2025-01-02",
                "date_status": "verified",
            },
            "cutoff_mismatch",
        ),
        (
            {
                "cutoff": "2025-05-01",
                "published_at": {"year": 2025},
                "date_status": "verified",
            },
            "invalid_published_at",
        ),
        (
            {
                "cutoff": "2025-05-01",
                "published_at": None,
                "date_status": "verified",
            },
            "date_status_conflict",
        ),
        (
            {
                "cutoff": "2025-05-01",
                "published_at": "2025-01-02",
                "date_status": "verified",
                "publication_dates": ["2025-01-02", "2025-01-03"],
            },
            "date_conflict",
        ),
    ),
)
def test_policy_inconsistent_fetch_metadata_is_rejected(
    tmp_path, metadata, reason_code
):
    provider = ScriptedProvider([
        ModelResponse(
            "",
            [ToolCall("fetch-1", "web_fetch", {"url": "https://example.com"})],
            "tool_calls",
        ),
        ModelResponse("research notes", [], "stop"),
    ])
    output = json.dumps({
        "ok": True,
        "operation": "fetch",
        "url": "https://example.com/report",
        "content": "must not become registered evidence",
        **metadata,
    })
    recorder, run = _direct_evidence_run(tmp_path, f"reject-{reason_code}")
    registry = EvidenceRegistry()

    outcome = agent.agent_loop(
        [{"role": "user", "content": "fetch"}],
        {},
        provider=provider,
        tools=[{"name": "web_fetch", "description": "fetch", "input_schema": {}}],
        handlers={"web_fetch": lambda **_: output},
        memory_enabled=False,
        run_context=run,
        evidence_registry=registry,
        research_cutoff="2025-05-01",
        finalize_user_turn=False,
    )

    assert outcome.final_text == "research notes"
    assert registry.records == ()
    rows, _ = read_trace_lines(recorder.trajectory_path)
    terminal = [row for row in rows if row["event_type"] == "tool_result"]
    rejected = [row for row in rows if row["event_type"] == "source_rejected"]
    assert len(terminal) == 1
    assert terminal[0]["payload"]["success"] is True
    assert len(rejected) == 1
    assert rejected[0]["payload"]["reason_code"] == reason_code


def test_dynamic_writing_gate_uses_plan_and_authority():
    registry = EvidenceRegistry()
    first = _register(registry, "https://alpha.example/report")
    second = _register(registry, "https://beta.example/data")
    registry.mark_authority(first.source_id, True, "official filing")
    registry.mark_authority(second.source_id, False, "secondary source")
    plan = ResearchPlan(ResearchRank.LIGHT, ("core facts",), "bounded")

    assert validate_research_final(
        "See https://alpha.example/report and https://beta.example/data",
        registry,
        plan,
    ) == []


def test_dynamic_writing_gate_reports_each_failure():
    registry = EvidenceRegistry()
    _register(registry, "https://alpha.example/report")
    plan = ResearchPlan(ResearchRank.LIGHT, ("core facts",), "bounded")
    assert validate_research_final(
        "See https://unfetched.example/report",
        registry,
        plan,
    ) == [
        "read at least 2 distinct sources",
        "use at least 2 independent domains",
        "use at least 1 authoritative source",
        "cite fetched sources in the final answer",
        "final answer contains unfetched citations",
    ]
