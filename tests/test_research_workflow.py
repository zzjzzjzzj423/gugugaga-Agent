from __future__ import annotations

import json

from simple_cc.evidence import evidence_record_from_result
from simple_cc.models import ModelResponse, ToolCall
from simple_cc.research_models import EvidenceRegistry, ResearchPlan, ResearchRank
from simple_cc.research_workflow import (
    ResearchWorkflow,
    build_evidence_packet,
    parse_research_gate,
    parse_research_plan,
)
from simple_cc.telemetry import TracingProvider
from simple_cc.trace import RunContext, TraceRecorder, bind_run_context
from tests.fakes import ScriptedProvider


def registry_with_two_sources() -> EvidenceRegistry:
    registry = EvidenceRegistry()
    for url in ("https://alpha.example/report", "https://beta.example/data"):
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "url": url,
                "title": "Evidence",
                "content": "direct evidence",
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }),
        )
        assert record is not None
        registry.register(record)
    return registry


def light_plan() -> ResearchPlan:
    return ResearchPlan(ResearchRank.LIGHT, ("primary filings",), "narrow")


def valid_gate_payload(registry: EvidenceRegistry) -> dict[str, object]:
    first = registry.records[0]
    return {
        "directions": [{
            "direction": "primary filings",
            "covered": True,
            "source_ids": [first.source_id],
            "reason": "direct support",
        }],
        "authorities": [{
            "source_id": first.source_id,
            "is_authoritative": True,
            "reason": "official disclosure",
        }],
        "gaps": [],
    }


def test_parse_plan_accepts_fixed_rank_and_exact_directions():
    plan = parse_research_plan(json.dumps({
        "rank": "light",
        "directions": ["primary filings"],
        "reason": "narrow factual question",
    }))

    assert plan.rank is ResearchRank.LIGHT
    assert plan.used_fallback is False


def test_parse_plan_accepts_one_fenced_json_object():
    plan = parse_research_plan(
        "```json\n"
        '{"rank":"light","directions":["primary filings"],'
        '"reason":"narrow"}\n'
        "```"
    )

    assert plan.rank is ResearchRank.LIGHT
    assert plan.used_fallback is False


def test_invalid_plan_uses_standard_fallback():
    plan = parse_research_plan(
        '{"rank":"deep","directions":["only one"],"reason":"bad count"}'
    )

    assert plan.rank is ResearchRank.STANDARD
    assert plan.used_fallback is True
    assert plan.directions == (
        "primary facts and first-party evidence",
        "impact, risk, and independent corroboration",
    )
    assert plan.validation_errors


def test_plan_rejects_trailing_prose_arrays_and_non_list_directions():
    values = (
        '{"rank":"light","directions":["primary"],"reason":"ok"} trailing',
        '[{"rank":"light"}]',
        '{"rank":"light","directions":"primary","reason":"bad shape"}',
    )

    for value in values:
        plan = parse_research_plan(value)
        assert plan.used_fallback is True
        assert plan.rank is ResearchRank.STANDARD
        assert plan.validation_errors


def test_build_evidence_packet_exposes_only_registered_bounded_fields():
    registry = registry_with_two_sources()

    packet = build_evidence_packet(registry)

    assert packet[0] == {
        "source_id": registry.records[0].source_id,
        "url": "https://alpha.example/report",
        "domain": "alpha.example",
        "title": "Evidence",
        "content_excerpt": "direct evidence",
        "published_at": "2025-01-02",
        "date_status": "verified",
        "cutoff": "2025-05-01",
    }


def test_gate_rejects_unknown_authority_id():
    registry = registry_with_two_sources()
    payload = valid_gate_payload(registry)
    payload["authorities"] = [{
        "source_id": "src_not_registered",
        "is_authoritative": True,
        "reason": "claimed official",
    }]

    decision = parse_research_gate(json.dumps(payload), light_plan(), registry)

    assert decision.passed is False
    assert "unknown evidence source id" in " ".join(decision.validation_errors)
    assert not any(item.authoritative for item in registry.records)


def test_gate_rejects_unknown_direction_source_id():
    registry = registry_with_two_sources()
    payload = valid_gate_payload(registry)
    payload["directions"][0]["source_ids"] = ["src_not_registered"]

    decision = parse_research_gate(json.dumps(payload), light_plan(), registry)

    assert decision.passed is False
    assert "unknown evidence source id" in " ".join(decision.validation_errors)


def test_gate_requires_exact_planned_directions_and_strict_json():
    registry = registry_with_two_sources()
    payload = valid_gate_payload(registry)
    payload["directions"][0]["direction"] = "different direction"

    wrong_direction = parse_research_gate(
        json.dumps(payload), light_plan(), registry
    )
    trailing_prose = parse_research_gate(
        json.dumps(valid_gate_payload(registry)) + " trailing",
        light_plan(),
        registry,
    )

    assert wrong_direction.passed is False
    assert wrong_direction.gaps == ("research gate output was invalid",)
    assert trailing_prose.passed is False
    assert trailing_prose.validation_errors


def test_gate_invalid_output_clears_old_authority_state():
    registry = registry_with_two_sources()
    first = registry.records[0]
    registry.mark_authority(first.source_id, True, "old decision")

    decision = parse_research_gate("not json", light_plan(), registry)

    assert decision.passed is False
    assert decision.authoritative_source_ids == ()
    assert not any(item.authoritative for item in registry.records)


def test_gate_passes_with_two_domains_covered_direction_and_authority():
    registry = registry_with_two_sources()
    first = registry.records[0]

    decision = parse_research_gate(
        json.dumps(valid_gate_payload(registry)), light_plan(), registry
    )

    assert decision.passed is True
    assert decision.source_count == 2
    assert decision.domain_count == 2
    assert decision.authoritative_source_ids == (first.source_id,)
    assert registry.get_by_id(first.source_id).authoritative is True


def test_gate_recomputes_hard_targets_and_uncovered_gaps():
    registry = registry_with_two_sources()
    payload = valid_gate_payload(registry)
    payload["directions"][0]["covered"] = False
    payload["directions"][0]["reason"] = "filing is not available"
    payload["authorities"] = []

    decision = parse_research_gate(json.dumps(payload), light_plan(), registry)

    assert decision.passed is False
    assert "authoritative source target not met" in decision.gaps
    assert "direction not covered: primary filings" in decision.gaps


def test_plan_and_gate_calls_are_tool_free_and_use_stable_requests():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        ModelResponse(json.dumps({
            "rank": "light",
            "directions": ["primary filings"],
            "reason": "narrow",
        })),
        ModelResponse(json.dumps(valid_gate_payload(registry))),
    ])
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: None,
    )

    plan = workflow.plan("question", "2025-05-01")
    decision = workflow.evaluate_research(
        "question", "2025-05-01", plan, registry
    )
    registry.clear_authority()

    assert decision.passed is True
    assert [request["tools"] for request in provider.requests] == [[], []]
    assert "question" in provider.requests[0]["messages"][0]["content"]
    assert "src_" in provider.requests[1]["messages"][0]["content"]


def test_research_phase_call_kinds_are_scoped_and_tool_free(tmp_path):
    registry = registry_with_two_sources()
    delegate = ScriptedProvider([
        ModelResponse(json.dumps({
            "rank": "light",
            "directions": ["primary filings"],
            "reason": "narrow",
        })),
        ModelResponse(json.dumps(valid_gate_payload(registry))),
        ModelResponse("draft"),
        ModelResponse("rewrite"),
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="workflow-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        "workflow-run",
        "research-task",
        "2025-05-01",
    )
    workflow = ResearchWorkflow(
        TracingProvider(delegate),
        lambda prompt, max_rounds, evidence_registry: None,
        run_context=run,
    )

    with bind_run_context(run):
        plan = workflow.plan("question", "2025-05-01")
        workflow.evaluate_research(
            "question", "2025-05-01", plan, registry
        )
        assert workflow._call_text("research_writing", "system", "write") == "draft"
        assert workflow._call_text("research_rewrite", "system", "rewrite") == "rewrite"

    rows = [
        json.loads(line)
        for line in recorder.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    call_kinds = [
        row["payload"]["call_kind"]
        for row in rows
        if row["event_type"] == "llm_request_started"
    ]
    assert call_kinds == [
        "research_planning",
        "research_gate",
        "research_writing",
        "research_rewrite",
    ]
    assert [request["tools"] for request in delegate.requests] == [[], [], [], []]
    plan_event = next(row for row in rows if row["event_type"] == "research_plan")
    gate_event = next(row for row in rows if row["event_type"] == "research_gate")
    assert plan_event["payload"]["rank"] == "light"
    assert gate_event["payload"]["passed"] is True


def test_tool_only_phase_response_extracts_empty_text():
    provider = ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("tool-1", "web_search", {})])
    ])
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: None,
    )

    result = workflow._call_text("research_writing", "system", "write")

    assert result == ""
    assert provider.requests[0]["tools"] == []
