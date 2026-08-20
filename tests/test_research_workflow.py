from __future__ import annotations

import json

import pytest

from simple_cc import agent, config
from simple_cc.agent import AgentLoopOutcome
from simple_cc.evidence import (
    evidence_record_from_result,
    source_id_for_url,
    validate_research_final,
)
from simple_cc.models import ModelResponse, ToolCall
from simple_cc.research_models import EvidenceRegistry, ResearchPlan, ResearchRank
from simple_cc.research_workflow import (
    EVIDENCE_PACKET_DOMAIN_CHARS_MAX,
    EVIDENCE_PACKET_MAX_RECORDS,
    EVIDENCE_PACKET_TEXT_CHARS_MAX,
    EVIDENCE_PACKET_TITLE_CHARS_MAX,
    EVIDENCE_PACKET_TOTAL_CHARS_MAX,
    EVIDENCE_PACKET_URL_CHARS_MAX,
    ResearchWorkflow,
    build_evidence_packet,
    parse_research_gate,
    parse_research_plan,
)
from simple_cc.telemetry import TracingProvider
from simple_cc.trace import (
    RunContext,
    TraceRecorder,
    bind_run_context,
    read_trace_lines,
)
from tests.fakes import ScriptedProvider


def registry_with_two_sources() -> EvidenceRegistry:
    registry = EvidenceRegistry()
    for url in ("https://alpha.example/report", "https://beta.example/data"):
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "operation": "fetch",
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


def light_plan_response() -> ModelResponse:
    return ModelResponse(json.dumps({
        "rank": "light",
        "directions": ["primary filings"],
        "reason": "narrow factual question",
    }))


def gate_response(
    registry: EvidenceRegistry,
    *,
    covered: bool,
    gap: str = "",
) -> ModelResponse:
    source_ids = [item.source_id for item in registry.records]
    return ModelResponse(json.dumps({
        "directions": [{
            "direction": "primary filings",
            "covered": covered,
            "source_ids": source_ids if covered else [],
            "reason": "direct support" if covered else "support is missing",
        }],
        "authorities": ([{
            "source_id": source_ids[0],
            "is_authoritative": True,
            "reason": "official disclosure",
        }] if source_ids else []),
        "gaps": [gap] if gap else [],
    }))


def valid_light_report() -> ModelResponse:
    return ModelResponse(
        "Report https://alpha.example/report https://beta.example/data"
    )


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


@pytest.mark.parametrize(
    ("value", "error_fragment"),
    (
        (
            '{"rank":"light","rank":"light","directions":['
            '"primary"],"reason":"duplicate"}',
            "duplicate JSON object key: rank",
        ),
        (
            '{"rank":"light","directions":["primary"],"reason":"ok",'
            '"unexpected":true}',
            "unexpected field: unexpected",
        ),
        (
            '{"rank":"light","directions":["primary"],"reason":NaN}',
            "non-standard JSON constant: NaN",
        ),
        (
            '{"rank":"light","directions":["primary"],"reason":Infinity}',
            "non-standard JSON constant: Infinity",
        ),
    ),
)
def test_plan_rejects_duplicate_extra_and_nonstandard_json(
    value, error_fragment
):
    plan = parse_research_plan(value)

    assert plan.used_fallback is True
    assert plan.rank is ResearchRank.STANDARD
    assert error_fragment in " ".join(plan.validation_errors)


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


def test_evidence_packet_is_deterministic_diverse_and_aggregate_bounded():
    registry = EvidenceRegistry()
    for index in range(200):
        domain = "repeated.example" if index < 190 else f"domain-{index}.example"
        url = f"https://{domain}/{'u' * 3000}/{index}"
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "operation": "fetch",
                "url": url,
                "title": "T" * 2000,
                "content": f"record {index} " + "evidence " * 1000,
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }),
        )
        assert record is not None
        registered = registry.register(record)
        if index >= 198:
            registry.mark_authority(
                registered.source_id, True, "official source " + "R" * 2000
            )

    first = build_evidence_packet(registry)
    second = build_evidence_packet(registry)

    assert first == second
    assert len(first) <= EVIDENCE_PACKET_MAX_RECORDS
    assert len(json.dumps(first, ensure_ascii=False, sort_keys=True)) <= (
        EVIDENCE_PACKET_TOTAL_CHARS_MAX
    )
    assert len({item["domain"] for item in first}) >= 4
    assert {
        item.source_id for item in registry.records if item.authoritative
    } <= {item["source_id"] for item in first}
    assert all(
        len(item["url"]) <= EVIDENCE_PACKET_URL_CHARS_MAX
        and len(item["domain"]) <= EVIDENCE_PACKET_DOMAIN_CHARS_MAX
        and len(item["title"] or "") <= EVIDENCE_PACKET_TITLE_CHARS_MAX
        and len(item["content_excerpt"]) <= EVIDENCE_PACKET_TEXT_CHARS_MAX
        for item in first
    )


def test_packet_preserves_2070_character_canonical_url_and_citation_linkage():
    long_url = "https://example.com/" + "a" * 2050
    other_url = "https://other.example/report"
    assert len(long_url) == 2070
    registry = EvidenceRegistry()
    for url in (long_url, other_url):
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "operation": "fetch",
                "url": url,
                "title": "Exact URL evidence",
                "content": "direct evidence",
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }),
        )
        assert record is not None
        registry.register(record)
    registry.mark_authority(
        source_id_for_url(long_url), True, "official first-party record"
    )

    packet = build_evidence_packet(registry)

    assert EVIDENCE_PACKET_URL_CHARS_MAX >= len(long_url)
    assert long_url in {item["url"] for item in packet}
    assert validate_research_final(
        f"Report {long_url} {other_url}", registry, light_plan()
    ) == []
    assert len(json.dumps(packet, ensure_ascii=False, sort_keys=True)) <= (
        EVIDENCE_PACKET_TOTAL_CHARS_MAX
    )


def test_packet_reserves_two_same_domain_authorities_before_diverse_fill():
    registry = EvidenceRegistry()
    authority_ids = set()
    for index in range(2):
        url = f"https://official.example/filing-{index}"
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "operation": "fetch",
                "url": url,
                "title": f"Official filing {index}",
                "content": "authoritative evidence",
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }),
        )
        assert record is not None
        registered = registry.register(record)
        registry.mark_authority(registered.source_id, True, "official filing")
        authority_ids.add(registered.source_id)
    for index in range(38):
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "operation": "fetch",
                "url": f"https://domain-{index}.example/report",
                "title": "Independent evidence",
                "content": "independent evidence",
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }),
        )
        assert record is not None
        registry.register(record)

    packet = build_evidence_packet(registry)

    assert authority_ids <= {item["source_id"] for item in packet}
    assert len({item["domain"] for item in packet}) >= 4
    assert len(packet) <= EVIDENCE_PACKET_MAX_RECORDS
    assert len(json.dumps(packet, ensure_ascii=False, sort_keys=True)) <= (
        EVIDENCE_PACKET_TOTAL_CHARS_MAX
    )


def test_200_record_gate_prompt_and_packet_selection_trace_stay_bounded(tmp_path):
    registry = EvidenceRegistry()
    for index in range(200):
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "operation": "fetch",
                "url": f"https://domain-{index}.example/report",
                "title": "Evidence " + "T" * 1000,
                "content": "X" * 6000,
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }),
        )
        assert record is not None
        registry.register(record)
    provider = ScriptedProvider([ModelResponse("invalid gate")])
    recorder = TraceRecorder(tmp_path / "run", run_id="packet-bound-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(recorder, "packet-bound-run", "research-task", "2025-05-01")

    ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: None,
        run_context=run,
    ).evaluate_research("question", "2025-05-01", light_plan(), registry)

    request_text = provider.requests[0]["messages"][0]["content"]
    assert len(request_text) < EVIDENCE_PACKET_TOTAL_CHARS_MAX + 10_000
    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    selection = next(
        row["payload"]
        for row in rows
        if row["event_type"] == "evidence_packet_selected"
    )
    assert incomplete is False
    assert selection["available_record_count"] == 200
    assert selection["selected_record_count"] <= EVIDENCE_PACKET_MAX_RECORDS
    assert selection["omitted_record_count"] >= 168
    assert selection["serialized_chars"] <= EVIDENCE_PACKET_TOTAL_CHARS_MAX
    assert selection["omitted_source_ids"]
    assert selection["omitted_source_ids_truncated"] is True
    assert selection["truncated_field_count"] > 0
    assert selection["truncated_fields"]


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


@pytest.mark.parametrize(
    "case",
    (
        "repeated_direction_source_id",
        "repeated_authority_id",
        "missing_direction_reason",
        "missing_authority_reason",
        "non_boolean_covered",
        "non_boolean_authority",
        "extra_root_field",
        "extra_direction_field",
        "extra_authority_field",
    ),
)
def test_gate_rejects_malformed_schema_and_clears_authority(case):
    registry = registry_with_two_sources()
    first = registry.records[0]
    registry.mark_authority(first.source_id, True, "old decision")
    payload = valid_gate_payload(registry)

    if case == "repeated_direction_source_id":
        payload["directions"][0]["source_ids"] = [
            first.source_id,
            first.source_id,
        ]
    elif case == "repeated_authority_id":
        payload["authorities"].append(dict(payload["authorities"][0]))
    elif case == "missing_direction_reason":
        del payload["directions"][0]["reason"]
    elif case == "missing_authority_reason":
        del payload["authorities"][0]["reason"]
    elif case == "non_boolean_covered":
        payload["directions"][0]["covered"] = 1
    elif case == "non_boolean_authority":
        payload["authorities"][0]["is_authoritative"] = 1
    elif case == "extra_root_field":
        payload["unexpected"] = True
    elif case == "extra_direction_field":
        payload["directions"][0]["unexpected"] = True
    elif case == "extra_authority_field":
        payload["authorities"][0]["unexpected"] = True

    decision = parse_research_gate(json.dumps(payload), light_plan(), registry)

    assert decision.passed is False
    assert decision.gaps == ("research gate output was invalid",)
    assert decision.validation_errors
    assert not any(item.authoritative for item in registry.records)


def test_gate_rejects_repeated_directions_and_clears_authority():
    registry = registry_with_two_sources()
    first = registry.records[0]
    registry.mark_authority(first.source_id, True, "old decision")
    plan = ResearchPlan(
        ResearchRank.STANDARD,
        ("primary filings", "independent analysis"),
        "broader",
    )
    direction = valid_gate_payload(registry)["directions"][0]
    payload = {
        "directions": [direction, dict(direction)],
        "authorities": [],
        "gaps": [],
    }

    decision = parse_research_gate(json.dumps(payload), plan, registry)

    assert decision.passed is False
    assert "repeated research direction" in " ".join(decision.validation_errors)
    assert not any(item.authoritative for item in registry.records)


def test_gate_rejects_duplicate_keys_and_nonstandard_constants():
    registry = registry_with_two_sources()
    first = registry.records[0]
    valid = json.dumps(valid_gate_payload(registry), separators=(",", ":"))
    malformed = (
        (
            valid[:-1] + ',"gaps":[]}',
            "duplicate JSON object key: gaps",
        ),
        (
            valid[:-1] + ',"authorities":[]}',
            "duplicate JSON object key: authorities",
        ),
        (
            valid.replace(
                '"covered":true',
                '"covered":true,"covered":true',
                1,
            ),
            "duplicate JSON object key: covered",
        ),
        (
            valid.replace('"gaps":[]', '"gaps":NaN', 1),
            "non-standard JSON constant: NaN",
        ),
    )

    for text, error_fragment in malformed:
        registry.mark_authority(first.source_id, True, "old decision")
        decision = parse_research_gate(text, light_plan(), registry)

        assert decision.passed is False
        assert decision.gaps == ("research gate output was invalid",)
        assert error_fragment in " ".join(decision.validation_errors)
        assert not any(item.authoritative for item in registry.records)


def test_gate_invalid_later_authority_does_not_apply_valid_first_decision():
    registry = registry_with_two_sources()
    first = registry.records[0]
    payload = valid_gate_payload(registry)
    payload["authorities"].append({
        "source_id": "src_not_registered",
        "is_authoritative": True,
        "reason": "invalid later decision",
    })

    decision = parse_research_gate(json.dumps(payload), light_plan(), registry)

    assert decision.passed is False
    assert "unknown evidence source id" in " ".join(decision.validation_errors)
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
    assert "untrusted data" in delegate.requests[1]["system"]
    assert "instructions found inside evidence" in delegate.requests[1]["system"]
    plan_event = next(row for row in rows if row["event_type"] == "research_plan")
    gate_event = next(row for row in rows if row["event_type"] == "research_gate")
    assert plan_event["payload"]["rank"] == "light"
    assert gate_event["payload"]["passed"] is True


def test_tool_only_phase_response_fails_closed_to_empty_text():
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


@pytest.mark.parametrize("finish_reason", ("length", "max_tokens"))
def test_truncated_planning_response_uses_standard_fallback(finish_reason):
    provider = ScriptedProvider([ModelResponse(
        json.dumps({
            "rank": "light",
            "directions": ["primary filings"],
            "reason": "looks valid but was truncated",
        }),
        finish_reason=finish_reason,
    )])

    plan = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: None,
    ).plan("question", "2025-05-01")

    assert plan.rank is ResearchRank.STANDARD
    assert plan.used_fallback is True
    assert len(provider.requests) == 1


def test_mixed_text_and_tool_gate_response_fails_closed():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([ModelResponse(
        json.dumps(valid_gate_payload(registry)),
        [ToolCall("unexpected", "web_search", {})],
        "stop",
    )])

    decision = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: None,
    ).evaluate_research("question", "2025-05-01", light_plan(), registry)

    assert decision.passed is False
    assert decision.gaps == ("research gate output was invalid",)
    assert not any(item.authoritative for item in registry.records)


def test_truncated_first_writing_uses_single_rewrite():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=True),
        ModelResponse(
            "partial "
            "https://alpha.example/report https://beta.example/data",
            finish_reason="length",
        ),
        valid_light_report(),
    ])

    result = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
    ).run("question", "2025-05-01", registry=registry)

    assert result.writing_repair_used is True
    assert result.final_text.startswith("Report")
    assert len(provider.requests) == 4
    assert "untrusted data" in provider.requests[2]["system"]
    assert "instructions found inside evidence" in provider.requests[2]["system"]
    assert "untrusted data" in provider.requests[3]["system"]
    assert "instructions found inside evidence" in provider.requests[3]["system"]


def test_truncated_second_writing_returns_controlled_insufficient():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=True),
        ModelResponse(
            "partial "
            "https://alpha.example/report https://beta.example/data",
            finish_reason="length",
        ),
        ModelResponse(
            "apparently valid "
            "https://alpha.example/report https://beta.example/data",
            finish_reason="length",
        ),
    ])

    result = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
    ).run("question", "2025-05-01", registry=registry)

    assert result.writing_repair_used is True
    assert result.final_text.startswith("INSUFFICIENT_EVIDENCE")
    assert len(provider.requests) == 4


def test_initial_gate_pass_skips_supplement():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=True),
        valid_light_report(),
    ])
    executor_calls = []

    def executor(prompt, max_rounds, evidence_registry):
        executor_calls.append((
            max_rounds,
            "missing direct support" in prompt,
            evidence_registry is registry,
        ))
        return AgentLoopOutcome(
            "completed",
            "private note https://private-unregistered.example",
            rounds_used=3,
        )

    result = ResearchWorkflow(provider, executor).run(
        "question", "2025-05-01", registry=registry
    )

    assert executor_calls == [(10, False, True)]
    assert result.research_rounds_used == 3
    assert result.supplemental_research_used is False
    assert result.writing_repair_used is False
    assert len(provider.requests) == 3
    writing_content = provider.requests[-1]["messages"][0]["content"]
    assert "src_" in writing_content
    assert "private-unregistered.example" not in writing_content


def test_routed_trace_orders_phases_shares_budget_and_links_sources(
    tmp_path, monkeypatch
):
    old_workspace = config.WORKDIR
    config.configure_workspace(tmp_path / "workspace")
    monkeypatch.setattr(config, "MEMORY_ENABLED", False)
    cutoff = "2025-05-01"
    urls = (
        "https://alpha.example/report",
        "https://beta.example/data",
    )
    source_ids = tuple(source_id_for_url(url) for url in urls)

    def fetch(**arguments):
        return json.dumps({
            "ok": True,
            "operation": "fetch",
            "url": arguments["url"],
            "title": "Registered evidence",
            "content": f"evidence from {arguments['url']}",
            "published_at": "2025-01-02",
            "date_status": "verified",
            "cutoff": arguments["cutoff"],
        })

    failed_gate = ModelResponse(json.dumps({
        "directions": [{
            "direction": "primary filings",
            "covered": False,
            "source_ids": [],
            "reason": "independent corroboration is still missing",
        }],
        "authorities": [{
            "source_id": source_ids[0],
            "is_authoritative": True,
            "reason": "official disclosure",
        }],
        "gaps": ["independent corroboration is still missing"],
    }))
    passing_gate = ModelResponse(json.dumps({
        "directions": [{
            "direction": "primary filings",
            "covered": True,
            "source_ids": list(source_ids),
            "reason": "the registered sources now corroborate the filing",
        }],
        "authorities": [{
            "source_id": source_ids[0],
            "is_authoritative": True,
            "reason": "official disclosure",
        }],
        "gaps": [],
    }))
    provider = ScriptedProvider([
        light_plan_response(),
        ModelResponse(
            "",
            [ToolCall("fetch-1", "web_fetch", {"url": urls[0]})],
            "tool_calls",
        ),
        ModelResponse(
            "",
            [ToolCall("fetch-2", "web_fetch", {"url": urls[1]})],
            "tool_calls",
        ),
        ModelResponse("initial research notes"),
        failed_gate,
        ModelResponse("supplemental research notes"),
        passing_gate,
        ModelResponse(f"Report {urls[0]} {urls[1]}"),
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="routed-trace-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff=cutoff,
        metadata={"task_type": "research"},
    )
    runtime = agent.SourceRuntime(
        provider,
        recorder=recorder,
        tool_definitions=[{
            "name": "web_fetch",
            "description": "fetch",
            "input_schema": {},
        }],
        tool_handlers={"web_fetch": fetch},
        memory_enabled=False,
    )

    try:
        final_answer = runtime.run_turn(
            "question",
            task_id="research-task",
            cutoff=cutoff,
            run_metadata={"task_type": "research"},
        )
    finally:
        config.configure_workspace(old_workspace)

    research_requests = [request for request in provider.requests if request["tools"]]
    initial_stage = json.loads(research_requests[0]["messages"][0]["content"])
    supplemental_stage = json.loads(
        research_requests[-1]["messages"][0]["content"]
    )
    assert initial_stage["existing_evidence"] == []
    assert {
        item["source_id"] for item in supplemental_stage["existing_evidence"]
    } == set(source_ids)
    assert {
        item["url"] for item in supplemental_stage["existing_evidence"]
    } == set(urls)
    assert all(
        item["content_excerpt"].startswith("evidence from")
        for item in supplemental_stage["existing_evidence"]
    )
    assert "independent corroboration is still missing" in (
        supplemental_stage["research_gaps"]
    )
    assert "untrusted data" in research_requests[-1]["system"]

    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    workflow_event_names = {
        "task_routed",
        "research_plan",
        "research_attempt_started",
        "research_attempt_finished",
        "research_gate",
        "writing_attempt_started",
        "writing_gate",
        "research_workflow_completed",
    }
    workflow_rows = [
        row for row in rows if row["event_type"] in workflow_event_names
    ]
    assert incomplete is False
    assert [row["event_type"] for row in workflow_rows] == [
        "task_routed",
        "research_plan",
        "research_attempt_started",
        "research_attempt_finished",
        "research_gate",
        "research_attempt_started",
        "research_attempt_finished",
        "research_gate",
        "writing_attempt_started",
        "writing_gate",
        "research_workflow_completed",
    ]
    assert {row["agent_id"] for row in workflow_rows} == {"root"}

    first_finished = next(
        row for row in workflow_rows
        if row["event_type"] == "research_attempt_finished"
        and row["payload"]["attempt"] == 1
    )
    second_started = next(
        row for row in workflow_rows
        if row["event_type"] == "research_attempt_started"
        and row["payload"]["attempt"] == 2
    )
    assert second_started["payload"]["supplied_rounds"] == (
        10 - first_finished["payload"]["used_rounds"]
    )

    registered_sequences = {
        row["payload"]["source_id"]: row["sequence"]
        for row in rows
        if row["event_type"] == "source_registered"
    }
    for gate in (
        row for row in workflow_rows if row["event_type"] == "research_gate"
    ):
        assert gate["payload"]["authority_decisions"] == [{
            "source_id": source_ids[0],
            "is_authoritative": True,
            "reason": "official disclosure",
        }]
        for source_id in gate["payload"]["authoritative_source_ids"]:
            assert registered_sequences[source_id] < gate["sequence"]

    registered_payloads = [
        row["payload"] for row in rows if row["event_type"] == "source_registered"
    ]
    assert {payload["domain"] for payload in registered_payloads} == {
        "alpha.example",
        "beta.example",
    }
    assert {payload["title"] for payload in registered_payloads} == {
        "Registered evidence"
    }
    assert {payload["tool_name"] for payload in registered_payloads} == {
        "web_fetch"
    }

    final_event = next(
        row for row in rows if row["event_type"] == "final_answer"
    )
    assert final_answer == f"Report {urls[0]} {urls[1]}"
    assert set(final_event["payload"]["matched_source_ids"]) == set(source_ids)
    assert final_event["payload"]["unmatched_citations"] == []


def test_routed_trace_caps_research_and_writing_retries(tmp_path):
    failed_gate = json.dumps({
        "directions": [{
            "direction": "primary filings",
            "covered": False,
            "source_ids": [],
            "reason": "no registered evidence",
        }],
        "authorities": [],
        "gaps": ["no registered evidence"],
    })
    provider = ScriptedProvider([
        light_plan_response(),
        ModelResponse("initial research notes"),
        ModelResponse(failed_gate),
        ModelResponse("supplemental research notes"),
        ModelResponse(failed_gate),
        ModelResponse("unsupported draft"),
        ModelResponse("unsupported rewrite"),
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="retry-cap-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff=None,
        metadata={"task_type": "research"},
    )
    runtime = agent.SourceRuntime(
        provider,
        recorder=recorder,
        tool_definitions=[],
        tool_handlers={},
        memory_enabled=False,
    )

    final_answer = runtime.run_turn(
        "question",
        task_id="research-task",
        run_metadata={"task_type": "research"},
    )

    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert incomplete is False
    assert len([
        row for row in rows
        if row["event_type"] == "research_attempt_started"
    ]) == 2
    assert len([
        row for row in rows
        if row["event_type"] == "writing_repair_started"
    ]) == 1
    assert len([
        row for row in rows
        if row["event_type"] in {
            "writing_attempt_started",
            "writing_repair_started",
        }
    ]) == 2
    assert len([
        row for row in rows
        if row["event_type"] == "writing_gate"
    ]) == 2
    assert final_answer.startswith("INSUFFICIENT_EVIDENCE")


def test_failed_gate_uses_remaining_budget_once():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=False, gap="missing direct support"),
        gate_response(registry, covered=True),
        valid_light_report(),
    ])
    executor_calls = []

    def executor(prompt, max_rounds, evidence_registry):
        executor_calls.append((
            max_rounds,
            "missing direct support" in prompt,
            evidence_registry is registry,
        ))
        return AgentLoopOutcome(
            "completed",
            "notes",
            rounds_used=4 if len(executor_calls) == 1 else 2,
        )

    result = ResearchWorkflow(provider, executor).run(
        "question", "2025-05-01", registry=registry
    )

    assert executor_calls == [(10, False, True), (6, True, True)]
    assert result.research_rounds_used == 6
    assert result.supplemental_research_used is True


def test_second_gate_failure_still_enters_tool_free_writing():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=False, gap="gap one"),
        gate_response(registry, covered=False, gap="gap remains"),
        valid_light_report(),
    ])
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
    )

    result = workflow.run("question", "2025-05-01", registry=registry)

    writing_request = provider.requests[-1]
    assert "gap remains" in writing_request["messages"][0]["content"]
    assert writing_request["tools"] == []
    assert result.final_text.startswith("Report")


def test_second_writing_failure_returns_controlled_insufficient():
    registry = EvidenceRegistry()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=False, gap="no evidence"),
        gate_response(registry, covered=False, gap="still no evidence"),
        ModelResponse("unsupported draft"),
        ModelResponse("unsupported rewrite"),
    ])
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
    )

    result = workflow.run("question", "2025-05-01", registry=registry)

    assert result.writing_repair_used is True
    assert result.final_text.startswith("INSUFFICIENT_EVIDENCE")
    assert provider.requests[-1]["tools"] == []
    assert len(provider.requests) == 5
    rewrite_content = json.loads(
        provider.requests[-1]["messages"][0]["content"]
    )
    assert rewrite_content["evidence"] == []
    assert rewrite_content["validation_errors"] == [
        "read at least 2 distinct sources",
        "use at least 2 independent domains",
        "use at least 1 authoritative source",
        "cite fetched sources in the final answer",
    ]


def test_executor_cannot_report_rounds_beyond_supplied_budget():
    provider = ScriptedProvider([light_plan_response()])
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=max_rounds + 1
        ),
    )

    with pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01")

    assert getattr(caught.value, "failure_class", None) == (
        "ResearchBudgetExceeded"
    )
    assert "reported 11 rounds" in str(caught.value)


def test_failed_executor_preserves_failure_class_and_message():
    provider = ScriptedProvider([light_plan_response()])
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "failed",
            "notes",
            failure_class="ProviderUnavailable",
            failure_message="upstream timed out",
            rounds_used=2,
        ),
    )

    with pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01")

    assert getattr(caught.value, "failure_class", None) == (
        "ProviderUnavailable"
    )
    assert getattr(caught.value, "failure_message", None) == (
        "upstream timed out"
    )


def test_max_rounds_consumes_supplied_budget_and_still_writes():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=False, gap="gap remains"),
        valid_light_report(),
    ])
    executor_calls = []

    def executor(prompt, max_rounds, evidence_registry):
        executor_calls.append(max_rounds)
        return AgentLoopOutcome("max_rounds", "notes", rounds_used=1)

    result = ResearchWorkflow(provider, executor).run(
        "question", "2025-05-01", registry=registry
    )

    assert executor_calls == [10]
    assert result.research_rounds_used == 10
    assert result.supplemental_research_used is False
    assert result.final_text.startswith("Report")


def test_trace_reconstructs_forward_only_workflow_with_active_agent(tmp_path):
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=True),
        valid_light_report(),
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
        agent_id="research-agent",
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "private notes", rounds_used=3
        ),
        run_context=run,
    )

    with bind_run_context(run):
        workflow.run("question", "2025-05-01", registry=registry)

    rows = [
        json.loads(line)
        for line in recorder.trajectory_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    phase_names = {
        "research_plan",
        "research_attempt_started",
        "research_attempt_finished",
        "research_gate",
        "supplemental_research_skipped",
        "writing_attempt_started",
        "writing_gate",
        "writing_repair_started",
        "research_workflow_completed",
    }
    phase_rows = [row for row in rows if row["event_type"] in phase_names]

    assert [row["event_type"] for row in phase_rows] == [
        "research_plan",
        "research_attempt_started",
        "research_attempt_finished",
        "research_gate",
        "supplemental_research_skipped",
        "writing_attempt_started",
        "writing_gate",
        "research_workflow_completed",
    ]
    assert {row["agent_id"] for row in phase_rows} == {"research-agent"}
    started = next(
        row for row in phase_rows
        if row["event_type"] == "research_attempt_started"
    )["payload"]
    finished = next(
        row for row in phase_rows
        if row["event_type"] == "research_attempt_finished"
    )["payload"]
    assert started["attempt"] == 1
    assert started["supplied_rounds"] == 10
    assert started["directions"] == ["primary filings"]
    assert finished["used_rounds"] == 3
    assert finished["remaining_rounds"] == 7
    json.dumps([row["payload"] for row in phase_rows])


def test_research_and_writing_retry_flags_are_independent():
    registry = EvidenceRegistry()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=False, gap="no evidence"),
        gate_response(registry, covered=False, gap="still no evidence"),
        ModelResponse("unsupported draft"),
        ModelResponse("unsupported rewrite"),
    ])
    executor_calls = []

    def executor(prompt, max_rounds, evidence_registry):
        executor_calls.append(max_rounds)
        return AgentLoopOutcome("completed", "notes", rounds_used=1)

    result = ResearchWorkflow(provider, executor).run(
        "question", "2025-05-01", registry=registry
    )

    assert executor_calls == [10, 9]
    assert result.supplemental_research_used is True
    assert result.writing_repair_used is True
    assert all(request["tools"] == [] for request in provider.requests)


def test_missing_rounds_used_is_safe_budget_error_not_attribute_error():
    class MissingRoundsOutcome:
        status = "completed"
        final_text = "notes"
        failure_class = None
        failure_message = None

    workflow = ResearchWorkflow(
        ScriptedProvider([light_plan_response()]),
        lambda prompt, max_rounds, evidence_registry: MissingRoundsOutcome(),
    )

    with pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01")

    assert getattr(caught.value, "failure_class", None) == (
        "ResearchBudgetExceeded"
    )
    assert "reported None rounds" in str(caught.value)


def test_invalid_round_type_records_serializable_terminal_failure(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="invalid-round-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        "invalid-round-run",
        "research-task",
        "2025-05-01",
        agent_id="research-agent",
    )
    workflow = ResearchWorkflow(
        ScriptedProvider([light_plan_response()]),
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=["three"]
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01")

    assert getattr(caught.value, "failure_class", None) == (
        "ResearchBudgetExceeded"
    )
    rows = [
        json.loads(line)
        for line in recorder.trajectory_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    finished = next(
        row for row in rows
        if row["event_type"] == "research_attempt_finished"
    )
    terminal = next(
        row for row in rows
        if row["event_type"] == "research_workflow_completed"
    )
    assert finished["payload"]["reported_rounds"] == "['three']"
    assert terminal["payload"]["terminal_reason"] == "failed"
    assert terminal["payload"]["failure_class"] == "ResearchBudgetExceeded"
    json.dumps(finished["payload"])
    json.dumps(terminal["payload"])


def test_failed_executor_consumes_reported_rounds_and_records_terminal(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="failed-executor-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        "failed-executor-run",
        "research-task",
        "2025-05-01",
        agent_id="research-agent",
    )
    workflow = ResearchWorkflow(
        ScriptedProvider([light_plan_response()]),
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "failed",
            "notes",
            failure_class="ProviderUnavailable",
            failure_message="upstream timed out",
            rounds_used=2,
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01")

    assert getattr(caught.value, "failure_class", None) == (
        "ProviderUnavailable"
    )
    assert getattr(caught.value, "failure_message", None) == (
        "upstream timed out"
    )
    rows = [
        json.loads(line)
        for line in recorder.trajectory_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    finished = next(
        row for row in rows
        if row["event_type"] == "research_attempt_finished"
    )
    terminal = next(
        row for row in rows
        if row["event_type"] == "research_workflow_completed"
    )
    assert finished["payload"]["reported_rounds"] == 2
    assert finished["payload"]["consumed_rounds"] == 2
    assert finished["payload"]["used_rounds"] == 2
    assert terminal["payload"]["research_rounds_used"] == 2
    assert terminal["payload"]["remaining_rounds"] == 8
    assert terminal["payload"]["failure_class"] == "ProviderUnavailable"
    assert terminal["payload"]["failure_message"] == "upstream timed out"


def test_model_error_is_preserved_and_records_workflow_terminal(tmp_path):
    model_error = RuntimeError("planner offline")
    recorder = TraceRecorder(tmp_path / "run", run_id="failed-model-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        "failed-model-run",
        "research-task",
        "2025-05-01",
        agent_id="research-agent",
    )
    workflow = ResearchWorkflow(
        ScriptedProvider([model_error]),
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01")

    assert caught.value is model_error
    rows = [
        json.loads(line)
        for line in recorder.trajectory_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    terminal = next(
        row for row in rows
        if row["event_type"] == "research_workflow_completed"
    )
    assert terminal["agent_id"] == "research-agent"
    assert terminal["payload"] == {
        "failure_class": "RuntimeError",
        "failure_message": "planner offline",
        "final_validation_errors": [],
        "rank": None,
        "remaining_gaps": [],
        "remaining_rounds": None,
        "research_rounds_used": 0,
        "supplemental_research_used": False,
        "terminal_reason": "failed",
        "writing_repair_used": False,
    }
