from dataclasses import replace

import pytest

from simple_cc.evidence import evidence_record_from_result, source_id_for_url
from simple_cc.research_models import (
    EvidenceRecord,
    EvidenceRegistrationError,
    EvidenceRegistry,
    RANK_POLICIES,
    ResearchBudget,
    ResearchPlan,
    ResearchRank,
    TaskKind,
    normalize_task_kind,
)


def test_task_kind_is_explicit_and_defaults_to_normal():
    assert normalize_task_kind("research") is TaskKind.RESEARCH
    assert normalize_task_kind("research_analysis") is TaskKind.RESEARCH
    assert normalize_task_kind("normal") is TaskKind.NORMAL
    assert normalize_task_kind(None) is TaskKind.NORMAL
    assert normalize_task_kind("future_kind") is TaskKind.NORMAL


@pytest.mark.parametrize(
    ("rank", "rounds", "sources", "authorities", "directions"),
    [
        (ResearchRank.LIGHT, 10, 2, 1, 1),
        (ResearchRank.STANDARD, 20, 3, 1, 2),
        (ResearchRank.DEEP, 30, 4, 2, 3),
    ],
)
def test_rank_policies_are_fixed(rank, rounds, sources, authorities, directions):
    policy = RANK_POLICIES[rank]
    assert (
        policy.max_research_rounds,
        policy.distinct_source_count,
        policy.authoritative_source_count,
        policy.research_direction_count,
    ) == (rounds, sources, authorities, directions)


def test_plan_requires_exact_unique_direction_count():
    with pytest.raises(ValueError, match="requires 2 research directions"):
        ResearchPlan(ResearchRank.STANDARD, ("one",), "reason")
    with pytest.raises(ValueError, match="unique"):
        ResearchPlan(ResearchRank.STANDARD, ("Same", " same "), "reason")


def test_budget_is_shared_and_bounded():
    budget = ResearchBudget(10)
    budget.consume(7)
    assert budget.remaining_rounds == 3
    with pytest.raises(ValueError, match="remaining research budget"):
        budget.consume(4)


def test_registry_deduplicates_canonical_urls():
    registry = EvidenceRegistry()
    first = evidence_record_from_result(
        "web_fetch",
        '{"ok":true,"operation":"fetch",'
        '"url":"https://example.com/report","title":"Report",'
        '"content":"facts","published_at":"2025-01-02",'
        '"date_status":"verified","cutoff":"2025-05-01"}',
    )
    assert first is not None
    duplicate = EvidenceRecord(**{**first.__dict__, "title": "Duplicate"})
    assert registry.register(first) is first
    assert registry.register(duplicate) is first
    assert registry.records == (first,)


@pytest.mark.parametrize(
    ("changes", "error_code"),
    (
        ({"canonical_url": "javascript:fake"}, "invalid_url"),
        ({"source_id": "src_forged"}, "source_id_mismatch"),
        ({"domain": "forged.example"}, "domain_mismatch"),
        ({"tool_name": "web_search"}, "invalid_tool_name"),
        ({"content_excerpt": ""}, "invalid_content"),
        ({"content_excerpt": "x" * 6001}, "invalid_content"),
        ({"date_status": "claimed"}, "invalid_metadata"),
        ({"authoritative": True}, "untrusted_authority"),
    ),
)
def test_registry_rejects_untrusted_record_shapes(changes, error_code):
    record = evidence_record_from_result(
        "web_fetch",
        '{"ok":true,"operation":"fetch",'
        '"url":"https://example.com/report","title":"Report",'
        '"content":"facts","published_at":"2025-01-02",'
        '"date_status":"verified","cutoff":"2025-05-01"}',
    )
    assert record is not None
    malformed = replace(record, **changes)
    registry = EvidenceRegistry()

    with pytest.raises(EvidenceRegistrationError) as caught:
        registry.register(malformed)

    assert caught.value.code == error_code
    assert registry.records == ()
