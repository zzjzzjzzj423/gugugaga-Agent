from dataclasses import replace

import pytest

from simple_cc.evidence import evidence_record_from_result, source_id_for_url
from simple_cc.research_models import (
    EvidenceFragment,
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
from simple_cc.trace import ArtifactRef


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


def test_registry_rejects_duplicate_url_with_different_fetch_tool_unchanged():
    url = "https://example.com/report"
    web_record = evidence_record_from_result(
        "web_fetch",
        '{"ok":true,"operation":"fetch",'
        f'"url":"{url}","title":"Web report",'
        '"content":"web facts","published_at":"2025-01-02",'
        '"date_status":"verified","cutoff":"2025-05-01"}',
    )
    pdf_record = evidence_record_from_result(
        "pdf_fetch",
        '{"ok":true,"operation":"pdf_fetch",'
        f'"url":"{url}","title":"PDF report",'
        '"start_page":1,"end_page":1,"pages":['
        '{"page_number":1,"content":"pdf facts"}],'
        '"published_at":"2025-01-02","date_status":"verified",'
        '"cutoff":"2025-05-01"}',
    )
    assert web_record is not None
    assert pdf_record is not None
    registry = EvidenceRegistry()
    registry.register(web_record)
    before = registry.records

    with pytest.raises(EvidenceRegistrationError) as caught:
        registry.register(pdf_record)

    assert caught.value.code == "conflicting_tool_name"
    assert registry.records == before


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


@pytest.mark.parametrize(
    "reason",
    ("", "   ", "x" * 513, "unsafe\u0000reason"),
)
def test_mark_authority_rejects_invalid_reason_without_mutation(reason):
    record = evidence_record_from_result(
        "web_fetch",
        '{"ok":true,"operation":"fetch",'
        '"url":"https://example.com/report","title":"Report",'
        '"content":"facts","published_at":"2025-01-02",'
        '"date_status":"verified","cutoff":"2025-05-01"}',
    )
    assert record is not None
    registry = EvidenceRegistry()
    registry.register(record)

    with pytest.raises(ValueError):
        registry.mark_authority(record.source_id, True, reason)

    assert registry.records[0].authoritative is False
    assert registry.records[0].authority_reason is None


def test_mark_authority_requires_bool_known_source_and_never_double_counts():
    record = evidence_record_from_result(
        "web_fetch",
        '{"ok":true,"operation":"fetch",'
        '"url":"https://example.com/report","title":"Report",'
        '"content":"facts","published_at":"2025-01-02",'
        '"date_status":"verified","cutoff":"2025-05-01"}',
    )
    assert record is not None
    registry = EvidenceRegistry()
    registry.register(record)

    with pytest.raises(ValueError):
        registry.mark_authority(record.source_id, 1, "numeric truthy")
    with pytest.raises(ValueError):
        registry.mark_authority("src_unknown", True, "official")
    registry.mark_authority(record.source_id, True, "official filing")
    registry.mark_authority(record.source_id, True, "official filing")

    assert len(registry.records) == 1
    assert sum(item.authoritative is True for item in registry.records) == 1


@pytest.mark.parametrize(
    "content",
    ("facts\r\nmore", "e\u0301vidence"),
)
def test_registry_rejects_noncanonical_excerpt_normalization(content):
    record = evidence_record_from_result(
        "web_fetch",
        '{"ok":true,"operation":"fetch",'
        '"url":"https://example.com/report","title":"Report",'
        '"content":"facts","published_at":"2025-01-02",'
        '"date_status":"verified","cutoff":"2025-05-01"}',
    )
    assert record is not None

    with pytest.raises(EvidenceRegistrationError) as caught:
        EvidenceRegistry().register(replace(record, content_excerpt=content))

    assert caught.value.code == "invalid_content"


@pytest.mark.parametrize(
    ("key", "content"),
    (
        ("pdf_page:1", "--- PAGE 1 START ---\nfacts\n--- PAGE 1 END ---"),
        ("pdf_page:0000000001", "facts"),
        (
            "pdf_page:0000000001",
            "--- PAGE 2 START ---\nfacts\n--- PAGE 2 END ---",
        ),
    ),
)
def test_registry_rejects_noncanonical_pdf_fragment_structure(key, content):
    record = evidence_record_from_result(
        "pdf_fetch",
        '{"ok":true,"operation":"pdf_fetch",'
        '"url":"https://example.com/report.pdf","title":"Report",'
        '"start_page":1,"end_page":1,"pages":['
        '{"page_number":1,"content":"facts"}],'
        '"published_at":"2025-01-02","date_status":"verified",'
        '"cutoff":"2025-05-01"}',
    )
    assert record is not None
    fragment = EvidenceFragment(key, content)
    malformed = replace(
        record,
        content_fragments=(fragment,),
        content_excerpt=content,
    )

    with pytest.raises(EvidenceRegistrationError) as caught:
        EvidenceRegistry().register(malformed)

    assert caught.value.code == "invalid_content"


@pytest.mark.parametrize(
    "artifact",
    (
        ArtifactRef("", "0" * 64, "application/json", 1),
        ArtifactRef("artifact.json", "not-a-sha", "application/json", 1),
        ArtifactRef("artifact.json", "0" * 64, "bad\u0000type", 1),
        ArtifactRef("artifact.json", "0" * 64, "application/json", True),
        ArtifactRef("artifact.json", "0" * 64, "application/json", -1),
        ArtifactRef("artifact.json", "0" * 64, "application/json", 1_000_000_001),
    ),
)
def test_registry_rejects_malformed_artifact_references(artifact):
    record = evidence_record_from_result(
        "web_fetch",
        '{"ok":true,"operation":"fetch",'
        '"url":"https://example.com/report","title":"Report",'
        '"content":"facts","published_at":"2025-01-02",'
        '"date_status":"verified","cutoff":"2025-05-01"}',
    )
    assert record is not None

    with pytest.raises(EvidenceRegistrationError) as caught:
        EvidenceRegistry().register(replace(
            record,
            artifact_references=(artifact,),
        ))

    assert caught.value.code == "invalid_record"
