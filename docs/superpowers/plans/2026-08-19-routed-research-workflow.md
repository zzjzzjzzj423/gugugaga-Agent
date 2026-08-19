# Routed Research Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route ordinary tasks through the retained agent loop and research tasks through a fixed-rank plan/research/gate/write workflow with one supplemental-research attempt and one writing repair.

**Architecture:** `SourceRuntime` remains the public entry point and delegates explicitly marked research tasks to a new `ResearchWorkflow`. The existing `agent_loop()` remains the tool executor but no longer owns evidence finalization; structured evidence and fixed rank policies cross phase boundaries, while planning, semantic gating, writing, and rewrite use direct tool-free provider calls.

**Tech Stack:** Python 3.11+, dataclasses, enums, existing `ChatProvider`/`TracingProvider`, existing S20 content blocks, existing JSONL trace recorder, pytest 8+.

**Spec:** `docs/superpowers/specs/2026-08-19-routed-research-workflow-design.md`

## Global Constraints

- Upstream `task_type` is the only task classifier; no LLM task-classification call is permitted.
- `research` and `research_analysis` select research; missing, `normal`, and unknown values select the ordinary loop.
- Fixed ranks are exactly `light=(10 rounds, 2 distinct sources, 1 authoritative source, 1 direction)`, `standard=(20, 3, 1, 2)`, and `deep=(30, 4, 2, 3)`.
- Initial and supplemental research share one evidence registry and one total rank round budget.
- Supplemental research is admitted at most once; writing repair is admitted at most once.
- Planning, semantic-gate, writing, and rewrite calls expose no tools and do not consume research rounds.
- Search snippets never count as evidence; only successfully registered fetched sources count.
- The writing gate requires the rank source count, the same number of independent domains, the rank authority count, at least one fetched citation, and no unfetched citation.
- Preserve cutoff injection, evidence artifacts, URL canonicalization, permissions, redaction, memory isolation, and benchmark process isolation.
- Tests must not access the live network.
- Preserve unrelated dirty-worktree files; stage only files named by the current task.

---

## File Map

- Create `simple_cc/research_models.py`: task-kind normalization, fixed rank policies, plans, evidence records/registry, budgets, gate decisions, and workflow result types.
- Create `simple_cc/research_workflow.py`: structured phase calls, parsing, orchestration, writing/rewrite, and workflow trace events.
- Modify `simple_cc/evidence.py`: structured extraction and dynamic final validation.
- Modify `simple_cc/prompts.py`: ordinary and research-phase prompt builders.
- Modify `simple_cc/agent.py`: gate-free loop seam, round accounting, evidence registration, and runtime routing.
- Modify `eval/run_task.py`: pass `task_type` through the runtime boundary.
- Modify `tests/fakes.py` and the focused test modules named below.

---

### Task 1: Add Fixed Research Domain Models

**Files:**
- Create: `simple_cc/research_models.py`
- Create: `tests/test_research_models.py`

**Interfaces:**
- Produces: `TaskKind`, `normalize_task_kind()`, `ResearchRank`, `RankPolicy`, `RANK_POLICIES`, `ResearchPlan`, `ResearchBudget`, `EvidenceRecord`, `EvidenceRegistry`, `ResearchExecutionOutcome`, `DirectionAssessment`, `ResearchGateDecision`, and `ResearchWorkflowResult`.
- No dependency on the agent loop or provider.

- [ ] **Step 1: Write failing rank, routing, budget, and registry tests**

~~~python
# tests/test_research_models.py
import pytest

from simple_cc.research_models import (
    EvidenceRecord,
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
    first = EvidenceRecord(
        source_id="src_a",
        canonical_url="https://example.com/report",
        domain="example.com",
        title="Report",
        content_excerpt="facts",
        published_at="2025-01-02",
        date_status="verified",
        cutoff="2025-05-01",
        tool_name="web_fetch",
    )
    duplicate = EvidenceRecord(**{**first.__dict__, "title": "Duplicate"})
    assert registry.register(first) is first
    assert registry.register(duplicate) is first
    assert registry.records == (first,)
~~~

- [ ] **Step 2: Run the tests and verify missing module**

Run: `python -m pytest tests/test_research_models.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'simple_cc.research_models'`.

- [ ] **Step 3: Implement the model shapes**

~~~python
# simple_cc/research_models.py
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol


class TaskKind(str, Enum):
    NORMAL = "normal"
    RESEARCH = "research"


def normalize_task_kind(task_type: str | None) -> TaskKind:
    value = str(task_type or "").strip().lower()
    return (
        TaskKind.RESEARCH
        if value in {"research", "research_analysis"}
        else TaskKind.NORMAL
    )


class ResearchRank(str, Enum):
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass(frozen=True)
class RankPolicy:
    max_research_rounds: int
    distinct_source_count: int
    authoritative_source_count: int
    research_direction_count: int


RANK_POLICIES: Mapping[ResearchRank, RankPolicy] = MappingProxyType({
    ResearchRank.LIGHT: RankPolicy(10, 2, 1, 1),
    ResearchRank.STANDARD: RankPolicy(20, 3, 1, 2),
    ResearchRank.DEEP: RankPolicy(30, 4, 2, 3),
})


@dataclass(frozen=True)
class ResearchPlan:
    rank: ResearchRank
    directions: tuple[str, ...]
    reason: str
    used_fallback: bool = False
    validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cleaned = tuple(item.strip() for item in self.directions if item.strip())
        required = self.policy.research_direction_count
        if len(cleaned) != required:
            raise ValueError(f"{self.rank.value} requires {required} research directions")
        normalized = tuple(" ".join(item.casefold().split()) for item in cleaned)
        if len(set(normalized)) != len(normalized):
            raise ValueError("research directions must be unique")
        object.__setattr__(self, "directions", cleaned)

    @property
    def policy(self) -> RankPolicy:
        return RANK_POLICIES[self.rank]


@dataclass
class ResearchBudget:
    max_rounds: int
    used_rounds: int = 0

    @property
    def remaining_rounds(self) -> int:
        return self.max_rounds - self.used_rounds

    def consume(self, rounds: int) -> None:
        if rounds < 0 or rounds > self.remaining_rounds:
            raise ValueError("rounds exceed remaining research budget")
        self.used_rounds += rounds


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    canonical_url: str
    domain: str
    title: str | None
    content_excerpt: str
    published_at: str | None
    date_status: str | None
    cutoff: str | None
    tool_name: str
    authoritative: bool = False
    authority_reason: str | None = None


class EvidenceRegistry:
    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records.values())

    def register(self, record: EvidenceRecord) -> EvidenceRecord:
        return self._records.setdefault(record.canonical_url, record)

    def get_by_id(self, source_id: str) -> EvidenceRecord | None:
        return next(
            (item for item in self._records.values() if item.source_id == source_id),
            None,
        )

    def clear_authority(self) -> None:
        self._records = {
            url: replace(item, authoritative=False, authority_reason=None)
            for url, item in self._records.items()
        }

    def mark_authority(self, source_id: str, authoritative: bool, reason: str) -> None:
        record = self.get_by_id(source_id)
        if record is None:
            raise ValueError(f"unknown evidence source id: {source_id}")
        self._records[record.canonical_url] = replace(
            record,
            authoritative=authoritative,
            authority_reason=reason.strip(),
        )


class ResearchExecutionOutcome(Protocol):
    status: str
    final_text: str
    failure_class: str | None
    failure_message: str | None
    rounds_used: int


@dataclass(frozen=True)
class DirectionAssessment:
    direction: str
    covered: bool
    source_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ResearchGateDecision:
    passed: bool
    source_count: int
    domain_count: int
    authoritative_source_ids: tuple[str, ...]
    directions: tuple[DirectionAssessment, ...]
    gaps: tuple[str, ...]
    validation_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchWorkflowResult:
    final_text: str
    plan: ResearchPlan
    research_rounds_used: int
    supplemental_research_used: bool
    writing_repair_used: bool
~~~

- [ ] **Step 4: Run model tests**

Run: `python -m pytest tests/test_research_models.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add simple_cc/research_models.py tests/test_research_models.py
git commit -m "feat: add fixed research workflow models"
~~~

---

### Task 2: Build Structured Evidence and Dynamic Writing Validation

**Files:**
- Modify: `simple_cc/evidence.py:19-158`
- Modify: `tests/test_evidence_trace.py:1-292`

**Interfaces:**
- Consumes: `EvidenceRecord`, `EvidenceRegistry`, and `ResearchPlan`.
- Produces: `evidence_record_from_result()`, `registered_source_map()`, and the new `validate_research_final(final_text, registry, plan)`.
- Preserves URL canonicalization, deterministic source IDs, cutoff enforcement, citation linking, and current trace event names.

- [ ] **Step 1: Write failing extraction and dynamic-gate tests**

~~~python
# focused additions/replacements in tests/test_evidence_trace.py
from simple_cc.research_models import EvidenceRegistry, ResearchPlan, ResearchRank


def _register(registry, url):
    record = evidence_record_from_result(
        "web_fetch",
        json.dumps({
            "ok": True,
            "url": url,
            "title": "Source",
            "content": f"evidence from {url}",
            "published_at": "2025-01-02",
            "date_status": "verified",
            "cutoff": "2025-05-01",
        }),
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
        json.dumps({
            "ok": True,
            "url": "HTTPS://Example.COM:443/report?b=2&a=1#part",
            "title": "Report",
            "content": "x" * 7000,
        }),
    )
    assert record.canonical_url == "https://example.com/report?a=1&b=2"
    assert record.domain == "example.com"
    assert len(record.content_excerpt) == 6000


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
~~~

- [ ] **Step 2: Run focused tests and verify missing interfaces**

Run: `python -m pytest tests/test_evidence_trace.py -k "evidence_record or dynamic_writing_gate" -v`

Expected: import/signature failures for the new structured evidence API.

- [ ] **Step 3: Implement extraction and validation**

~~~python
# simple_cc/evidence.py
from .research_models import EvidenceRecord, EvidenceRegistry, ResearchPlan

EVIDENCE_EXCERPT_CHARS = 6000


def evidence_record_from_result(
    tool_name: str,
    output: str,
    *,
    excerpt_chars: int = EVIDENCE_EXCERPT_CHARS,
) -> EvidenceRecord | None:
    if tool_name not in {"web_fetch", "pdf_fetch"}:
        return None
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("ok") or not payload.get("url"):
        return None
    canonical = canonicalize_url(payload["url"])
    parsed = urlsplit(canonical)
    content = str(payload.get("content") or payload.get("text") or "")
    return EvidenceRecord(
        source_id=source_id_for_url(canonical),
        canonical_url=canonical,
        domain=parsed.hostname or "",
        title=str(payload["title"]) if payload.get("title") else None,
        content_excerpt=content[:excerpt_chars],
        published_at=payload.get("published_at"),
        date_status=payload.get("date_status"),
        cutoff=payload.get("cutoff"),
        tool_name=tool_name,
    )


def registered_source_map(registry: EvidenceRegistry) -> dict[str, str]:
    return {item.canonical_url: item.source_id for item in registry.records}


def validate_research_final(
    final_text: str,
    registry: EvidenceRegistry,
    plan: ResearchPlan,
) -> list[str]:
    policy = plan.policy
    records = registry.records
    linkage = link_final_answer_sources(final_text, registered_source_map(registry))
    errors = []
    if len(records) < policy.distinct_source_count:
        errors.append(f"read at least {policy.distinct_source_count} distinct sources")
    if len({item.domain for item in records if item.domain}) < policy.distinct_source_count:
        errors.append(f"use at least {policy.distinct_source_count} independent domains")
    if sum(item.authoritative for item in records) < policy.authoritative_source_count:
        errors.append(
            f"use at least {policy.authoritative_source_count} authoritative source"
        )
    if not linkage["matched_source_ids"]:
        errors.append("cite fetched sources in the final answer")
    if linkage["unmatched_citations"]:
        errors.append("final answer contains unfetched citations")
    return errors
~~~

Do not change the existing `record_research_evidence()` caller contract in this task; Task 3 changes it atomically with the agent-loop caller. This task adds the independent extractor and dynamic validator while current runtime tests still exercise the old generic gate.

- [ ] **Step 4: Run evidence and trace tests**

Run: `python -m pytest tests/test_evidence_trace.py -k "canonicalize or prepare_research or link_final or evidence_record or dynamic_writing_gate" -v && python -m pytest tests/test_trace.py -v`

Expected: all selected pure evidence tests and all trace-recorder tests pass. Full agent/evidence integration runs after Task 3 removes the old validator call and updates evidence recording atomically.

- [ ] **Step 5: Commit**

~~~bash
git add simple_cc/evidence.py tests/test_evidence_trace.py
git commit -m "feat: add structured research evidence"
~~~

---

### Task 3: Separate Prompt Profiles and Add a Stage-Aware Loop Seam

**Files:**
- Modify: `simple_cc/prompts.py:8-129`
- Modify: `simple_cc/agent.py:86-821`
- Modify: `tests/test_context_prompts.py:45-66`
- Modify: `tests/test_agent_loop_source.py`

**Interfaces:**
- Produces: `ordinary_system_prompt(context)` and `research_execution_prompt(...)`.
- Extends: `AgentLoopOutcome.rounds_used` and `agent_loop(..., system_prompt=None, evidence_registry=None, finalize_user_turn=True)`.
- Removes: `research_final_repair_used` and all calls to `validate_research_final()` from generic `agent_loop()`.

- [ ] **Step 1: Write failing prompt and loop-seam tests**

~~~python
# tests/test_context_prompts.py
def test_ordinary_prompt_is_not_financial_research_only():
    prompt = ordinary_system_prompt({"workspace": "C:/repo", "tools": "read_file"})
    assert "workspace agent" in prompt
    assert "financial research agent" not in prompt
    assert "read at least two sources" not in prompt


def test_research_execution_prompt_contains_plan_and_gaps():
    plan = ResearchPlan(
        ResearchRank.STANDARD,
        ("first-party facts", "independent impact analysis"),
        "requires corroboration",
    )
    prompt = research_execution_prompt(
        {"workspace": "C:/repo", "tools": "web_search, web_fetch"},
        question="How did the event affect the company?",
        cutoff="2025-05-01",
        plan=plan,
        gaps=("second direction lacks evidence",),
        remaining_rounds=8,
    )
    assert "standard" in prompt
    assert "second direction lacks evidence" in prompt
    assert "2025-05-01" in prompt
    assert "8" in prompt
~~~

~~~python
# tests/test_agent_loop_source.py
def test_agent_loop_uses_injected_prompt_and_reports_rounds():
    provider = ScriptedProvider([ModelResponse("stage notes")])
    outcome = agent.agent_loop(
        [{"role": "user", "content": "research"}],
        {},
        provider=provider,
        tools=[],
        handlers={},
        memory_enabled=False,
        system_prompt="STAGE PROMPT",
        finalize_user_turn=False,
    )
    assert outcome.final_text == "stage notes"
    assert outcome.rounds_used == 1
    assert provider.requests[0]["system"] == "STAGE PROMPT"


def test_traced_ordinary_loop_no_longer_applies_research_gate(tmp_path):
    provider = ScriptedProvider([ModelResponse("ordinary answer")])
    recorder = TraceRecorder(tmp_path / "run", run_id="ordinary-run")
    recorder.start_run(task_id="ordinary", question="q", cutoff=None, metadata={})
    runtime = agent.SourceRuntime(provider, recorder=recorder, memory_enabled=False)
    assert runtime.run_turn(
        "q",
        task_id="ordinary",
        run_metadata={"task_type": "normal"},
    ) == "ordinary answer"
    assert len(provider.requests) == 1
~~~

- [ ] **Step 2: Run the new tests and verify failures**

Run: `python -m pytest tests/test_context_prompts.py tests/test_agent_loop_source.py -k "ordinary_prompt or research_execution_prompt or injected_prompt or ordinary_loop" -v`

Expected: missing builder/keyword/round-accounting failures.

- [ ] **Step 3: Implement prompt profiles**

Use ordinary identity:

~~~python
ORDINARY_IDENTITY = (
    "You are a general workspace agent. Complete the user's task using the "
    "available tools, respect permissions, and verify work before claiming success."
)
~~~

Change `assemble_system_prompt()` to accept keyword-only `identity`, `include_research`, and `stage_context`. Ordinary prompts exclude the financial research contract. Research execution includes question, cutoff, selected rank, fixed targets, exact directions, known gaps, and remaining rounds as serialized JSON. Change `PromptAssembler` and `subagent_system_prompt()` back to ordinary/general wording; the new research workflow owns research identity.

- [ ] **Step 4: Remove the global gate and add exact loop seams**

~~~python
@dataclass(frozen=True)
class AgentLoopOutcome:
    status: str
    final_text: str
    failure_class: str | None = None
    failure_message: str | None = None
    rounds_used: int = 0
~~~

Apply these behaviors:

- `call_llm()` accepts `system_prompt` and uses it instead of assembling a prompt when non-null.
- `agent_loop()` accepts `system_prompt`, `evidence_registry`, and `finalize_user_turn`.
- Delete the generic final evidence gate and its repair counter.
- Every return inside the loop sets `rounds_used=_round_index + 1`; terminal max-rounds sets `rounds_used=max_rounds`.
- Stop hooks and memory extraction run only when `finalize_user_turn=True`.
- Every successful foreground `web_fetch`/`pdf_fetch` is passed to `evidence_record_from_result()` and registered whenever a registry is supplied, even without a trace.
- The same extracted record is passed to `record_research_evidence()` when a run context exists.

- [ ] **Step 5: Run prompt, loop, evidence, and recovery tests**

Run: `python -m pytest tests/test_context_prompts.py tests/test_agent_loop_source.py tests/test_evidence_trace.py tests/test_context_recovery.py -v`

Expected: all tests pass, and the former generic-loop repair test is replaced by the ordinary traced-loop assertion.

- [ ] **Step 6: Commit**

~~~bash
git add simple_cc/prompts.py simple_cc/agent.py tests/test_context_prompts.py tests/test_agent_loop_source.py tests/test_evidence_trace.py
git commit -m "refactor: separate research stages from agent loop"
~~~

---

### Task 4: Implement Planning and Hybrid Research-Gate Protocols

**Files:**
- Create: `simple_cc/research_workflow.py`
- Create: `tests/test_research_workflow.py`
- Modify: `tests/fakes.py:5-28`

**Interfaces:**
- Produces: `parse_research_plan()`, `build_evidence_packet()`, `parse_research_gate()`, and tool-free stage-call helpers.
- Consumes the Task 1 models, `ChatProvider`, `model_call_scope()`, and `extract_text()`.

- [ ] **Step 1: Write failing planner and gate tests**

~~~python
# tests/test_research_workflow.py
import json

from simple_cc.evidence import evidence_record_from_result
from simple_cc.research_models import EvidenceRegistry, ResearchPlan, ResearchRank
from simple_cc.research_workflow import parse_research_gate, parse_research_plan


def registry_with_two_sources():
    registry = EvidenceRegistry()
    for url in ("https://alpha.example/report", "https://beta.example/data"):
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "url": url,
                "title": "Evidence",
                "content": "direct evidence",
            }),
        )
        assert record is not None
        registry.register(record)
    return registry


def test_parse_plan_accepts_fixed_rank_and_exact_directions():
    plan = parse_research_plan(json.dumps({
        "rank": "light",
        "directions": ["primary filings"],
        "reason": "narrow factual question",
    }))
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


def test_gate_rejects_unknown_authority_id():
    registry = registry_with_two_sources()
    plan = ResearchPlan(ResearchRank.LIGHT, ("primary filings",), "narrow")
    decision = parse_research_gate(json.dumps({
        "directions": [{
            "direction": "primary filings",
            "covered": True,
            "source_ids": [registry.records[0].source_id],
            "reason": "direct filing",
        }],
        "authorities": [{
            "source_id": "src_not_registered",
            "is_authoritative": True,
            "reason": "claimed official",
        }],
        "gaps": [],
    }), plan, registry)
    assert decision.passed is False
    assert "unknown evidence source id" in " ".join(decision.validation_errors)


def test_gate_passes_with_two_domains_covered_direction_and_authority():
    registry = registry_with_two_sources()
    first = registry.records[0]
    plan = ResearchPlan(ResearchRank.LIGHT, ("primary filings",), "narrow")
    decision = parse_research_gate(json.dumps({
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
    }), plan, registry)
    assert decision.passed is True
    assert decision.source_count == 2
    assert decision.domain_count == 2
    assert registry.get_by_id(first.source_id).authoritative is True
~~~

- [ ] **Step 2: Run protocol tests and verify missing module**

Run: `python -m pytest tests/test_research_workflow.py -k "parse_plan or invalid_plan or gate_" -v`

Expected: collection fails because `simple_cc.research_workflow` does not exist.

- [ ] **Step 3: Implement strict plan parsing**

~~~python
STANDARD_FALLBACK_DIRECTIONS = (
    "primary facts and first-party evidence",
    "impact, risk, and independent corroboration",
)


def parse_research_plan(text: str) -> ResearchPlan:
    try:
        value = json_object(text)
        rank = ResearchRank(value.get("rank"))
        reason = str(value.get("reason") or "").strip()
        if not reason:
            raise ValueError("plan reason must be non-empty")
        return ResearchPlan(rank, tuple(value.get("directions") or ()), reason)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return ResearchPlan(
            ResearchRank.STANDARD,
            STANDARD_FALLBACK_DIRECTIONS,
            "planner output was invalid; using the standard fallback",
            used_fallback=True,
            validation_errors=(str(error),),
        )
~~~

`json_object()` accepts a single JSON object or one fenced `json` block and rejects arrays or trailing prose. Invalid plans are not retried.

- [ ] **Step 4: Implement evidence packets and strict gate parsing**

The gate schema is exactly:

~~~json
{
  "directions": [{
    "direction": "exact planned direction",
    "covered": true,
    "source_ids": ["src_example"],
    "reason": "why registered evidence covers it"
  }],
  "authorities": [{
    "source_id": "src_example",
    "is_authoritative": true,
    "reason": "authority rationale"
  }],
  "gaps": []
}
~~~

The parser must require one assessment per planned direction, reject unknown/repeated directions and source IDs, require reasons, clear old authority decisions, and recompute source/domain/authority counts from the registry. Passage requires all directions covered and all fixed targets met. Invalid gate output returns a failed decision with `gaps=("research gate output was invalid",)` and exact `validation_errors`; it does not raise into the loop.

Implement the packet and parser with these concrete shapes:

~~~python
def build_evidence_packet(registry: EvidenceRegistry) -> list[dict[str, Any]]:
    return [{
        "source_id": item.source_id,
        "url": item.canonical_url,
        "domain": item.domain,
        "title": item.title,
        "content_excerpt": item.content_excerpt,
        "published_at": item.published_at,
        "date_status": item.date_status,
        "cutoff": item.cutoff,
    } for item in registry.records]


def parse_research_gate(
    text: str,
    plan: ResearchPlan,
    registry: EvidenceRegistry,
) -> ResearchGateDecision:
    records = registry.records
    source_count = len(records)
    domain_count = len({item.domain for item in records if item.domain})
    registry.clear_authority()
    try:
        value = json_object(text)
        raw_directions = value.get("directions")
        raw_authorities = value.get("authorities")
        if not isinstance(raw_directions, list) or len(raw_directions) != len(plan.directions):
            raise ValueError("gate must assess every planned direction exactly once")
        if not isinstance(raw_authorities, list):
            raise ValueError("gate authorities must be a list")

        known_ids = {item.source_id for item in records}
        seen_directions = set()
        assessments = []
        for raw in raw_directions:
            direction = str(raw.get("direction") or "").strip()
            if direction not in plan.directions or direction in seen_directions:
                raise ValueError(f"unknown or repeated research direction: {direction}")
            seen_directions.add(direction)
            reason = str(raw.get("reason") or "").strip()
            if not reason:
                raise ValueError(f"missing direction reason: {direction}")
            source_ids = tuple(raw.get("source_ids") or ())
            unknown = sorted(set(source_ids) - known_ids)
            if unknown:
                raise ValueError(f"unknown evidence source id: {unknown[0]}")
            assessments.append(DirectionAssessment(
                direction,
                raw.get("covered") is True,
                source_ids,
                reason,
            ))

        for raw in raw_authorities:
            source_id = str(raw.get("source_id") or "")
            reason = str(raw.get("reason") or "").strip()
            if source_id not in known_ids:
                raise ValueError(f"unknown evidence source id: {source_id}")
            if not reason:
                raise ValueError(f"missing authority reason: {source_id}")
            registry.mark_authority(
                source_id,
                raw.get("is_authoritative") is True,
                reason,
            )

        authoritative_ids = tuple(
            item.source_id for item in registry.records if item.authoritative
        )
        gaps = [str(item).strip() for item in value.get("gaps", []) if str(item).strip()]
        if source_count < plan.policy.distinct_source_count:
            gaps.append("distinct source target not met")
        if domain_count < plan.policy.distinct_source_count:
            gaps.append("independent domain target not met")
        if len(authoritative_ids) < plan.policy.authoritative_source_count:
            gaps.append("authoritative source target not met")
        uncovered = [item.direction for item in assessments if not item.covered]
        gaps.extend(f"direction not covered: {item}" for item in uncovered)
        return ResearchGateDecision(
            passed=not gaps,
            source_count=source_count,
            domain_count=domain_count,
            authoritative_source_ids=authoritative_ids,
            directions=tuple(assessments),
            gaps=tuple(dict.fromkeys(gaps)),
        )
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return ResearchGateDecision(
            False,
            source_count,
            domain_count,
            (),
            (),
            ("research gate output was invalid",),
            (str(error),),
        )
~~~

- [ ] **Step 5: Implement tool-free phase calls and stable fake capture**

`ResearchWorkflow._call_text(kind, system, user_content)` calls the injected provider inside `model_call_scope(kind)` with `tools=[]`. Use kinds `research_planning`, `research_gate`, `research_writing`, and `research_rewrite`. Extract text blocks only; a tool-only response becomes empty text and therefore fails the relevant parser or final validator.

~~~python
class ResearchWorkflow:
    def __init__(
        self,
        provider: ChatProvider,
        research_executor: Callable[
            [str, int, EvidenceRegistry], ResearchExecutionOutcome
        ],
        *,
        run_context: RunContext | None = None,
    ) -> None:
        self.provider = provider
        self.research_executor = research_executor
        self.run_context = run_context

    def _call_text(self, kind: str, system: str, user_content: str) -> str:
        with model_call_scope(kind):
            response = self.provider.create(
                messages=[{"role": "user", "content": user_content}],
                system=system,
                tools=[],
                max_tokens=config.DEFAULT_MAX_TOKENS,
            )
        return extract_text(response.content)
~~~

`plan()` sends the original question and cutoff and instructs the model to return only `{"rank":"light|standard|deep","directions":["specific direction"],"reason":"brief rationale"}` with the direction count matching the selected rank. `evaluate_research()` sends the question, cutoff, normalized plan, fixed policy, and `build_evidence_packet(registry)` and instructs the model to return only the gate schema from Step 4. Both methods immediately feed returned text to their strict parser and record the normalized result when a run context exists.

Change `tests/fakes.py` request storage to `copy.deepcopy(messages)` and `copy.deepcopy(tools)` so subsequent workflow mutation cannot alter assertions.

- [ ] **Step 6: Run protocol and telemetry tests**

Run: `python -m pytest tests/test_research_workflow.py tests/test_telemetry.py -k "plan or gate or call_kind" -v`

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

~~~bash
git add simple_cc/research_workflow.py tests/test_research_workflow.py tests/fakes.py
git commit -m "feat: add research planning and gate protocols"
~~~

---

### Task 5: Implement the Bounded Research and Writing State Machine

**Files:**
- Modify: `simple_cc/research_workflow.py`
- Modify: `tests/test_research_workflow.py`

**Interfaces:**
- Consumes an injected `research_executor(prompt, max_rounds, registry) -> AgentLoopOutcome`.
- Produces `ResearchWorkflow.run(question, cutoff, registry=None) -> ResearchWorkflowResult`.

- [ ] **Step 1: Write failing forward-transition tests**

~~~python
# tests/test_research_workflow.py additions
from simple_cc.agent import AgentLoopOutcome
from simple_cc.models import ModelResponse
from simple_cc.research_workflow import ResearchWorkflow
from tests.fakes import ScriptedProvider


def light_plan_response():
    return ModelResponse(json.dumps({
        "rank": "light",
        "directions": ["primary filings"],
        "reason": "narrow factual question",
    }))


def gate_response(registry, *, covered, gap=""):
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


def valid_light_report():
    return ModelResponse(
        "Report https://alpha.example/report https://beta.example/data"
    )


def test_initial_gate_pass_skips_supplement():
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=True),
        valid_light_report(),
    ])
    executor_calls = []

    def executor(prompt, max_rounds, evidence_registry):
        executor_calls.append((max_rounds, "missing direct support" in prompt))
        return AgentLoopOutcome("completed", "notes", rounds_used=3)

    result = ResearchWorkflow(provider, executor).run(
        "question", "2025-05-01", registry=registry
    )
    assert executor_calls == [(10, False)]
    assert result.research_rounds_used == 3
    assert result.supplemental_research_used is False


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
        executor_calls.append((max_rounds, "missing direct support" in prompt))
        return AgentLoopOutcome(
            "completed",
            "notes",
            rounds_used=4 if len(executor_calls) == 1 else 2,
        )

    result = ResearchWorkflow(provider, executor).run(
        "question", "2025-05-01", registry=registry
    )
    assert executor_calls == [(10, False), (6, True)]
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
~~~

- [ ] **Step 2: Run state-machine tests and verify failures**

Run: `python -m pytest tests/test_research_workflow.py -k "initial_gate or remaining_budget or still_enters or writing_failure" -v`

Expected: failures because `ResearchWorkflow.run()` transitions are incomplete.

- [ ] **Step 3: Implement forward-only orchestration**

~~~python
def run(self, question, cutoff, *, registry=None):
    registry = registry or EvidenceRegistry()
    plan = self.plan(question, cutoff)
    budget = ResearchBudget(plan.policy.max_research_rounds)

    first = self.execute_research(
        question, cutoff, plan, (), budget.remaining_rounds, registry
    )
    self.consume_outcome(budget, first)
    gate = self.evaluate_research(question, cutoff, plan, registry)

    supplemental_used = False
    if not gate.passed and budget.remaining_rounds > 0:
        supplemental_used = True
        second = self.execute_research(
            question, cutoff, plan, gate.gaps, budget.remaining_rounds, registry
        )
        self.consume_outcome(budget, second)
        gate = self.evaluate_research(question, cutoff, plan, registry)

    draft = self.write(question, cutoff, plan, registry, gate.gaps)
    errors = validate_research_final(draft, registry, plan)
    repair_used = False
    if errors:
        repair_used = True
        draft = self.rewrite(
            question, cutoff, plan, registry, gate.gaps, draft, errors
        )
        errors = validate_research_final(draft, registry, plan)
    if errors:
        draft = (
            "INSUFFICIENT_EVIDENCE\n\nResearch finalization failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return ResearchWorkflowResult(
        draft, plan, budget.used_rounds, supplemental_used, repair_used
    )
~~~

`consume_outcome()` rejects `rounds_used` above the supplied budget. A failed executor outcome raises a workflow error carrying its failure class/message. A max-rounds outcome consumes the supplied budget and proceeds to the gate with gathered evidence.

- [ ] **Step 4: Implement writing and rewrite prompts**

Writing receives question, cutoff, normalized plan, fixed requirements, evidence packet, authority decisions, and gate gaps. Rewrite additionally receives the rejected draft and exact deterministic errors. Both use these instructions and no tools:

~~~text
Write the final answer only from the supplied registered evidence. Cite exact fetched URLs.
Distinguish verified facts, inference, and uncertainty. Disclose every unresolved research gap.
Do not invent or cite any URL absent from the evidence packet.
~~~

~~~text
Rewrite the rejected report once using only the supplied evidence. Fix every listed validation error.
Do not search, request tools, or introduce new URLs. Return only the revised final report.
~~~

- [ ] **Step 5: Add phase trace events**

Record `research_plan`, `research_attempt_started`, `research_attempt_finished`, `research_gate`, `supplemental_research_skipped` when applicable, `writing_attempt_started`, `writing_gate`, `writing_repair_started`, and `research_workflow_completed`. Payloads include attempts, supplied/used/remaining rounds, directions, hard counts, authority IDs, gaps, validation errors, and retry flags. Use the active run’s agent ID and immutable serializable data.

- [ ] **Step 6: Run workflow tests**

Run: `python -m pytest tests/test_research_workflow.py -v`

Expected: all tests pass with exact provider call counts and empty tool lists for every non-research-execution phase.

- [ ] **Step 7: Commit**

~~~bash
git add simple_cc/research_workflow.py tests/test_research_workflow.py
git commit -m "feat: orchestrate bounded research and writing"
~~~

---

### Task 6: Route SourceRuntime and Propagate Benchmark Task Type

**Files:**
- Modify: `simple_cc/agent.py:93-229`
- Modify: `eval/run_task.py:131-206`
- Modify: `tests/test_agent_loop_source.py`
- Modify: `tests/test_benchmark_worker.py:58-210`

**Interfaces:**
- Consumes `normalize_task_kind()`, `ResearchWorkflow`, retained loop configuration, and current `RunContext`.
- Preserves `SourceRuntime.run_turn() -> str` and `last_outcome`.

- [ ] **Step 1: Write failing routing tests**

~~~python
from simple_cc.research_models import (
    ResearchPlan,
    ResearchRank,
    ResearchWorkflowResult,
)


def test_source_runtime_routes_only_explicit_research(monkeypatch):
    provider = ScriptedProvider([ModelResponse("ordinary")])
    runtime = agent.SourceRuntime(provider, memory_enabled=False)

    class ForbiddenWorkflow:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ordinary task must not construct research workflow")

    monkeypatch.setattr(agent, "ResearchWorkflow", ForbiddenWorkflow)
    assert runtime.run_turn(
        "q", run_metadata={"task_type": "normal"}
    ) == "ordinary"


@pytest.mark.parametrize("task_type", ["research", "research_analysis"])
def test_source_runtime_routes_research_aliases(monkeypatch, task_type):
    provider = ScriptedProvider([])
    runtime = agent.SourceRuntime(provider, memory_enabled=False)

    class FakeWorkflow:
        def __init__(self, *args, **kwargs):
            self.constructor_arguments = (args, kwargs)

        def run(self, question, cutoff, *, registry=None):
            return ResearchWorkflowResult(
                "research answer",
                ResearchPlan(ResearchRank.LIGHT, ("facts",), "narrow"),
                2,
                False,
                False,
            )

    monkeypatch.setattr(agent, "ResearchWorkflow", FakeWorkflow)
    assert runtime.run_turn(
        "q", run_metadata={"task_type": task_type}
    ) == "research answer"
~~~

Add this benchmark propagation test:

~~~python
from simple_cc.agent import AgentLoopOutcome
from simple_cc.benchmark import BenchmarkCloseOutcome


def test_execute_task_passes_task_type_to_runtime(tmp_path, monkeypatch):
    captured = {}

    class FakeRuntime:
        last_outcome = AgentLoopOutcome("completed", "answer")

        def run_turn(self, query, **kwargs):
            captured.update(kwargs)
            return "answer"

    class FakeSession:
        runtime = FakeRuntime()

        def close(self):
            return BenchmarkCloseOutcome(True, ())

    monkeypatch.setattr("eval.run_task.build_benchmark_runtime", lambda **kwargs: FakeSession())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = TaskInput(
        "run-route", "task-route", "q", None, "test", "research_analysis"
    )
    assert execute_task(
        task,
        run_dir,
        run_dir / "agent_workspace",
        ScriptedProvider([]),
        model="test-model",
    ) == 0
    assert captured["run_metadata"]["task_type"] == "research_analysis"
~~~

- [ ] **Step 2: Run routing tests and verify metadata is still discarded**

Run: `python -m pytest tests/test_agent_loop_source.py tests/test_benchmark_worker.py -k "routes or task_type" -v`

Expected: failures because `run_metadata` is discarded and the runtime has no workflow route.

- [ ] **Step 3: Implement routing inside the current RunContext boundary**

Normalize with:

~~~python
metadata = dict(run_metadata or {})
task_kind = normalize_task_kind(metadata.get("task_type"))
~~~

For ordinary tasks, call the retained loop with `ordinary_system_prompt(self.state_builder())` and no evidence registry.

For research tasks:

1. create a run-local `EvidenceRegistry`;
2. create an executor closure that calls `agent_loop()` with the research prompt, supplied `max_rounds`, shared registry, `finalize_user_turn=False`, current tools/handlers/permissions, and active run context;
3. construct `ResearchWorkflow(self.tracing_provider, executor, run_context=run)`;
4. run it with query and cutoff;
5. adapt its result to `AgentLoopOutcome("completed", result.final_text, rounds_used=result.research_rounds_used)`;
6. append exactly one terminal assistant text message to `self.messages`;
7. trigger the Stop hook once.

Record `task_routed` before either path with raw task type, normalized kind, and default/explicit reason. For final-answer linkage, research uses the run-local registry’s source map; ordinary tasks use an empty source map and are never gated.

- [ ] **Step 4: Pass task type explicitly from worker**

~~~python
session.runtime.run_turn(
    task.question,
    task_id=task.task_id,
    cutoff=task.cutoff,
    run_metadata={**metadata, "task_type": task.task_type},
)
~~~

Replace the old two-response `unsupported_research_provider()` test helper with scripted planner, two failed gates, draft, and rewrite responses so benchmark tests reflect the staged workflow.

- [ ] **Step 5: Run runtime and worker tests**

Run: `python -m pytest tests/test_agent_loop_source.py tests/test_benchmark_worker.py -v`

Expected: all tests pass. Ordinary traced output is accepted directly; explicit research input runs the staged workflow.

- [ ] **Step 6: Commit**

~~~bash
git add simple_cc/agent.py eval/run_task.py tests/test_agent_loop_source.py tests/test_benchmark_worker.py
git commit -m "feat: route explicit research tasks"
~~~

---

### Task 7: Verify Trace Semantics and Full Regression Compatibility

**Files:**
- Modify: `tests/test_research_workflow.py`
- Modify: `tests/test_evidence_trace.py`
- Modify: `tests/test_benchmark_worker.py`
- Modify: `README.md` only if it still claims every task is financial research.

**Interfaces:**
- Consumes the completed workflow and existing `TraceRecorder`/`read_trace_lines()`.
- Produces end-to-end acceptance evidence for routing, budgets, phase order, retry caps, source linkage, and publication.

- [ ] **Step 1: Add phase-order and budget trace test**

Create an offline research run whose first gate fails, supplemental research succeeds, and writing passes. Assert filtered events equal:

~~~python
assert workflow_events == [
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
assert second_attempt["payload"]["supplied_rounds"] == (
    selected_policy.max_research_rounds
    - first_attempt["payload"]["used_rounds"]
)
~~~

Assert every authority ID appears in a prior `source_registered` event and every workflow event has `agent_id="root"`.

- [ ] **Step 2: Add exact retry-cap tests**

~~~python
assert len([
    row for row in rows
    if row["event_type"] == "research_attempt_started"
]) == 2
assert len([
    row for row in rows
    if row["event_type"] == "writing_repair_started"
]) == 1
assert final_answer.startswith("INSUFFICIENT_EVIDENCE")
~~~

Add a passing initial-gate case asserting one research attempt and no rewrite.

- [ ] **Step 3: Run focused acceptance tests**

Run: `python -m pytest tests/test_research_models.py tests/test_research_workflow.py tests/test_evidence_trace.py tests/test_agent_loop_source.py tests/test_context_prompts.py tests/test_benchmark_worker.py -v`

Expected: all pass without network access.

- [ ] **Step 4: Run the complete suite**

Run: `python -m pytest -q`

Expected: all tests pass. Warnings about pre-existing inaccessible temporary directories are acceptable; new tests may not be skipped.

- [ ] **Step 5: Check formatting, placeholders, and scope**

Run:

~~~bash
git diff --check
rg -n "TBD|TODO|implement later|fill in details" simple_cc tests eval README.md
git status --short
~~~

Expected: no whitespace errors, no new placeholders, and only intended task files plus pre-existing unrelated untracked files.

- [ ] **Step 6: Update README when needed and commit acceptance coverage**

If README still says every invocation is financial research, replace it with:

~~~text
Simple CC routes explicitly marked research tasks through a bounded evidence workflow; ordinary tasks continue to use the general agent loop.
~~~

Commit only changed acceptance files:

~~~bash
git add tests/test_research_workflow.py tests/test_evidence_trace.py tests/test_benchmark_worker.py README.md
git commit -m "test: verify routed research workflow"
~~~

Omit README from `git add` when it did not change.

---

## Final Acceptance Commands

~~~bash
python -m pytest tests/test_research_models.py tests/test_research_workflow.py tests/test_evidence_trace.py tests/test_agent_loop_source.py tests/test_context_prompts.py tests/test_benchmark_worker.py -v
python -m pytest -q
git diff --check
git status --short
~~~

Completion requires all tests to pass, exact fixed rank budgets, no research workflow for ordinary tasks, no more than two research attempts, no more than two writing attempts, auditable phase events, and no modifications to unrelated dirty-worktree files.
