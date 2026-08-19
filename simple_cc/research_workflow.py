from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Callable

from . import config
from .models import ChatProvider
from .research_models import (
    DirectionAssessment,
    EvidenceRegistry,
    ResearchExecutionOutcome,
    ResearchGateDecision,
    ResearchPlan,
    ResearchRank,
)
from .subagents import extract_text
from .telemetry import model_call_scope
from .trace import RunContext


STANDARD_FALLBACK_DIRECTIONS = (
    "primary facts and first-party evidence",
    "impact, risk, and independent corroboration",
)

_MODEL_CALL_KINDS = frozenset({
    "research_planning",
    "research_gate",
    "research_writing",
    "research_rewrite",
})
_FENCED_JSON = re.compile(
    r"\A```json[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _require_exact_schema(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ValueError(f"{label} missing field: {missing[0]}")
    if unexpected:
        raise ValueError(f"{label} unexpected field: {unexpected[0]}")


def json_object(text: str) -> dict[str, Any]:
    """Parse exactly one raw or fenced JSON object, with no surrounding prose."""
    if not isinstance(text, str):
        raise TypeError("structured model output must be text")
    candidate = text.strip()
    fenced = _FENCED_JSON.fullmatch(candidate)
    if candidate.startswith("```"):
        if fenced is None:
            raise ValueError("structured output must be one fenced json block")
        candidate = fenced.group("body").strip()
    value = json.loads(
        candidate,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonstandard_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("structured output must be a JSON object")
    return value


def parse_research_plan(text: str) -> ResearchPlan:
    try:
        value = json_object(text)
        _require_exact_schema(
            value,
            {"rank", "directions", "reason"},
            "research plan",
        )
        rank = ResearchRank(value.get("rank"))
        raw_directions = value.get("directions")
        if not isinstance(raw_directions, list):
            raise ValueError("plan directions must be a list")
        if not all(isinstance(item, str) for item in raw_directions):
            raise ValueError("plan directions must contain only strings")
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("plan reason must be non-empty")
        return ResearchPlan(rank, tuple(raw_directions), reason.strip())
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return ResearchPlan(
            ResearchRank.STANDARD,
            STANDARD_FALLBACK_DIRECTIONS,
            "planner output was invalid; using the standard fallback",
            used_fallback=True,
            validation_errors=(str(error),),
        )


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


def _non_empty_string(value: Any, error: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
    return value.strip()


def _source_ids(value: Any, known_ids: set[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError("direction source_ids must be a list of source ids")
    source_ids = tuple(value)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("repeated evidence source id in direction assessment")
    unknown = sorted(set(source_ids) - known_ids)
    if unknown:
        raise ValueError(f"unknown evidence source id: {unknown[0]}")
    return source_ids


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
        _require_exact_schema(
            value,
            {"directions", "authorities", "gaps"},
            "research gate",
        )
        raw_directions = value.get("directions")
        raw_authorities = value.get("authorities")
        raw_gaps = value.get("gaps")
        if not isinstance(raw_directions, list) or len(raw_directions) != len(
            plan.directions
        ):
            raise ValueError("gate must assess every planned direction exactly once")
        if not isinstance(raw_authorities, list):
            raise ValueError("gate authorities must be a list")
        if not isinstance(raw_gaps, list) or not all(
            isinstance(item, str) for item in raw_gaps
        ):
            raise ValueError("gate gaps must be a list of strings")

        known_ids = {item.source_id for item in records}
        seen_directions: set[str] = set()
        assessments: list[DirectionAssessment] = []
        for raw in raw_directions:
            if not isinstance(raw, dict):
                raise ValueError("gate direction assessments must be objects")
            _require_exact_schema(
                raw,
                {"direction", "covered", "source_ids", "reason"},
                "gate direction assessment",
            )
            direction = _non_empty_string(
                raw.get("direction"), "gate direction must be non-empty"
            )
            if direction not in plan.directions or direction in seen_directions:
                raise ValueError(
                    f"unknown or repeated research direction: {direction}"
                )
            seen_directions.add(direction)
            covered = raw.get("covered")
            if not isinstance(covered, bool):
                raise ValueError(f"direction covered must be boolean: {direction}")
            reason = _non_empty_string(
                raw.get("reason"), f"missing direction reason: {direction}"
            )
            source_ids = _source_ids(raw.get("source_ids"), known_ids)
            if covered and not source_ids:
                raise ValueError(
                    f"covered direction requires evidence source ids: {direction}"
                )
            assessments.append(DirectionAssessment(
                direction,
                covered,
                source_ids,
                reason,
            ))

        authority_decisions: list[tuple[str, bool, str]] = []
        seen_authority_ids: set[str] = set()
        for raw in raw_authorities:
            if not isinstance(raw, dict):
                raise ValueError("gate authority assessments must be objects")
            _require_exact_schema(
                raw,
                {"source_id", "is_authoritative", "reason"},
                "gate authority assessment",
            )
            source_id = _non_empty_string(
                raw.get("source_id"), "authority source_id must be non-empty"
            )
            if source_id not in known_ids:
                raise ValueError(f"unknown evidence source id: {source_id}")
            if source_id in seen_authority_ids:
                raise ValueError(f"repeated evidence source id: {source_id}")
            seen_authority_ids.add(source_id)
            authoritative = raw.get("is_authoritative")
            if not isinstance(authoritative, bool):
                raise ValueError(
                    f"authority decision must be boolean: {source_id}"
                )
            reason = _non_empty_string(
                raw.get("reason"), f"missing authority reason: {source_id}"
            )
            authority_decisions.append((source_id, authoritative, reason))

        for source_id, authoritative, reason in authority_decisions:
            registry.mark_authority(source_id, authoritative, reason)

        authoritative_ids = tuple(
            item.source_id for item in registry.records if item.authoritative
        )
        gaps = [item.strip() for item in raw_gaps if item.strip()]
        if source_count < plan.policy.distinct_source_count:
            gaps.append("distinct source target not met")
        if domain_count < plan.policy.distinct_source_count:
            gaps.append("independent domain target not met")
        if len(authoritative_ids) < plan.policy.authoritative_source_count:
            gaps.append("authoritative source target not met")
        gaps.extend(
            f"direction not covered: {item.direction}"
            for item in assessments
            if not item.covered
        )
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
        if kind not in _MODEL_CALL_KINDS:
            raise ValueError(f"unknown research model call kind: {kind}")
        with model_call_scope(kind):
            response = self.provider.create(
                messages=[{"role": "user", "content": user_content}],
                system=system,
                tools=[],
                max_tokens=config.DEFAULT_MAX_TOKENS,
            )
        return extract_text(response.content)

    def _record_output_artifact(self, text: str, source: str):
        if self.run_context is None:
            return None
        return self.run_context.recorder.store_artifact(
            text,
            media_type="text/plain",
            source=source,
            suffix=".txt",
        )

    def _record(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.run_context is None:
            return
        self.run_context.recorder.record(
            event_type,
            payload,
            parent_span_id=self.run_context.parent_span_id,
            agent_id=self.run_context.agent_id,
        )

    def plan(self, question: str, cutoff: str | None) -> ResearchPlan:
        system = (
            "Plan a bounded research task. Select exactly one fixed rank and "
            "return only a single JSON object, without Markdown or prose. Light "
            "requires 1 direction, standard 2, and deep 3. Do not invent numeric "
            "budgets. The schema is "
            '{"rank":"light|standard|deep","directions":["specific direction"],'
            '"reason":"brief rationale"}.'
        )
        user_content = json.dumps(
            {"question": question, "cutoff": cutoff},
            ensure_ascii=False,
            sort_keys=True,
        )
        raw = self._call_text("research_planning", system, user_content)
        plan = parse_research_plan(raw)
        artifact = self._record_output_artifact(raw, "research_plan_output")
        self._record("research_plan", {
            "rank": plan.rank.value,
            "directions": list(plan.directions),
            "reason": plan.reason,
            "fixed_policy": asdict(plan.policy),
            "used_fallback": plan.used_fallback,
            "validation_errors": list(plan.validation_errors),
            "raw_output_artifact": artifact.as_dict() if artifact else None,
        })
        return plan

    def evaluate_research(
        self,
        question: str,
        cutoff: str | None,
        plan: ResearchPlan,
        registry: EvidenceRegistry,
    ) -> ResearchGateDecision:
        system = (
            "Evaluate research sufficiency only from the supplied registered "
            "evidence. Return only one JSON object, without Markdown or prose, "
            "with this schema: "
            '{"directions":[{"direction":"exact planned direction",'
            '"covered":true,"source_ids":["registered source id"],'
            '"reason":"evidence rationale"}],"authorities":'
            '[{"source_id":"registered source id","is_authoritative":true,'
            '"reason":"authority rationale"}],"gaps":[]}. '
            "Assess every planned direction exactly once. Use only registered "
            "source IDs. Authority is contextual to the question and normally "
            "includes original publishers, official disclosures, regulators, "
            "exchanges, filings, and government sources rather than aggregators."
        )
        user_content = json.dumps({
            "question": question,
            "cutoff": cutoff,
            "plan": {
                "rank": plan.rank.value,
                "directions": list(plan.directions),
                "reason": plan.reason,
            },
            "fixed_policy": asdict(plan.policy),
            "evidence": build_evidence_packet(registry),
        }, ensure_ascii=False, sort_keys=True)
        raw = self._call_text("research_gate", system, user_content)
        decision = parse_research_gate(raw, plan, registry)
        artifact = self._record_output_artifact(raw, "research_gate_output")
        self._record("research_gate", {
            "passed": decision.passed,
            "source_count": decision.source_count,
            "domain_count": decision.domain_count,
            "authoritative_source_ids": list(
                decision.authoritative_source_ids
            ),
            "directions": [asdict(item) for item in decision.directions],
            "gaps": list(decision.gaps),
            "validation_errors": list(decision.validation_errors),
            "raw_output_artifact": artifact.as_dict() if artifact else None,
        })
        return decision
