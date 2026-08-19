from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Callable

from . import config
from .evidence import validate_research_final
from .models import ChatProvider
from .research_models import (
    DirectionAssessment,
    EvidenceRegistry,
    ResearchBudget,
    ResearchExecutionOutcome,
    ResearchGateDecision,
    ResearchPlan,
    ResearchRank,
    ResearchWorkflowResult,
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


class ResearchWorkflowError(RuntimeError):
    """Terminal research-execution failure with benchmark-safe details."""

    def __init__(self, failure_class: str, failure_message: str) -> None:
        self.failure_class = str(failure_class or "ResearchWorkflowError")
        self.failure_message = str(
            failure_message or "research workflow execution failed"
        )
        super().__init__(f"{self.failure_class}: {self.failure_message}")


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
        *,
        attempt: int | None = None,
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
            "attempt": attempt,
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

    def _research_prompt(
        self,
        question: str,
        cutoff: str | None,
        plan: ResearchPlan,
        gaps: tuple[str, ...],
        remaining_rounds: int,
    ) -> str:
        return json.dumps({
            "instructions": (
                "Research the supplied directions using fetched primary and "
                "independent sources. Register successful fetches in the shared "
                "evidence registry. Search snippets are leads, not evidence."
            ),
            "question": question,
            "cutoff": cutoff,
            "rank": plan.rank.value,
            "fixed_policy": asdict(plan.policy),
            "directions": list(plan.directions),
            "remaining_rounds": remaining_rounds,
            "research_gaps": list(gaps),
        }, ensure_ascii=False, sort_keys=True)

    def _consume_outcome(
        self,
        budget: ResearchBudget,
        outcome: ResearchExecutionOutcome,
        supplied_rounds: int,
    ) -> int:
        rounds_used = getattr(outcome, "rounds_used", None)
        if (
            isinstance(rounds_used, bool)
            or not isinstance(rounds_used, int)
            or rounds_used < 0
            or rounds_used > supplied_rounds
        ):
            raise ResearchWorkflowError(
                "ResearchBudgetExceeded",
                f"executor reported {rounds_used!r} rounds for a supplied "
                f"budget of {supplied_rounds}",
            )

        status = str(getattr(outcome, "status", ""))
        if status == "failed":
            raise ResearchWorkflowError(
                getattr(outcome, "failure_class", None)
                or "ResearchExecutionFailed",
                getattr(outcome, "failure_message", None)
                or "research executor returned a failed outcome",
            )
        if status == "max_rounds":
            budget.consume(supplied_rounds)
            return supplied_rounds
        if status != "completed":
            raise ResearchWorkflowError(
                "ResearchExecutionInvalidOutcome",
                f"unsupported research executor status: {status or '<empty>'}",
            )
        budget.consume(rounds_used)
        return rounds_used

    def _execute_research(
        self,
        question: str,
        cutoff: str | None,
        plan: ResearchPlan,
        gaps: tuple[str, ...],
        budget: ResearchBudget,
        registry: EvidenceRegistry,
        *,
        attempt: int,
    ) -> ResearchExecutionOutcome:
        supplied_rounds = budget.remaining_rounds
        supplemental = attempt == 2
        self._record("research_attempt_started", {
            "attempt": attempt,
            "supplemental": supplemental,
            "supplied_rounds": supplied_rounds,
            "used_rounds": budget.used_rounds,
            "remaining_rounds": budget.remaining_rounds,
            "directions": tuple(plan.directions),
            "gaps": tuple(gaps),
            "supplemental_research_used": supplemental,
        })
        prompt = self._research_prompt(
            question,
            cutoff,
            plan,
            gaps,
            supplied_rounds,
        )
        before = budget.used_rounds
        outcome: ResearchExecutionOutcome | None = None
        try:
            outcome = self.research_executor(prompt, supplied_rounds, registry)
            consumed = self._consume_outcome(budget, outcome, supplied_rounds)
        except ResearchWorkflowError as error:
            self._record("research_attempt_finished", {
                "attempt": attempt,
                "supplemental": supplemental,
                "status": "failed",
                "supplied_rounds": supplied_rounds,
                "reported_rounds": (
                    outcome.rounds_used if outcome is not None else None
                ),
                "consumed_rounds": budget.used_rounds - before,
                "used_rounds": budget.used_rounds,
                "remaining_rounds": budget.remaining_rounds,
                "failure_class": error.failure_class,
                "failure_message": error.failure_message,
            })
            raise
        except Exception as error:
            wrapped = ResearchWorkflowError(type(error).__name__, str(error))
            self._record("research_attempt_finished", {
                "attempt": attempt,
                "supplemental": supplemental,
                "status": "failed",
                "supplied_rounds": supplied_rounds,
                "reported_rounds": None,
                "consumed_rounds": 0,
                "used_rounds": budget.used_rounds,
                "remaining_rounds": budget.remaining_rounds,
                "failure_class": wrapped.failure_class,
                "failure_message": wrapped.failure_message,
            })
            raise wrapped from error
        self._record("research_attempt_finished", {
            "attempt": attempt,
            "supplemental": supplemental,
            "status": outcome.status,
            "supplied_rounds": supplied_rounds,
            "reported_rounds": outcome.rounds_used,
            "consumed_rounds": consumed,
            "used_rounds": budget.used_rounds,
            "remaining_rounds": budget.remaining_rounds,
            "failure_class": outcome.failure_class,
            "failure_message": outcome.failure_message,
        })
        return outcome

    def _writing_input(
        self,
        question: str,
        cutoff: str | None,
        plan: ResearchPlan,
        registry: EvidenceRegistry,
        gaps: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "question": question,
            "cutoff": cutoff,
            "plan": {
                "rank": plan.rank.value,
                "directions": list(plan.directions),
                "reason": plan.reason,
            },
            "fixed_requirements": asdict(plan.policy),
            "evidence": build_evidence_packet(registry),
            "authority_decisions": [{
                "source_id": item.source_id,
                "is_authoritative": item.authoritative,
                "reason": item.authority_reason,
            } for item in registry.records],
            "unresolved_research_gaps": list(gaps),
        }

    def write(
        self,
        question: str,
        cutoff: str | None,
        plan: ResearchPlan,
        registry: EvidenceRegistry,
        gaps: tuple[str, ...],
    ) -> str:
        system = (
            "Write the final answer only from the supplied registered evidence. "
            "Cite exact fetched URLs. Distinguish verified facts, inference, and "
            "uncertainty. Disclose every unresolved research gap. Do not invent "
            "or cite any URL absent from the evidence packet."
        )
        return self._call_text(
            "research_writing",
            system,
            json.dumps(
                self._writing_input(question, cutoff, plan, registry, gaps),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    def rewrite(
        self,
        question: str,
        cutoff: str | None,
        plan: ResearchPlan,
        registry: EvidenceRegistry,
        gaps: tuple[str, ...],
        rejected_draft: str,
        errors: list[str],
    ) -> str:
        system = (
            "Rewrite the rejected report once using only the supplied evidence. "
            "Fix every listed validation error. Do not search, request tools, or "
            "introduce new URLs. Return only the revised final report."
        )
        content = self._writing_input(question, cutoff, plan, registry, gaps)
        content.update({
            "rejected_draft": rejected_draft,
            "validation_errors": list(errors),
        })
        return self._call_text(
            "research_rewrite",
            system,
            json.dumps(content, ensure_ascii=False, sort_keys=True),
        )

    def run(
        self,
        question: str,
        cutoff: str | None,
        *,
        registry: EvidenceRegistry | None = None,
    ) -> ResearchWorkflowResult:
        evidence_registry = registry if registry is not None else EvidenceRegistry()
        plan = self.plan(question, cutoff)
        budget = ResearchBudget(plan.policy.max_research_rounds)

        self._execute_research(
            question,
            cutoff,
            plan,
            (),
            budget,
            evidence_registry,
            attempt=1,
        )
        gate = self.evaluate_research(
            question,
            cutoff,
            plan,
            evidence_registry,
            attempt=1,
        )

        supplemental_used = False
        if not gate.passed and budget.remaining_rounds > 0:
            supplemental_used = True
            self._execute_research(
                question,
                cutoff,
                plan,
                gate.gaps,
                budget,
                evidence_registry,
                attempt=2,
            )
            gate = self.evaluate_research(
                question,
                cutoff,
                plan,
                evidence_registry,
                attempt=2,
            )
        else:
            self._record("supplemental_research_skipped", {
                "reason": (
                    "initial research gate passed"
                    if gate.passed
                    else "research round budget exhausted"
                ),
                "remaining_rounds": budget.remaining_rounds,
                "gate_gaps": tuple(gate.gaps),
                "supplemental_research_used": False,
            })

        self._record("writing_attempt_started", {
            "attempt": 1,
            "repair": False,
            "gaps": tuple(gate.gaps),
            "source_ids": tuple(
                item.source_id for item in evidence_registry.records
            ),
            "authoritative_source_ids": tuple(
                item.source_id
                for item in evidence_registry.records
                if item.authoritative
            ),
            "supplemental_research_used": supplemental_used,
            "writing_repair_used": False,
        })
        draft = self.write(
            question,
            cutoff,
            plan,
            evidence_registry,
            gate.gaps,
        )
        errors = validate_research_final(draft, evidence_registry, plan)
        self._record("writing_gate", {
            "attempt": 1,
            "passed": not errors,
            "validation_errors": tuple(errors),
            "source_count": len(evidence_registry.records),
            "domain_count": len({
                item.domain for item in evidence_registry.records if item.domain
            }),
            "authoritative_source_ids": tuple(
                item.source_id
                for item in evidence_registry.records
                if item.authoritative
            ),
            "writing_repair_used": False,
        })

        repair_used = False
        if errors:
            repair_used = True
            self._record("writing_repair_started", {
                "attempt": 2,
                "validation_errors": tuple(errors),
                "gaps": tuple(gate.gaps),
                "supplemental_research_used": supplemental_used,
                "writing_repair_used": True,
            })
            draft = self.rewrite(
                question,
                cutoff,
                plan,
                evidence_registry,
                gate.gaps,
                draft,
                errors,
            )
            errors = validate_research_final(draft, evidence_registry, plan)
            self._record("writing_gate", {
                "attempt": 2,
                "passed": not errors,
                "validation_errors": tuple(errors),
                "source_count": len(evidence_registry.records),
                "domain_count": len({
                    item.domain
                    for item in evidence_registry.records
                    if item.domain
                }),
                "authoritative_source_ids": tuple(
                    item.source_id
                    for item in evidence_registry.records
                    if item.authoritative
                ),
                "writing_repair_used": True,
            })

        terminal_reason = "completed"
        if errors:
            terminal_reason = "insufficient_evidence"
            draft = (
                "INSUFFICIENT_EVIDENCE\n\nResearch finalization failed:\n"
                + "\n".join(f"- {error}" for error in errors)
            )
        result = ResearchWorkflowResult(
            draft,
            plan,
            budget.used_rounds,
            supplemental_used,
            repair_used,
        )
        self._record("research_workflow_completed", {
            "terminal_reason": terminal_reason,
            "rank": plan.rank.value,
            "research_rounds_used": budget.used_rounds,
            "remaining_rounds": budget.remaining_rounds,
            "supplemental_research_used": supplemental_used,
            "writing_repair_used": repair_used,
            "final_validation_errors": tuple(errors),
            "remaining_gaps": tuple(gate.gaps),
        })
        return result
