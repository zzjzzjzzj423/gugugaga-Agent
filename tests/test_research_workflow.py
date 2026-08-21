from __future__ import annotations

import json
import sys

import pytest

from simple_cc import agent, config, research_workflow
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
    TraceWriteError,
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


def _exception_slot(name: str):
    return vars(BaseException)[name]


def _get_exception_slot(error: BaseException, name: str):
    return _exception_slot(name).__get__(error, BaseException)


def _set_exception_slot(error: BaseException, name: str, value) -> None:
    _exception_slot(name).__set__(error, value)


def _provider_error_with_origin(message: str) -> tuple[RuntimeError, object]:
    return _error_with_origin(RuntimeError(message))


def _error_with_origin(error: RuntimeError) -> tuple[RuntimeError, object]:
    try:
        raise error
    except RuntimeError as caught:
        return caught, _get_exception_slot(caught, "__traceback__")


def _install_provider_diagnostics(
    error: RuntimeError,
    mode: str,
) -> tuple[BaseException | None, BaseException | None, bool]:
    cause: BaseException | None = None
    context: BaseException | None = None
    suppress_context = False
    if mode == "explicit_cause":
        cause = LookupError("provider cause")
        suppress_context = True
    elif mode == "context":
        context = LookupError("provider context")
    elif mode == "suppressed_context":
        context = LookupError("suppressed provider context")
        suppress_context = True
    _set_exception_slot(error, "__cause__", cause)
    _set_exception_slot(error, "__context__", context)
    _set_exception_slot(error, "__suppress_context__", suppress_context)
    return cause, context, suppress_context


def _assert_provider_diagnostics(
    error: RuntimeError,
    *,
    cause: BaseException | None,
    context: BaseException | None,
    suppress_context: bool,
) -> None:
    if _get_exception_slot(error, "__cause__") is not cause:
        pytest.fail("provider cause identity changed", pytrace=False)
    if _get_exception_slot(error, "__context__") is not context:
        pytest.fail("provider context identity changed", pytrace=False)
    if _get_exception_slot(error, "__suppress_context__") is not suppress_context:
        pytest.fail("provider suppress-context flag changed", pytrace=False)


def _audit_notes(error: BaseException, event_type: str) -> list[str]:
    try:
        notes = BaseException.__getattribute__(error, "__notes__")
    except AttributeError:
        return []
    return [note for note in notes if event_type in note]


class _HostileProtocolTrap(RuntimeError):
    pass


class _HostileExceptionMeta(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            raise _HostileProtocolTrap("hostile class-name lookup")
        return type.__getattribute__(cls, name)


class _HostileProviderError(RuntimeError, metaclass=_HostileExceptionMeta):
    @property
    def failure_class(self):
        raise _HostileProtocolTrap("hostile failure_class getter")

    @property
    def failure_message(self):
        raise _HostileProtocolTrap("hostile failure_message getter")

    def __getattribute__(self, name):
        if name in {
            "failure_class",
            "failure_message",
            "add_note",
            "with_traceback",
            "__traceback__",
            "__cause__",
            "__context__",
            "__suppress_context__",
        }:
            raise _HostileProtocolTrap(f"hostile {name} lookup")
        return BaseException.__getattribute__(self, name)

    def __str__(self):
        raise _HostileProtocolTrap("hostile string conversion")

    def add_note(self, note):
        raise _HostileProtocolTrap("hostile add_note call")

    def with_traceback(self, traceback):
        raise _HostileProtocolTrap("hostile with_traceback call")


class _MaskedDiagnosticProviderError(RuntimeError):
    def __init__(self, message: str, mask_mode: str) -> None:
        super().__init__(message)
        self.mask_mode = mask_mode

    def _masked(self, fallback):
        if self.mask_mode == "raise":
            raise _HostileProtocolTrap("masked diagnostic descriptor")
        return fallback

    @property
    def __cause__(self):
        return self._masked(None)

    @property
    def __context__(self):
        return self._masked(None)

    @property
    def __suppress_context__(self):
        return self._masked(False)

    @property
    def __traceback__(self):
        return self._masked(None)

    @__traceback__.setter
    def __traceback__(self, value):
        _set_exception_slot(self, "__traceback__", value)


class _HostileSpecialSetattrProviderError(RuntimeError):
    def __setattr__(self, name, value):
        if name in {
            "__cause__",
            "__context__",
            "__suppress_context__",
        }:
            raise _HostileProtocolTrap(f"hostile {name} assignment")
        return BaseException.__setattr__(self, name, value)


class _HostileNoteProviderError(RuntimeError):
    def __setattr__(self, name, value):
        if name == "__notes__":
            raise _HostileProtocolTrap("hostile note assignment")
        return BaseException.__setattr__(self, name, value)


class _MaskedAuditTraceError(TraceWriteError):
    def __init__(self, message: str, mask_mode: str = "return_none") -> None:
        super().__init__(message)
        self.mask_mode = mask_mode

    def _masked(self):
        if self.mask_mode == "raise":
            raise _HostileProtocolTrap("masked audit descriptor")
        return None

    @property
    def __cause__(self):
        return self._masked()

    @property
    def __context__(self):
        return self._masked()


class _HostileStringTraceError(TraceWriteError):
    def __str__(self):
        raise _HostileProtocolTrap("hostile audit string conversion")


class _AttachMutationCauseDescriptor:
    def __init__(
        self,
        provider_error: RuntimeError,
        trace_errors: tuple[TraceWriteError, ...],
    ) -> None:
        self.delegate = _exception_slot("__cause__")
        self.provider_error = provider_error
        self.trace_error_ids = {id(error) for error in trace_errors}
        self.attach_mutations = 0

    def __get__(self, instance, owner=None):
        return self.delegate.__get__(instance, BaseException)

    def __set__(self, instance, value) -> None:
        is_attach = (
            instance is self.provider_error
            and id(value) in self.trace_error_ids
        )
        self.delegate.__set__(instance, value)
        if is_attach:
            self.attach_mutations += 1
            _set_exception_slot(value, "__context__", self.provider_error)


def _trace_failure_kind(event_type: str, payload: dict) -> str | None:
    if (
        event_type == "writing_attempt_finished"
        and payload.get("status") == "failed"
    ):
        return "writing_failed"
    if event_type == "research_workflow_completed":
        return "terminal"
    return None


def _trace_failure_message(kind: str) -> str:
    if kind == "terminal":
        return "terminal\ntrace " + "x" * 2_000
    return "failed\nfinished trace " + "x" * 2_000


def _install_trace_failures(
    monkeypatch,
    recorder: TraceRecorder,
    *,
    kinds: set[str],
    mode: str,
) -> tuple[
    dict[str, TraceWriteError],
    dict[str, OSError],
    dict[str, BaseException | None],
]:
    trace_errors: dict[str, TraceWriteError] = {}
    os_errors: dict[str, OSError] = {}
    active_exceptions: dict[str, BaseException | None] = {}
    original_record = recorder.record
    if mode == "preconstructed":
        trace_errors.update({
            kind: TraceWriteError(_trace_failure_message(kind))
            for kind in kinds
        })

        def fail_record(event_type, payload, **kwargs):
            kind = _trace_failure_kind(event_type, payload)
            if kind in kinds:
                active_exceptions[kind] = sys.exception()
                raise trace_errors[kind]
            return original_record(event_type, payload, **kwargs)

        monkeypatch.setattr(recorder, "record", fail_record)
        return trace_errors, os_errors, active_exceptions

    original_append = recorder._append_line

    def fail_append(serialized):
        event = json.loads(serialized)
        kind = _trace_failure_kind(event["event_type"], event["payload"])
        if kind in kinds:
            active_exceptions[kind] = sys.exception()
            os_error = OSError(_trace_failure_message(kind))
            os_errors[kind] = os_error
            raise os_error
        return original_append(serialized)

    def capture_translated_trace(event_type, payload, **kwargs):
        kind = _trace_failure_kind(event_type, payload)
        try:
            return original_record(event_type, payload, **kwargs)
        except TraceWriteError as trace_error:
            if kind in kinds:
                trace_errors[kind] = trace_error
            raise

    monkeypatch.setattr(recorder, "_append_line", fail_append)
    monkeypatch.setattr(recorder, "record", capture_translated_trace)
    return trace_errors, os_errors, active_exceptions


def _provider_failure_workflow(
    tmp_path,
    monkeypatch,
    *,
    provider_error: RuntimeError,
    failed_attempt: int,
    failed_kinds: set[str],
    run_id: str,
):
    registry = registry_with_two_sources()
    responses = [
        light_plan_response(),
        gate_response(registry, covered=True),
    ]
    if failed_attempt == 2:
        responses.append(ModelResponse("unsupported draft"))
    responses.append(provider_error)
    provider = ScriptedProvider(responses)
    recorder = TraceRecorder(tmp_path / "run", run_id=run_id)
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        run_id,
        "research-task",
        "2025-05-01",
    )
    trace_errors, _, active_exceptions = _install_trace_failures(
        monkeypatch,
        recorder,
        kinds=failed_kinds,
        mode="preconstructed",
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )
    return workflow, run, registry, trace_errors, active_exceptions


def _configure_audit_graph(
    trace_error: TraceWriteError,
    provider_error: RuntimeError,
    graph_kind: str,
) -> None:
    if graph_kind in {"clean", "unreadable"}:
        return
    if graph_kind == "cause_provider":
        _set_exception_slot(trace_error, "__cause__", provider_error)
        return
    if graph_kind == "context_provider":
        _set_exception_slot(trace_error, "__context__", provider_error)
        return
    if graph_kind == "self_cycle":
        _set_exception_slot(trace_error, "__cause__", trace_error)
        return
    if graph_kind == "two_node_cycle":
        other = OSError("audit graph peer")
        _set_exception_slot(trace_error, "__cause__", other)
        _set_exception_slot(other, "__context__", trace_error)
        return
    if graph_kind == "too_deep":
        current: BaseException = trace_error
        for index in range(40):
            next_error = RuntimeError(f"audit graph node {index}")
            _set_exception_slot(current, "__cause__", next_error)
            current = next_error
        return
    raise AssertionError(f"unknown graph kind: {graph_kind}")


def _mask_audit_graph_back_reference(
    trace_errors: dict[str, TraceWriteError],
    provider_error: RuntimeError,
    graph_kind: str,
) -> bool:
    if graph_kind not in {
        "masked_cause_provider",
        "masked_context_provider",
        "masked_cause_provider_raise",
        "masked_context_provider_raise",
    }:
        return False
    field = (
        "__cause__"
        if graph_kind.startswith("masked_cause_provider")
        else "__context__"
    )
    mask_mode = "raise" if graph_kind.endswith("_raise") else "return_none"
    for kind in tuple(trace_errors):
        masked_error = _MaskedAuditTraceError(
            _trace_failure_message(kind),
            mask_mode,
        )
        _set_exception_slot(masked_error, field, provider_error)
        trace_errors[kind] = masked_error
    return True


def _assert_acyclic_exception_graph(error: BaseException) -> list[BaseException]:
    visiting: set[int] = set()
    visited: set[int] = set()
    nodes: list[BaseException] = []

    def visit(node: BaseException | None) -> None:
        if node is None:
            return
        identity = id(node)
        if identity in visiting:
            raise AssertionError("exception graph contains a cycle")
        if identity in visited:
            return
        visiting.add(identity)
        nodes.append(node)
        visit(_get_exception_slot(node, "__cause__"))
        visit(_get_exception_slot(node, "__context__"))
        visiting.remove(identity)
        visited.add(identity)

    visit(error)
    return nodes


def _traceback_contains(traceback, target) -> bool:
    while traceback is not None:
        if traceback is target:
            return True
        traceback = traceback.tb_next
    return False


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
                registered.source_id, True, "official source"
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


def _register_packet_size_probe(
    registry,
    url,
    *,
    large=False,
    authoritative=False,
):
    record = evidence_record_from_result(
        "web_fetch",
        json.dumps({
            "ok": True,
            "operation": "fetch",
            "url": url,
            "title": "\\" * 1000 if large else "medium",
            "content": "\\" * 6000 if large else "useful evidence",
            "published_at": "2025-01-02",
            "date_status": "verified",
            "cutoff": "2025-05-01",
        }),
    )
    assert record is not None
    registered = registry.register(record)
    if authoritative:
        registry.mark_authority(registered.source_id, True, "official")
    return registered


def _large_probe_url(domain, label):
    prefix = f"https://{domain}/{label}/"
    return prefix + "\\" * (4000 - len(prefix))


def test_size_aware_seed_finds_feasible_deep_minimum_before_stable_fill():
    registry = EvidenceRegistry()
    authorities = {
        _register_packet_size_probe(
            registry,
            _large_probe_url("official.example", f"authority-{index}"),
            large=True,
            authoritative=True,
        ).source_id
        for index in range(2)
    }
    for index in range(3):
        _register_packet_size_probe(
            registry,
            _large_probe_url(f"large-{index}.example", "large"),
            large=True,
        )
    for index in range(40):
        _register_packet_size_probe(
            registry,
            f"https://official.example/filler-{index}",
        )
    for index in range(3):
        _register_packet_size_probe(
            registry,
            f"https://medium-{index}.example/report",
        )

    packet = build_evidence_packet(registry)

    assert authorities <= {item["source_id"] for item in packet}
    assert len({item["domain"] for item in packet}) >= 4
    assert len(packet) <= EVIDENCE_PACKET_MAX_RECORDS
    assert len(json.dumps(packet, ensure_ascii=False, sort_keys=True)) <= (
        EVIDENCE_PACKET_TOTAL_CHARS_MAX
    )


def test_size_aware_seed_stays_bounded_when_no_deep_subset_is_feasible():
    registry = EvidenceRegistry()
    authorities = {
        _register_packet_size_probe(
            registry,
            _large_probe_url("official.example", f"authority-{index}"),
            large=True,
            authoritative=True,
        ).source_id
        for index in range(2)
    }
    for index in range(3):
        _register_packet_size_probe(
            registry,
            _large_probe_url(f"only-{index}.example", "large"),
            large=True,
        )

    packet = build_evidence_packet(registry)
    selected_ids = {item["source_id"] for item in packet}

    assert not (
        authorities <= selected_ids
        and len({item["domain"] for item in packet}) >= 4
    )
    assert len(packet) <= EVIDENCE_PACKET_MAX_RECORDS
    assert len(json.dumps(packet, ensure_ascii=False, sort_keys=True)) <= (
        EVIDENCE_PACKET_TOTAL_CHARS_MAX
    )


def _escaped_probe_url(domain, label, target_length):
    prefix = f"https://{domain}/{label}/"
    return prefix + "\\" * (target_length - len(prefix))


def test_exact_seed_optimizer_finds_feasible_authorities_beyond_old_64_cap():
    registry = EvidenceRegistry()
    for index in range(64):
        _register_packet_size_probe(
            registry,
            _escaped_probe_url(
                "crowded.example", f"small-authority-{index}", 3200
            ),
            large=True,
            authoritative=True,
        )
    medium_authorities = {
        _register_packet_size_probe(
            registry,
            _escaped_probe_url(
                f"medium-authority-{index}.example", "authority", 3700
            ),
            large=True,
            authoritative=True,
        ).source_id
        for index in range(2)
    }
    _register_packet_size_probe(
        registry,
        _escaped_probe_url("fourth-domain.example", "evidence", 3700),
        large=True,
    )

    packet = build_evidence_packet(registry)

    assert medium_authorities <= {item["source_id"] for item in packet}
    assert sum(
        registry.get_by_id(item["source_id"]).authoritative is True
        for item in packet
    ) >= 2
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


@pytest.mark.parametrize("omitted_use", ("direction", "authority"))
def test_actual_gate_rejects_ids_omitted_from_40_record_packet(omitted_use):
    registry = EvidenceRegistry()
    for index in range(40):
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "operation": "fetch",
                "url": f"https://domain-{index}.example/report",
                "title": f"Evidence {index}",
                "content": f"direct evidence {index}",
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }),
        )
        assert record is not None
        registry.register(record)
    packet = build_evidence_packet(registry)
    assert len(packet) == EVIDENCE_PACKET_MAX_RECORDS
    selected_ids = {item["source_id"] for item in packet}
    omitted_id = next(
        item.source_id
        for item in registry.records
        if item.source_id not in selected_ids
    )
    payload = valid_gate_payload(registry)
    selected_id = packet[0]["source_id"]
    payload["directions"][0]["source_ids"] = [selected_id]
    payload["authorities"][0]["source_id"] = selected_id
    if omitted_use == "direction":
        payload["directions"][0]["source_ids"] = [omitted_id]
    else:
        payload["authorities"][0]["source_id"] = omitted_id
    provider = ScriptedProvider([ModelResponse(json.dumps(payload))])

    decision = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: None,
    ).evaluate_research("question", "2025-05-01", light_plan(), registry)

    request = json.loads(provider.requests[0]["messages"][0]["content"])
    assert omitted_id not in {item["source_id"] for item in request["evidence"]}
    assert decision.passed is False
    assert decision.source_count == 40
    assert decision.domain_count == 40
    assert "unknown evidence source id" in " ".join(decision.validation_errors)
    assert not any(item.authoritative for item in registry.records)


def test_final_gate_packet_is_reused_for_writing_rewrite_and_validation(tmp_path):
    registry = EvidenceRegistry()
    for index in range(40):
        record = evidence_record_from_result(
            "web_fetch",
            json.dumps({
                "ok": True,
                "operation": "fetch",
                "url": f"https://packet-{index}.example/report",
                "title": f"Evidence {index}",
                "content": f"direct evidence {index}",
                "published_at": "2025-01-02",
                "date_status": "verified",
                "cutoff": "2025-05-01",
            }),
        )
        assert record is not None
        registry.register(record)

    class PacketAwareProvider:
        def __init__(self):
            self.requests = []
            self.gate_packet = None
            self.writing_packet = None
            self.rewrite_packet = None
            self.gate_source_id = None
            self.omitted_url = None

        def create(self, messages, system, tools, max_tokens=8192, model=None):
            self.requests.append({
                "messages": messages,
                "system": system,
                "tools": tools,
                "max_tokens": max_tokens,
                "model": model,
            })
            call_number = len(self.requests)
            if call_number == 1:
                return light_plan_response()
            payload = json.loads(messages[0]["content"])
            if call_number == 2:
                self.gate_packet = payload["evidence"]
                assert len(self.gate_packet) == EVIDENCE_PACKET_MAX_RECORDS
                selected_ids = {
                    item["source_id"] for item in self.gate_packet
                }
                self.gate_source_id = self.gate_packet[-1]["source_id"]
                self.omitted_url = next(
                    item.canonical_url
                    for item in registry.records
                    if item.source_id not in selected_ids
                )
                return ModelResponse(json.dumps({
                    "directions": [{
                        "direction": "primary filings",
                        "covered": True,
                        "source_ids": [self.gate_source_id],
                        "reason": "direct support",
                    }],
                    "authorities": [{
                        "source_id": self.gate_source_id,
                        "is_authoritative": True,
                        "reason": "official disclosure",
                    }],
                    "gaps": [],
                }))
            if call_number == 3:
                self.writing_packet = payload["evidence"]
                selected_url = self.writing_packet[0]["url"]
                return ModelResponse(
                    f"Draft {selected_url} {self.omitted_url}"
                )
            if call_number == 4:
                self.rewrite_packet = payload["evidence"]
                assert "final answer contains unfetched citations" in (
                    payload["validation_errors"]
                )
                return ModelResponse(
                    f"Rewritten {self.rewrite_packet[0]['url']}"
                )
            raise AssertionError("unexpected model call")

    provider = PacketAwareProvider()
    recorder = TraceRecorder(tmp_path / "run", run_id="packet-finalization-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        "packet-finalization-run",
        "research-task",
        "2025-05-01",
    )

    result = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    ).run("question", "2025-05-01", registry=registry)

    assert result.writing_repair_used is True
    assert provider.gate_source_id in {
        item["source_id"] for item in provider.gate_packet
    }
    assert provider.writing_packet == provider.gate_packet
    assert provider.rewrite_packet == provider.gate_packet
    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert incomplete is False
    selections = [
        row["payload"]
        for row in rows
        if row["event_type"] == "evidence_packet_selected"
    ]
    assert [item["stage"] for item in selections] == [
        "research_execution",
        "research_gate",
    ]
    reused = [
        row["payload"]
        for row in rows
        if row["event_type"] == "evidence_packet_reused"
    ]
    assert [item["stage"] for item in reused] == [
        "research_writing",
        "research_rewrite",
    ]
    assert all(
        item["selected_source_ids"]
        == [record["source_id"] for record in provider.gate_packet]
        for item in reused
    )


def test_research_agent_trace_failure_escapes_source_runtime(tmp_path, monkeypatch):
    provider = ScriptedProvider([
        light_plan_response(),
        ModelResponse("private research notes"),
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="agent-trace-failure-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff=None,
        metadata={"task_type": "research"},
    )
    original_record = recorder.record
    failed = False

    def fail_research_agent_request_once(event_type, payload, **kwargs):
        nonlocal failed
        if (
            not failed
            and event_type == "llm_request_started"
            and payload.get("call_kind") == "agent"
        ):
            failed = True
            raise TraceWriteError("research agent trace unavailable")
        return original_record(event_type, payload, **kwargs)

    monkeypatch.setattr(recorder, "record", fail_research_agent_request_once)
    runtime = agent.SourceRuntime(
        provider,
        recorder=recorder,
        tool_definitions=[],
        tool_handlers={},
        memory_enabled=False,
    )

    with pytest.raises(TraceWriteError, match="research agent trace unavailable"):
        runtime.run_turn(
            "question",
            task_id="research-task",
            run_metadata={"task_type": "research"},
        )

    assert failed is True
    assert runtime.messages == []


def test_research_failure_stage_trace_failure_escapes_source_runtime(
    tmp_path, monkeypatch
):
    provider = ScriptedProvider([
        light_plan_response(),
        RuntimeError("research provider offline"),
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="stage-trace-failure-run")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff=None,
        metadata={"task_type": "research"},
    )
    original_record = recorder.record
    failed = False

    def fail_attempt_finished_once(event_type, payload, **kwargs):
        nonlocal failed
        if not failed and event_type == "research_attempt_finished":
            failed = True
            raise TraceWriteError("research stage trace unavailable")
        return original_record(event_type, payload, **kwargs)

    monkeypatch.setattr(recorder, "record", fail_attempt_finished_once)
    runtime = agent.SourceRuntime(
        provider,
        recorder=recorder,
        tool_definitions=[],
        tool_handlers={},
        memory_enabled=False,
    )

    with pytest.raises(TraceWriteError, match="research stage trace unavailable"):
        runtime.run_turn(
            "question",
            task_id="research-task",
            run_metadata={"task_type": "research"},
        )

    assert failed is True
    assert runtime.messages == []


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


def test_gate_nul_in_second_authority_is_transactional_and_fails_writing_quota():
    registry = registry_with_two_sources()
    payload = valid_gate_payload(registry)
    payload["authorities"].append({
        "source_id": registry.records[1].source_id,
        "is_authoritative": True,
        "reason": "unsafe\u0000reason",
    })

    decision = parse_research_gate(json.dumps(payload), light_plan(), registry)

    assert decision.passed is False
    assert "control-safe" in " ".join(decision.validation_errors)
    assert not any(item.authoritative for item in registry.records)
    assert "use at least 1 authoritative source" in validate_research_final(
        "Report https://alpha.example/report https://beta.example/data",
        registry,
        light_plan(),
    )


def test_gate_authority_reason_uses_same_nfc_then_length_rule_as_registry():
    registry = registry_with_two_sources()
    payload = valid_gate_payload(registry)
    decomposed_reason = "e\u0301" * 257
    payload["authorities"][0]["reason"] = decomposed_reason

    decision = parse_research_gate(json.dumps(payload), light_plan(), registry)

    assert decision.passed is True
    authoritative = registry.get_by_id(registry.records[0].source_id)
    assert authoritative is not None
    assert authoritative.authority_reason == "é" * 257


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
        "writing_attempt_finished",
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
        "writing_attempt_finished",
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
    writing_finished = [
        row["payload"]
        for row in rows
        if row["event_type"] == "writing_attempt_finished"
    ]
    assert writing_finished == [
        {
            "attempt": 1,
            "repair": False,
            "status": "completed",
            "failure_class": None,
            "failure_message": None,
        },
        {
            "attempt": 2,
            "repair": True,
            "status": "completed",
            "failure_class": None,
            "failure_message": None,
        },
    ]
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
    assert not hasattr(caught.value, "rounds_used")
    assert workflow.consumed_rounds == 2


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
        "writing_attempt_finished",
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
        "writing_attempt_finished",
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
    writing_finished = next(
        row["payload"] for row in phase_rows
        if row["event_type"] == "writing_attempt_finished"
    )
    assert writing_finished == {
        "attempt": 1,
        "repair": False,
        "status": "completed",
        "failure_class": None,
        "failure_message": None,
    }
    json.dumps([row["payload"] for row in phase_rows])


@pytest.mark.parametrize("failed_attempt", (1, 2))
def test_writing_provider_failure_records_bounded_finished_then_reraises(
    tmp_path,
    failed_attempt,
):
    registry = registry_with_two_sources()
    provider_error = RuntimeError("writer offline\n" + "x" * 900)
    responses = [
        light_plan_response(),
        gate_response(registry, covered=True),
    ]
    if failed_attempt == 2:
        responses.append(ModelResponse("unsupported draft"))
    responses.append(provider_error)
    provider = ScriptedProvider(responses)
    recorder = TraceRecorder(tmp_path / "run", run_id=f"writing-{failed_attempt}-failure")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        f"writing-{failed_attempt}-failure",
        "research-task",
        "2025-05-01",
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is provider_error
    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert incomplete is False
    finished = [
        row["payload"]
        for row in rows
        if row["event_type"] == "writing_attempt_finished"
    ]
    if failed_attempt == 2:
        assert finished[0] == {
            "attempt": 1,
            "repair": False,
            "status": "completed",
            "failure_class": None,
            "failure_message": None,
        }
    failed = finished[-1]
    assert set(failed) == {
        "attempt",
        "repair",
        "status",
        "failure_class",
        "failure_message",
    }
    assert failed["attempt"] == failed_attempt
    assert failed["repair"] is (failed_attempt == 2)
    assert failed["status"] == "failed"
    assert failed["failure_class"] == "RuntimeError"
    assert failed["failure_message"].startswith("writer offline x")
    assert "\n" not in failed["failure_message"]
    assert len(failed["failure_message"]) <= 512


def test_writing_trace_error_preserves_identity_without_finished_event(tmp_path):
    registry = registry_with_two_sources()
    trace_error = TraceWriteError("writing provider trace unavailable")
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=True),
        trace_error,
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="writing-trace-failure")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        "writing-trace-failure",
        "research-task",
        "2025-05-01",
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(TraceWriteError) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is trace_error
    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert incomplete is False
    assert not [
        row for row in rows
        if row["event_type"] == "writing_attempt_finished"
    ]


def test_writing_finished_trace_failure_preserves_identity_and_skips_gate(
    tmp_path,
    monkeypatch,
):
    registry = registry_with_two_sources()
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=True),
        valid_light_report(),
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="writing-finished-trace-failure")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        "writing-finished-trace-failure",
        "research-task",
        "2025-05-01",
    )
    trace_error = TraceWriteError("writing finished trace unavailable")
    original_record = recorder.record

    def fail_writing_finished(event_type, payload, **kwargs):
        if event_type == "writing_attempt_finished":
            raise trace_error
        return original_record(event_type, payload, **kwargs)

    monkeypatch.setattr(recorder, "record", fail_writing_finished)
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(TraceWriteError) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is trace_error
    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert incomplete is False
    assert not [row for row in rows if row["event_type"] == "writing_gate"]


@pytest.mark.parametrize("trace_mode", ("preconstructed", "write_failure"))
@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    "provider_diagnostics",
    ("empty", "explicit_cause", "context", "suppressed_context"),
)
def test_provider_error_remains_primary_when_failed_finished_trace_also_fails(
    tmp_path,
    monkeypatch,
    failed_attempt,
    trace_mode,
    provider_diagnostics,
):
    registry = registry_with_two_sources()
    provider_error, origin_traceback = _provider_error_with_origin(
        f"writer {failed_attempt} offline"
    )
    original_cause, original_context, original_suppress = (
        _install_provider_diagnostics(provider_error, provider_diagnostics)
    )
    responses = [
        light_plan_response(),
        gate_response(registry, covered=True),
    ]
    if failed_attempt == 2:
        responses.append(ModelResponse("unsupported draft"))
    responses.append(provider_error)
    provider = ScriptedProvider(responses)
    recorder = TraceRecorder(tmp_path / "run", run_id=f"writing-{failed_attempt}-double")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        f"writing-{failed_attempt}-double",
        "research-task",
        "2025-05-01",
    )
    trace_errors, os_errors, active_exceptions = _install_trace_failures(
        monkeypatch,
        recorder,
        kinds={"writing_failed"},
        mode=trace_mode,
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is provider_error
    _assert_provider_diagnostics(
        caught.value,
        cause=original_cause,
        context=original_context,
        suppress_context=original_suppress,
    )
    assert active_exceptions == {"writing_failed": None}
    if trace_mode == "write_failure":
        assert trace_errors["writing_failed"].__cause__ is (
            os_errors["writing_failed"]
        )
    _assert_acyclic_exception_graph(caught.value)
    assert _traceback_contains(
        BaseException.__getattribute__(caught.value, "__traceback__"),
        origin_traceback,
    )
    writing_notes = _audit_notes(caught.value, "writing_attempt_finished")
    assert len(writing_notes) == 1
    assert "\n" not in writing_notes[0]
    assert len(writing_notes[0]) <= 600


@pytest.mark.parametrize("trace_mode", ("preconstructed", "write_failure"))
@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    "provider_diagnostics",
    ("empty", "explicit_cause", "context", "suppressed_context"),
)
def test_provider_error_uses_event_notes_when_both_audit_writes_fail(
    tmp_path,
    monkeypatch,
    failed_attempt,
    trace_mode,
    provider_diagnostics,
):
    registry = registry_with_two_sources()
    provider_error, origin_traceback = _provider_error_with_origin(
        f"writer {failed_attempt} offline"
    )
    original_cause, original_context, original_suppress = (
        _install_provider_diagnostics(provider_error, provider_diagnostics)
    )
    responses = [
        light_plan_response(),
        gate_response(registry, covered=True),
    ]
    if failed_attempt == 2:
        responses.append(ModelResponse("unsupported draft"))
    responses.append(provider_error)
    provider = ScriptedProvider(responses)
    recorder = TraceRecorder(tmp_path / "run", run_id=f"persistent-{failed_attempt}")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        f"persistent-{failed_attempt}",
        "research-task",
        "2025-05-01",
    )
    trace_errors, os_errors, active_exceptions = _install_trace_failures(
        monkeypatch,
        recorder,
        kinds={"writing_failed", "terminal"},
        mode=trace_mode,
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is provider_error
    _assert_provider_diagnostics(
        caught.value,
        cause=original_cause,
        context=original_context,
        suppress_context=original_suppress,
    )
    assert active_exceptions == {"writing_failed": None, "terminal": None}
    if trace_mode == "write_failure":
        assert trace_errors["writing_failed"].__cause__ is (
            os_errors["writing_failed"]
        )
        assert trace_errors["terminal"].__cause__ is os_errors["terminal"]
    _assert_acyclic_exception_graph(caught.value)
    _assert_acyclic_exception_graph(trace_errors["terminal"])
    assert _traceback_contains(
        BaseException.__getattribute__(caught.value, "__traceback__"),
        origin_traceback,
    )
    writing_notes = _audit_notes(caught.value, "writing_attempt_finished")
    assert len(writing_notes) == 1
    assert "\n" not in writing_notes[0]
    assert len(writing_notes[0]) <= 600
    terminal_notes = _audit_notes(caught.value, "research_workflow_completed")
    assert len(terminal_notes) == 1
    assert "TraceWriteError" in terminal_notes[0]
    assert "\n" not in terminal_notes[0]
    assert len(terminal_notes[0]) <= 600


@pytest.mark.parametrize("trace_mode", ("preconstructed", "write_failure"))
@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    "provider_diagnostics",
    ("empty", "explicit_cause", "context", "suppressed_context"),
)
def test_provider_error_uses_note_when_only_terminal_audit_write_fails(
    tmp_path,
    monkeypatch,
    failed_attempt,
    trace_mode,
    provider_diagnostics,
):
    registry = registry_with_two_sources()
    provider_error, origin_traceback = _provider_error_with_origin(
        f"writer {failed_attempt} offline"
    )
    original_cause, original_context, original_suppress = (
        _install_provider_diagnostics(provider_error, provider_diagnostics)
    )
    responses = [
        light_plan_response(),
        gate_response(registry, covered=True),
    ]
    if failed_attempt == 2:
        responses.append(ModelResponse("unsupported draft"))
    responses.append(provider_error)
    provider = ScriptedProvider(responses)
    recorder = TraceRecorder(
        tmp_path / "run",
        run_id=f"terminal-only-{failed_attempt}-{trace_mode}",
    )
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        f"terminal-only-{failed_attempt}-{trace_mode}",
        "research-task",
        "2025-05-01",
    )
    trace_errors, os_errors, active_exceptions = _install_trace_failures(
        monkeypatch,
        recorder,
        kinds={"terminal"},
        mode=trace_mode,
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(RuntimeError) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is provider_error
    _assert_provider_diagnostics(
        caught.value,
        cause=original_cause,
        context=original_context,
        suppress_context=original_suppress,
    )
    assert active_exceptions == {"terminal": None}
    if trace_mode == "write_failure":
        assert trace_errors["terminal"].__cause__ is os_errors["terminal"]
    _assert_acyclic_exception_graph(caught.value)
    assert _traceback_contains(
        BaseException.__getattribute__(caught.value, "__traceback__"),
        origin_traceback,
    )
    terminal_notes = _audit_notes(caught.value, "research_workflow_completed")
    assert len(terminal_notes) == 1
    if terminal_notes:
        assert "TraceWriteError" in terminal_notes[0]
        assert "\n" not in terminal_notes[0]
        assert len(terminal_notes[0]) <= 600


@pytest.mark.parametrize("mask_mode", ("return_none", "raise"))
@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    ("failure_combination", "failed_kinds"),
    (
        ("failed_only", {"writing_failed"}),
        ("persistent", {"writing_failed", "terminal"}),
        ("terminal_only", {"terminal"}),
    ),
)
def test_masked_diagnostic_descriptors_cannot_hide_provider_slots(
    tmp_path,
    monkeypatch,
    mask_mode,
    failed_attempt,
    failure_combination,
    failed_kinds,
):
    provider_error, origin_traceback = _error_with_origin(
        _MaskedDiagnosticProviderError("writer offline", mask_mode)
    )
    original_cause = LookupError("underlying provider cause")
    original_context = OSError("underlying provider context")
    _set_exception_slot(provider_error, "__cause__", original_cause)
    _set_exception_slot(provider_error, "__context__", original_context)
    _set_exception_slot(provider_error, "__suppress_context__", True)
    expected_visibility = (
        {
            "__cause__": "raised",
            "__context__": "raised",
            "__suppress_context__": "raised",
            "__traceback__": "raised",
        }
        if mask_mode == "raise"
        else {
            "__cause__": None,
            "__context__": None,
            "__suppress_context__": False,
            "__traceback__": None,
        }
    )

    def visible_slots():
        values = {}
        for name in expected_visibility:
            try:
                values[name] = BaseException.__getattribute__(
                    provider_error,
                    name,
                )
            except _HostileProtocolTrap:
                values[name] = "raised"
        return values

    assert visible_slots() == expected_visibility
    workflow, run, registry, trace_errors, active_exceptions = (
        _provider_failure_workflow(
            tmp_path,
            monkeypatch,
            provider_error=provider_error,
            failed_attempt=failed_attempt,
            failed_kinds=failed_kinds,
            run_id=(
                f"masked-{mask_mode}-{failure_combination}-{failed_attempt}"
            ),
        )
    )

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    if caught.value is not provider_error:
        caught_name = type.__getattribute__(type(caught.value), "__name__")
        pytest.fail(
            f"masked descriptor replaced provider with {caught_name}: "
            f"{str(caught.value)}",
            pytrace=False,
        )
    _assert_provider_diagnostics(
        provider_error,
        cause=original_cause,
        context=original_context,
        suppress_context=True,
    )
    assert visible_slots() == expected_visibility
    assert active_exceptions == {kind: None for kind in failed_kinds}
    _assert_acyclic_exception_graph(provider_error)
    assert _traceback_contains(
        _get_exception_slot(provider_error, "__traceback__"),
        origin_traceback,
    )
    for trace_error in trace_errors.values():
        assert provider_error not in _assert_acyclic_exception_graph(trace_error)

    writing_notes = _audit_notes(provider_error, "writing_attempt_finished")
    assert len(writing_notes) == (1 if "writing_failed" in failed_kinds else 0)
    terminal_notes = _audit_notes(provider_error, "research_workflow_completed")
    assert len(terminal_notes) == (1 if "terminal" in failed_kinds else 0)
    for note in [*writing_notes, *terminal_notes]:
        assert "\n" not in note
        assert len(note) <= 600


@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    ("failure_combination", "failed_kinds"),
    (
        ("failed_only", {"writing_failed"}),
        ("persistent", {"writing_failed", "terminal"}),
        ("terminal_only", {"terminal"}),
    ),
)
@pytest.mark.parametrize(
    "graph_kind",
    (
        "cause_provider",
        "context_provider",
        "masked_cause_provider",
        "masked_context_provider",
        "masked_cause_provider_raise",
        "masked_context_provider_raise",
        "self_cycle",
        "two_node_cycle",
        "too_deep",
        "clean",
    ),
)
def test_audit_graph_shape_never_changes_provider_diagnostics(
    tmp_path,
    monkeypatch,
    failed_attempt,
    failure_combination,
    failed_kinds,
    graph_kind,
):
    provider_error, origin_traceback = _provider_error_with_origin(
        "writer offline"
    )
    workflow, run, registry, trace_errors, active_exceptions = (
        _provider_failure_workflow(
            tmp_path,
            monkeypatch,
            provider_error=provider_error,
            failed_attempt=failed_attempt,
            failed_kinds=failed_kinds,
            run_id=f"graph-{graph_kind}-{failure_combination}-{failed_attempt}",
        )
    )
    if not _mask_audit_graph_back_reference(
        trace_errors,
        provider_error,
        graph_kind,
    ):
        for trace_error in trace_errors.values():
            _configure_audit_graph(trace_error, provider_error, graph_kind)

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    if caught.value is not provider_error:
        pytest.fail("audit graph replaced the provider error", pytrace=False)
    _assert_provider_diagnostics(
        provider_error,
        cause=None,
        context=None,
        suppress_context=False,
    )
    assert active_exceptions == {kind: None for kind in failed_kinds}
    _assert_acyclic_exception_graph(provider_error)
    assert _traceback_contains(
        _get_exception_slot(provider_error, "__traceback__"),
        origin_traceback,
    )

    writing_notes = _audit_notes(provider_error, "writing_attempt_finished")
    assert len(writing_notes) == (1 if "writing_failed" in failed_kinds else 0)
    terminal_notes = _audit_notes(provider_error, "research_workflow_completed")
    expected_terminal_notes = int("terminal" in failed_kinds)
    assert len(terminal_notes) == expected_terminal_notes
    for note in [*writing_notes, *terminal_notes]:
        assert "\n" not in note
        assert len(note) <= 600


@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    ("failure_combination", "failed_kinds"),
    (
        ("failed_only", {"writing_failed"}),
        ("persistent", {"writing_failed", "terminal"}),
        ("terminal_only", {"terminal"}),
    ),
)
def test_audit_failures_never_start_cause_attachment_transaction(
    tmp_path,
    monkeypatch,
    failed_attempt,
    failure_combination,
    failed_kinds,
):
    provider_error, origin_traceback = _provider_error_with_origin(
        "writer offline"
    )
    workflow, run, registry, trace_errors, active_exceptions = (
        _provider_failure_workflow(
            tmp_path,
            monkeypatch,
            provider_error=provider_error,
            failed_attempt=failed_attempt,
            failed_kinds=failed_kinds,
            run_id=f"no-attach-{failure_combination}-{failed_attempt}",
        )
    )
    mutation_descriptor = _AttachMutationCauseDescriptor(
        provider_error,
        tuple(trace_errors.values()),
    )
    monkeypatch.setitem(
        research_workflow._EXCEPTION_SLOT_DESCRIPTORS,
        "__cause__",
        mutation_descriptor,
    )

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    if caught.value is not provider_error:
        pytest.fail("post-attach mutation replaced provider error", pytrace=False)
    _assert_provider_diagnostics(
        provider_error,
        cause=None,
        context=None,
        suppress_context=False,
    )
    assert mutation_descriptor.attach_mutations == 0
    assert active_exceptions == {kind: None for kind in failed_kinds}
    _assert_acyclic_exception_graph(provider_error)
    assert _traceback_contains(
        _get_exception_slot(provider_error, "__traceback__"),
        origin_traceback,
    )

    writing_notes = _audit_notes(provider_error, "writing_attempt_finished")
    assert len(writing_notes) == (1 if "writing_failed" in failed_kinds else 0)
    terminal_notes = _audit_notes(provider_error, "research_workflow_completed")
    expected_terminal_notes = int("terminal" in failed_kinds)
    assert len(terminal_notes) == expected_terminal_notes
    for note in [*writing_notes, *terminal_notes]:
        assert "\n" not in note
        assert len(note) <= 600


@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    ("failure_combination", "failed_kinds"),
    (
        ("failed_only", {"writing_failed"}),
        ("persistent", {"writing_failed", "terminal"}),
        ("terminal_only", {"terminal"}),
    ),
)
@pytest.mark.parametrize(
    "graph_kind",
    (
        "cause_provider",
        "self_cycle",
        "too_deep",
        "unreadable",
        "hostile_str",
        "clean",
    ),
)
def test_note_only_audit_does_not_read_provider_slots_or_audit_graph(
    tmp_path,
    monkeypatch,
    failed_attempt,
    failure_combination,
    failed_kinds,
    graph_kind,
):
    provider_error, origin_traceback = _provider_error_with_origin(
        "writer offline"
    )
    workflow, run, registry, trace_errors, active_exceptions = (
        _provider_failure_workflow(
            tmp_path,
            monkeypatch,
            provider_error=provider_error,
            failed_attempt=failed_attempt,
            failed_kinds=failed_kinds,
            run_id=f"note-only-{graph_kind}-{failure_combination}-{failed_attempt}",
        )
    )
    if graph_kind in {"unreadable", "hostile_str"}:
        for kind in tuple(trace_errors):
            trace_errors[kind] = (
                _MaskedAuditTraceError(
                    _trace_failure_message(kind),
                    "raise",
                )
                if graph_kind == "unreadable"
                else _HostileStringTraceError(_trace_failure_message(kind))
            )
    else:
        for trace_error in trace_errors.values():
            _configure_audit_graph(trace_error, provider_error, graph_kind)

    original_read_slot = research_workflow._read_exception_slot
    original_write_slot = research_workflow._write_exception_slot
    forbidden_slot_accesses: list[tuple[str, str]] = []

    def guarded_read_slot(error, name, default):
        if name != "__traceback__":
            forbidden_slot_accesses.append(("read", name))
            raise _HostileProtocolTrap(f"forbidden provider slot read: {name}")
        return original_read_slot(error, name, default)

    def guarded_write_slot(error, name, value):
        if name != "__traceback__":
            forbidden_slot_accesses.append(("write", name))
            raise _HostileProtocolTrap(f"forbidden provider slot write: {name}")
        return original_write_slot(error, name, value)

    def reject_graph_traversal(*args, **kwargs):
        raise _HostileProtocolTrap("audit graph traversal is forbidden")

    monkeypatch.setattr(
        research_workflow,
        "_read_exception_slot",
        guarded_read_slot,
    )
    monkeypatch.setattr(
        research_workflow,
        "_write_exception_slot",
        guarded_write_slot,
    )
    monkeypatch.setattr(
        research_workflow,
        "_exception_graph_is_safe",
        reject_graph_traversal,
        raising=False,
    )

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is provider_error
    assert forbidden_slot_accesses == []
    _assert_provider_diagnostics(
        provider_error,
        cause=None,
        context=None,
        suppress_context=False,
    )
    assert active_exceptions == {kind: None for kind in failed_kinds}
    assert _traceback_contains(
        _get_exception_slot(provider_error, "__traceback__"),
        origin_traceback,
    )
    writing_notes = _audit_notes(provider_error, "writing_attempt_finished")
    assert len(writing_notes) == (1 if "writing_failed" in failed_kinds else 0)
    terminal_notes = _audit_notes(provider_error, "research_workflow_completed")
    assert len(terminal_notes) == (1 if "terminal" in failed_kinds else 0)
    for note in [*writing_notes, *terminal_notes]:
        assert "\n" not in note
        assert len(note) <= 600


@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    ("failure_combination", "failed_kinds"),
    (
        ("failed_only", {"writing_failed"}),
        ("persistent", {"writing_failed", "terminal"}),
        ("terminal_only", {"terminal"}),
    ),
)
def test_explicit_suppress_context_forces_audit_failures_to_notes(
    tmp_path,
    monkeypatch,
    failed_attempt,
    failure_combination,
    failed_kinds,
):
    provider_error, origin_traceback = _provider_error_with_origin(
        "writer offline"
    )
    _set_exception_slot(provider_error, "__suppress_context__", True)
    workflow, run, registry, trace_errors, active_exceptions = (
        _provider_failure_workflow(
            tmp_path,
            monkeypatch,
            provider_error=provider_error,
            failed_attempt=failed_attempt,
            failed_kinds=failed_kinds,
            run_id=f"suppressed-empty-{failure_combination}-{failed_attempt}",
        )
    )

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is provider_error
    _assert_provider_diagnostics(
        provider_error,
        cause=None,
        context=None,
        suppress_context=True,
    )
    assert active_exceptions == {kind: None for kind in failed_kinds}
    _assert_acyclic_exception_graph(provider_error)
    assert _traceback_contains(
        _get_exception_slot(provider_error, "__traceback__"),
        origin_traceback,
    )
    for trace_error in trace_errors.values():
        assert provider_error not in _assert_acyclic_exception_graph(trace_error)
    writing_notes = _audit_notes(provider_error, "writing_attempt_finished")
    assert len(writing_notes) == (1 if "writing_failed" in failed_kinds else 0)
    terminal_notes = _audit_notes(provider_error, "research_workflow_completed")
    assert len(terminal_notes) == (1 if "terminal" in failed_kinds else 0)
    for note in [*writing_notes, *terminal_notes]:
        assert "\n" not in note
        assert len(note) <= 600


@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    ("failure_combination", "failed_kinds"),
    (
        ("failed_only", {"writing_failed"}),
        ("persistent", {"writing_failed", "terminal"}),
        ("terminal_only", {"terminal"}),
    ),
)
def test_hostile_special_setattr_cannot_replace_provider_during_attach(
    tmp_path,
    monkeypatch,
    failed_attempt,
    failure_combination,
    failed_kinds,
):
    provider_error, origin_traceback = _error_with_origin(
        _HostileSpecialSetattrProviderError("writer offline")
    )
    workflow, run, registry, trace_errors, _ = _provider_failure_workflow(
        tmp_path,
        monkeypatch,
        provider_error=provider_error,
        failed_attempt=failed_attempt,
        failed_kinds=failed_kinds,
        run_id=f"setattr-{failure_combination}-{failed_attempt}",
    )

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    if caught.value is not provider_error:
        caught_name = type.__getattribute__(type(caught.value), "__name__")
        pytest.fail(
            f"hostile setattr replaced provider with {caught_name}: "
            f"{str(caught.value)}",
            pytrace=False,
        )
    _assert_provider_diagnostics(
        provider_error,
        cause=None,
        context=None,
        suppress_context=False,
    )
    _assert_acyclic_exception_graph(provider_error)
    assert _traceback_contains(
        _get_exception_slot(provider_error, "__traceback__"),
        origin_traceback,
    )


@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    ("failure_combination", "failed_kinds"),
    (
        ("failed_only", {"writing_failed"}),
        ("persistent", {"writing_failed", "terminal"}),
        ("terminal_only", {"terminal"}),
    ),
)
def test_hostile_provider_protocols_cannot_replace_primary_error(
    tmp_path,
    monkeypatch,
    failed_attempt,
    failure_combination,
    failed_kinds,
):
    registry = registry_with_two_sources()
    provider_error, origin_traceback = _error_with_origin(
        _HostileProviderError("private\nprovider " + "s" * 2_000)
    )
    responses = [
        light_plan_response(),
        gate_response(registry, covered=True),
    ]
    if failed_attempt == 2:
        responses.append(ModelResponse("unsupported draft"))
    responses.append(provider_error)
    provider = ScriptedProvider(responses)
    recorder = TraceRecorder(
        tmp_path / "run",
        run_id=f"hostile-{failure_combination}-{failed_attempt}",
    )
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        f"hostile-{failure_combination}-{failed_attempt}",
        "research-task",
        "2025-05-01",
    )
    trace_errors, _, active_exceptions = _install_trace_failures(
        monkeypatch,
        recorder,
        kinds=failed_kinds,
        mode="preconstructed",
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    if caught.value is not provider_error:
        pytest.fail("hostile protocol replaced the provider exception", pytrace=False)
    _assert_provider_diagnostics(
        provider_error,
        cause=None,
        context=None,
        suppress_context=False,
    )
    assert active_exceptions == {kind: None for kind in failed_kinds}
    _assert_acyclic_exception_graph(provider_error)
    assert _traceback_contains(
        BaseException.__getattribute__(provider_error, "__traceback__"),
        origin_traceback,
    )

    writing_notes = _audit_notes(provider_error, "writing_attempt_finished")
    assert len(writing_notes) == (
        1 if "writing_failed" in failed_kinds else 0
    )
    terminal_notes = _audit_notes(provider_error, "research_workflow_completed")
    assert len(terminal_notes) == (1 if "terminal" in failed_kinds else 0)
    for note in [*writing_notes, *terminal_notes]:
        assert "\n" not in note
        assert len(note) <= 600

    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert incomplete is False
    failed_events = [
        row["payload"]
        for row in rows
        if row["event_type"] == "writing_attempt_finished"
        and row["payload"].get("status") == "failed"
    ]
    if failure_combination == "terminal_only":
        assert len(failed_events) == 1
        payload = failed_events[0]
        assert isinstance(payload["failure_class"], str)
        assert len(payload["failure_class"]) <= 128
        assert isinstance(payload["failure_message"], str)
        assert len(payload["failure_message"]) <= 512
        assert "private" not in payload["failure_message"]
        assert "\n" not in payload["failure_message"]


@pytest.mark.parametrize("failed_attempt", (1, 2))
@pytest.mark.parametrize(
    ("failure_combination", "failed_kinds"),
    (
        ("failed_only", {"writing_failed"}),
        ("persistent", {"writing_failed", "terminal"}),
        ("terminal_only", {"terminal"}),
    ),
)
def test_failed_nonvirtual_add_note_does_not_replace_provider(
    tmp_path,
    monkeypatch,
    failed_attempt,
    failure_combination,
    failed_kinds,
):
    provider_error, origin_traceback = _error_with_origin(
        _HostileNoteProviderError("writer offline")
    )
    workflow, run, registry, trace_errors, active_exceptions = (
        _provider_failure_workflow(
            tmp_path,
            monkeypatch,
            provider_error=provider_error,
            failed_attempt=failed_attempt,
            failed_kinds=failed_kinds,
            run_id=f"note-failure-{failure_combination}-{failed_attempt}",
        )
    )

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is provider_error
    _assert_provider_diagnostics(
        provider_error,
        cause=None,
        context=None,
        suppress_context=False,
    )
    assert active_exceptions == {kind: None for kind in failed_kinds}
    assert _traceback_contains(
        _get_exception_slot(provider_error, "__traceback__"),
        origin_traceback,
    )
    assert _audit_notes(provider_error, "writing_attempt_finished") == []
    assert _audit_notes(provider_error, "research_workflow_completed") == []
    for trace_error in trace_errors.values():
        assert provider_error not in _assert_acyclic_exception_graph(trace_error)


@pytest.mark.parametrize("interrupt_type", (KeyboardInterrupt, SystemExit))
def test_non_trace_base_exception_from_audit_recorder_is_not_swallowed(
    tmp_path,
    monkeypatch,
    interrupt_type,
):
    registry = registry_with_two_sources()
    provider_error = RuntimeError("writer offline")
    provider = ScriptedProvider([
        light_plan_response(),
        gate_response(registry, covered=True),
        provider_error,
    ])
    recorder = TraceRecorder(tmp_path / "run", run_id="audit-base-exception")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        "audit-base-exception",
        "research-task",
        "2025-05-01",
    )
    interrupt = interrupt_type("audit interrupted")
    original_record = recorder.record

    def interrupt_audit(event_type, payload, **kwargs):
        if _trace_failure_kind(event_type, payload) == "writing_failed":
            raise interrupt
        return original_record(event_type, payload, **kwargs)

    monkeypatch.setattr(recorder, "record", interrupt_audit)
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )

    with bind_run_context(run), pytest.raises(BaseException) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is interrupt


@pytest.mark.parametrize("failed_attempt", (1, 2))
def test_direct_writing_trace_error_is_rethrown_unchanged(
    tmp_path,
    monkeypatch,
    failed_attempt,
):
    registry = registry_with_two_sources()
    responses = [
        light_plan_response(),
        gate_response(registry, covered=True),
    ]
    if failed_attempt == 2:
        responses.append(ModelResponse("unsupported draft"))
    provider = ScriptedProvider(responses)
    recorder = TraceRecorder(tmp_path / "run", run_id=f"writing-{failed_attempt}-direct")
    recorder.start_run(
        task_id="research-task",
        question="question",
        cutoff="2025-05-01",
        metadata={"task_type": "research"},
    )
    run = RunContext(
        recorder,
        f"writing-{failed_attempt}-direct",
        "research-task",
        "2025-05-01",
    )
    workflow = ResearchWorkflow(
        provider,
        lambda prompt, max_rounds, evidence_registry: AgentLoopOutcome(
            "completed", "notes", rounds_used=1
        ),
        run_context=run,
    )
    trace_error = TraceWriteError(f"writing attempt {failed_attempt} trace failure")

    def fail_directly(*args, **kwargs):
        raise trace_error

    monkeypatch.setattr(
        workflow,
        "write" if failed_attempt == 1 else "rewrite",
        fail_directly,
    )

    with bind_run_context(run), pytest.raises(TraceWriteError) as caught:
        workflow.run("question", "2025-05-01", registry=registry)

    assert caught.value is trace_error
    rows, incomplete = read_trace_lines(recorder.trajectory_path)
    assert incomplete is False
    assert not [
        row for row in rows
        if row["event_type"] == "writing_attempt_finished"
        and row["payload"]["attempt"] == failed_attempt
    ]


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
