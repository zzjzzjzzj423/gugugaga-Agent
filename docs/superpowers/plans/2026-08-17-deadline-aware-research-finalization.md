# Deadline-Aware Financial Research Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every recoverable financial-research run returns and publishes either a substantive report or a deterministic `INSUFFICIENT_EVIDENCE` report before the external 1,800-second timeout.

**Architecture:** Add a run-scoped `ResearchController` that owns irreversible phases, absolute monotonic deadlines, evidence progress, stop decisions, and cache state while the existing `agent_loop()` remains the executor. Provider and foreground tool calls receive bounded timeouts; research exits always flow through a tool-free finalizer, one tool-free repair, and then deterministic fallback. The controller APIs deliberately match future workflow-node boundaries without adding LangGraph now.

**Tech Stack:** Python 3.11+, dataclasses, enums, `time.monotonic`, OpenAI Python SDK, existing S20-shaped provider/content blocks, pytest 8+, existing trace/manifest recorder.

## Global Constraints

- Treat `simple_cc` as a financial-research Agent; the deadline policy is enabled by default for all `agent_loop()` runs.
- External timeout is 1,800 seconds; internal completion deadline is 1,750 seconds.
- Default phase offsets from loop start are: research 1,450 seconds, finalization 1,580 seconds, repair/fallback 1,690 seconds, hard completion 1,750 seconds.
- Finalization is one attempt with no tools, thinking disabled where supported, and at most 120 seconds.
- Repair is one attempt with no tools, thinking disabled where supported, and at most 90 seconds.
- A transition out of `RESEARCHING` is irreversible.
- Deterministic fallback must not infer financial facts from arbitrary tool output.
- Fetch/search cache state is scoped to one `agent_loop()` invocation and is never persisted across tasks.
- Reuse `simple_cc.evidence.canonicalize_url()`; do not introduce a second URL canonicalizer.
- Preserve all unrelated dirty-worktree changes. Each commit must stage only the files listed in its task.
- Do not add LangGraph, Supervisor/Researcher agents, embedding-based query similarity, or cross-run disk caching.
- Keep existing point-in-time cutoff enforcement, evidence registration, citation linkage, and trace redaction behavior intact.

## File Map

- Create `simple_cc/research_control.py`: pure phases, deadlines, authority policy, progress, cache, stop policy, fallback formatter, and serializable summary.
- Create `simple_cc/tool_execution.py`: deadline-bounded foreground handler execution using a daemon worker and copied context.
- Modify `simple_cc/config.py`: environment-backed research-control defaults.
- Modify `simple_cc/models.py`: deadline/attempt options on `ChatProvider.create()`.
- Modify `simple_cc/provider.py`: request timeout and retry admission against an absolute deadline.
- Modify `simple_cc/recovery.py`: bound outer retry/backoff by the same absolute deadline.
- Modify `simple_cc/telemetry.py`: forward and trace deadline/timeout/attempt options.
- Modify `simple_cc/prompts.py`: host-owned finalization and repair prompts.
- Modify `simple_cc/agent.py`: controller integration, finalization, repair, fallback, cache, progress, and tool timeout execution.
- Modify `simple_cc/benchmark.py`: pass an optional test/runtime controller configuration into `SourceRuntime`.
- Modify `eval/run_task.py`: publish controller summary into the terminal manifest.
- Modify `tests/fakes.py`: accept and record the expanded provider call contract.
- Create `tests/test_research_control.py`: pure controller, cache, coverage, and fallback tests.
- Create `tests/test_research_finalization.py`: Agent-level finalization, repair, timeout, cache, and stagnation tests.
- Create `tests/test_tool_execution.py`: foreground timeout wrapper tests.
- Modify `tests/test_config_provider.py`, `tests/test_telemetry.py`, `tests/test_evidence_trace.py`, `tests/test_benchmark_worker.py`, and `tests/test_benchmark_runtime.py` at their existing boundaries.
- Create `tests/fixtures/deadline_worker.py`: subprocess fixture for a tiny-deadline supervisor/worker integration test.

---

### Task 1: Build the Phase and Deadline Controller Foundation

**Files:**
- Create: `simple_cc/research_control.py`
- Modify: `simple_cc/config.py:21-52`
- Create: `tests/test_research_control.py`

**Interfaces:**
- Consumes: `Callable[[], float]` clock, defaulting to `time.monotonic`.
- Produces: `ResearchPhase`, `ResearchAction`, `ResearchControlConfig`, `StopDecision`, `DeadlinePolicy`, `AuthorityPolicy`, and `ResearchController`.
- Produces exact controller methods used later: `admit_call(requested_timeout: float) -> float`, `tool_timeout(tool_name: str) -> float`, `transition(target: ResearchPhase, reason: str) -> None`, `deadline_for(phase: ResearchPhase | None = None) -> float`, and `summary() -> dict[str, Any]`.

- [ ] **Step 1: Add failing tests for default deadlines, clamping, and irreversible transitions**

```python
from dataclasses import replace

import pytest

from simple_cc.research_control import (
    ResearchAction,
    ResearchControlConfig,
    ResearchController,
    ResearchPhase,
)


class FakeClock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_controller_uses_absolute_phase_deadlines_and_clamps_calls():
    clock = FakeClock()
    controller = ResearchController(ResearchControlConfig(), clock=clock)

    assert controller.phase is ResearchPhase.RESEARCHING
    assert controller.deadline_for() == 1550.0
    assert controller.admit_call(120.0) == 120.0
    assert controller.tool_timeout("web_search") == 15.0
    assert controller.tool_timeout("web_fetch") == 30.0
    assert controller.tool_timeout("pdf_fetch") == 60.0
    assert controller.tool_timeout("read_file") == 120.0

    clock.advance(1445.0)
    assert controller.admit_call(120.0) == 5.0

    clock.advance(5.0)
    assert controller.next_action() is ResearchAction.FINALIZE_WITHOUT_TOOLS


def test_controller_cannot_return_to_research_after_finalizing():
    controller = ResearchController(ResearchControlConfig(), clock=FakeClock())
    controller.transition(ResearchPhase.FINALIZING, "deadline")

    with pytest.raises(ValueError, match="irreversible"):
        controller.transition(ResearchPhase.RESEARCHING, "try_again")


def test_config_rejects_non_monotonic_offsets():
    with pytest.raises(ValueError, match="research.*finalization.*repair.*hard"):
        ResearchControlConfig(
            research_deadline_seconds=100,
            finalization_deadline_seconds=90,
            repair_deadline_seconds=120,
            hard_deadline_seconds=150,
        )
```

- [ ] **Step 2: Run the new controller tests and verify import failure**

Run: `python -m pytest tests/test_research_control.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'simple_cc.research_control'`.

- [ ] **Step 3: Add environment-backed default constants to `config.py`**

```python
RESEARCH_DEADLINE_SECONDS = float(
    os.getenv("SIMPLE_CC_RESEARCH_DEADLINE_SECONDS", "1450")
)
FINALIZATION_DEADLINE_SECONDS = float(
    os.getenv("SIMPLE_CC_FINALIZATION_DEADLINE_SECONDS", "1580")
)
REPAIR_DEADLINE_SECONDS = float(
    os.getenv("SIMPLE_CC_REPAIR_DEADLINE_SECONDS", "1690")
)
HARD_DEADLINE_SECONDS = float(
    os.getenv("SIMPLE_CC_HARD_DEADLINE_SECONDS", "1750")
)
FINALIZATION_TIMEOUT_SECONDS = float(
    os.getenv("SIMPLE_CC_FINALIZATION_TIMEOUT_SECONDS", "120")
)
REPAIR_TIMEOUT_SECONDS = float(
    os.getenv("SIMPLE_CC_REPAIR_TIMEOUT_SECONDS", "90")
)
RESEARCH_LLM_TIMEOUT_SECONDS = float(
    os.getenv("SIMPLE_CC_RESEARCH_LLM_TIMEOUT_SECONDS", "120")
)
TOOL_TIMEOUT_SECONDS = float(
    os.getenv("SIMPLE_CC_TOOL_TIMEOUT_SECONDS", "120")
)
WEB_SEARCH_TIMEOUT_SECONDS = float(
    os.getenv("SIMPLE_CC_WEB_SEARCH_TIMEOUT_SECONDS", "15")
)
WEB_FETCH_TIMEOUT_SECONDS = float(
    os.getenv("SIMPLE_CC_WEB_FETCH_TIMEOUT_SECONDS", "30")
)
PDF_FETCH_TIMEOUT_SECONDS = float(
    os.getenv("SIMPLE_CC_PDF_FETCH_TIMEOUT_SECONDS", "60")
)
```

- [ ] **Step 4: Implement the minimal phase/deadline types and controller**

Use these exact public shapes in `simple_cc/research_control.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from . import config


class ResearchPhase(str, Enum):
    RESEARCHING = "researching"
    FINALIZING = "finalizing"
    REPAIRING = "repairing"
    DONE = "done"


class ResearchAction(str, Enum):
    CONTINUE_RESEARCH = "continue_research"
    FINALIZE_WITHOUT_TOOLS = "finalize_without_tools"
    REPAIR_WITHOUT_TOOLS = "repair_without_tools"
    RETURN_FALLBACK = "return_fallback"
    DONE = "done"


@dataclass(frozen=True)
class ResearchControlConfig:
    research_deadline_seconds: float = config.RESEARCH_DEADLINE_SECONDS
    finalization_deadline_seconds: float = config.FINALIZATION_DEADLINE_SECONDS
    repair_deadline_seconds: float = config.REPAIR_DEADLINE_SECONDS
    hard_deadline_seconds: float = config.HARD_DEADLINE_SECONDS
    research_llm_timeout_seconds: float = config.RESEARCH_LLM_TIMEOUT_SECONDS
    finalization_timeout_seconds: float = config.FINALIZATION_TIMEOUT_SECONDS
    repair_timeout_seconds: float = config.REPAIR_TIMEOUT_SECONDS
    tool_timeout_seconds: float = config.TOOL_TIMEOUT_SECONDS
    web_search_timeout_seconds: float = config.WEB_SEARCH_TIMEOUT_SECONDS
    web_fetch_timeout_seconds: float = config.WEB_FETCH_TIMEOUT_SECONDS
    pdf_fetch_timeout_seconds: float = config.PDF_FETCH_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        offsets = (
            self.research_deadline_seconds,
            self.finalization_deadline_seconds,
            self.repair_deadline_seconds,
            self.hard_deadline_seconds,
        )
        if not all(value > 0 for value in offsets) or not all(
            left < right for left, right in zip(offsets, offsets[1:])
        ):
            raise ValueError(
                "research, finalization, repair, and hard deadlines must be positive and monotonic"
            )


@dataclass(frozen=True)
class StopDecision:
    action: ResearchAction
    reason: str


class DeadlinePolicy:
    def __init__(
        self,
        settings: ResearchControlConfig,
        *,
        started_at: float,
        clock: Callable[[], float],
    ) -> None:
        self.settings = settings
        self.started_at = started_at
        self.clock = clock

    def deadline_for(self, phase: ResearchPhase) -> float:
        offsets = {
            ResearchPhase.RESEARCHING: self.settings.research_deadline_seconds,
            ResearchPhase.FINALIZING: self.settings.finalization_deadline_seconds,
            ResearchPhase.REPAIRING: self.settings.repair_deadline_seconds,
            ResearchPhase.DONE: self.settings.hard_deadline_seconds,
        }
        return self.started_at + offsets[phase]

    def remaining(self, phase: ResearchPhase) -> float:
        return max(0.0, self.deadline_for(phase) - self.clock())

    def clamp_timeout(self, phase: ResearchPhase, requested: float) -> float:
        return max(0.0, min(float(requested), self.remaining(phase)))
```

Implement `ResearchController` with `phase`, `stop_reason`, `phase_started_at`, a list of phase-duration records, and the public methods in the interface block. `next_action()` returns finalization at the research deadline, fallback at the repair deadline, and `DONE` only after `mark_done()`.

- [ ] **Step 5: Run the controller tests and confirm they pass**

Run: `python -m pytest tests/test_research_control.py -v`

Expected: all Task 1 tests pass.

- [ ] **Step 6: Run configuration regression tests**

Run: `python -m pytest tests/test_config_provider.py -v`

Expected: existing settings/provider tests pass unchanged.

- [ ] **Step 7: Commit the controller foundation**

```bash
git add simple_cc/research_control.py simple_cc/config.py tests/test_research_control.py
git commit -m "feat: add research phase and deadline controller"
```

---

### Task 2: Add Evidence Progress, Authority Coverage, Stop Policy, and Run Cache

**Files:**
- Modify: `simple_cc/research_control.py`
- Modify: `simple_cc/config.py:21-70`
- Modify: `tests/test_research_control.py`

**Interfaces:**
- Consumes: `evidence.canonicalize_url(url: str) -> str`.
- Produces: `AuthorityPolicy.is_authoritative(url: str) -> bool`, `ResearchProgress`, `CacheKey`, `CacheEntry`, `FetchCache`, `StopPolicy.evaluate(progress: ResearchProgress) -> str | None`.
- Produces controller methods used by the Agent: `record_llm_call()`, `record_research_round(progressed: bool)`, `record_tool_result(...) -> bool`, `lookup_tool_result(...) -> str | None`, `store_tool_result(...) -> None`, and `coverage_sufficient() -> bool`.

- [ ] **Step 1: Add failing tests for authority, coverage, stagnation, and caching**

Append concrete cases to `tests/test_research_control.py`:

```python
import json

from simple_cc.research_control import AuthorityPolicy, FetchCache, ResearchProgress, StopPolicy


def test_authority_policy_recognizes_regulators_and_official_ir_paths():
    policy = AuthorityPolicy.default()

    assert policy.is_authoritative("https://www.sec.gov/Archives/report.htm")
    assert policy.is_authoritative("https://company.example/investors/results")
    assert not policy.is_authoritative("https://finance-blog.example/opinion")


def test_coverage_requires_two_domains_and_one_authority():
    progress = ResearchProgress()
    progress.register_fetched_source("https://www.sec.gov/report")
    progress.register_fetched_source("https://example.com/copy")

    assert progress.coverage_sufficient() is True


def test_stop_policy_finalizes_after_two_stagnant_rounds():
    progress = ResearchProgress(consecutive_stagnant_rounds=2)
    reason = StopPolicy.default().evaluate(progress)

    assert reason == "stagnation_rounds"


def test_fetch_cache_reuses_success_and_allows_only_one_transient_retry():
    cache = FetchCache()
    arguments = {"url": "https://example.com/report?a=1&b=2", "cutoff": "2025-05-01"}
    success = json.dumps({"ok": True, "operation": "fetch", "url": arguments["url"]})

    cache.store("web_fetch", arguments, success)
    assert cache.lookup("web_fetch", arguments) == success

    transient_args = {"url": "https://example.com/down", "cutoff": None}
    transient = json.dumps({
        "ok": False,
        "error": {"code": "fetch_failed", "message": "timeout"},
    })
    cache.store("web_fetch", transient_args, transient)
    assert cache.lookup("web_fetch", transient_args) is None
    assert cache.allow_transient_attempt("web_fetch", transient_args) is True
    cache.store("web_fetch", transient_args, transient)
    assert cache.allow_transient_attempt("web_fetch", transient_args) is False


def test_exact_normalized_search_query_is_cached():
    cache = FetchCache()
    args_a = {"query": "  USD JPY   options ", "cutoff": "2025-05-01"}
    args_b = {"query": "usd jpy options", "cutoff": "2025-05-01"}
    output = json.dumps({"ok": True, "operation": "search", "results": []})

    cache.store("web_search", args_a, output)
    assert cache.lookup("web_search", args_b) == output
```

- [ ] **Step 2: Run the focused tests and verify missing types/methods**

Run: `python -m pytest tests/test_research_control.py -v`

Expected: new tests fail because progress, cache, and authority APIs do not exist.

- [ ] **Step 3: Add stop/coverage defaults to `config.py`**

```python
STAGNANT_ROUND_LIMIT = int(os.getenv("SIMPLE_CC_STAGNANT_ROUND_LIMIT", "2"))
STAGNANT_TOOL_CALL_LIMIT = int(
    os.getenv("SIMPLE_CC_STAGNANT_TOOL_CALL_LIMIT", "8")
)
MIN_FETCHED_SOURCES = int(os.getenv("SIMPLE_CC_MIN_FETCHED_SOURCES", "2"))
MIN_SOURCE_DOMAINS = int(os.getenv("SIMPLE_CC_MIN_SOURCE_DOMAINS", "2"))
MIN_AUTHORITATIVE_SOURCES = int(
    os.getenv("SIMPLE_CC_MIN_AUTHORITATIVE_SOURCES", "1")
)
DEFAULT_AUTHORITY_DOMAINS = (
    "sec.gov",
    "federalreserve.gov",
    "newyorkfed.org",
    "treasury.gov",
    "cftc.gov",
    "fdic.gov",
    "ecb.europa.eu",
    "bankofengland.co.uk",
    "bis.org",
    "imf.org",
    "worldbank.org",
    "oecd.org",
    "esma.europa.eu",
)
```

Add matching fields to `ResearchControlConfig` so tests can override every threshold.

- [ ] **Step 4: Implement authority and progress tracking**

Use `urllib.parse.urlsplit`, domain suffix checks, and these exact IR path markers:

```python
OFFICIAL_PATH_MARKERS = (
    "/investor",
    "/investors",
    "/investor-relations",
    "/ir/",
    "/newsroom",
    "/press-release",
    "/regulatory-filings",
)
```

Define `ResearchProgress` with `settings: ResearchControlConfig = field(default_factory=ResearchControlConfig)` and `authority_policy: AuthorityPolicy = field(default_factory=AuthorityPolicy.default)` so its no-argument coverage method and source registration use one explicit policy snapshot. `ResearchProgress.register_fetched_source()` must canonicalize the URL, add its lower-case hostname, classify authority once, and return `True` only if one of the evidence sets grew. `record_research_round(False)` increments `consecutive_stagnant_rounds`; meaningful progress resets both stagnant counters. `coverage_sufficient()` uses the three configured minimums.

- [ ] **Step 5: Implement deterministic cache classification and normalized search keys**

Use these cache types and classifications:

```python
DETERMINISTIC_FETCH_ERRORS = {
    "unsafe_url",
    "post_cutoff",
    "published_after_cutoff",
    "date_conflict",
    "page_out_of_range",
    "unsupported_content_type",
    "response_too_large",
}


@dataclass(frozen=True)
class CacheKey:
    tool_name: str
    identity: str
    cutoff: str | None


@dataclass(frozen=True)
class CacheEntry:
    output: str
    classification: str  # success | deterministic_failure
```

Normalize search queries with lower-casing and whitespace collapse. Parse JSON outputs; cache `ok: true` and deterministic error codes. Track transient attempts separately and permit at most the initial attempt plus one retry.

- [ ] **Step 6: Implement stop evaluation and compose it into `ResearchController`**

Evaluation order must be deterministic:

```python
def evaluate(self, progress: ResearchProgress) -> str | None:
    if progress.repeated_query:
        return "repeated_query"
    if progress.consecutive_stagnant_tool_calls >= self.settings.stagnant_tool_call_limit:
        return "stagnation_tool_calls"
    if progress.consecutive_stagnant_rounds >= self.settings.stagnant_round_limit:
        return "stagnation_rounds"
    if progress.coverage_sufficient() and progress.rounds_since_coverage >= 1:
        return "coverage_sufficient"
    return None
```

Store the chosen reason once; later calls may not replace it.

- [ ] **Step 7: Run policy tests**

Run: `python -m pytest tests/test_research_control.py -v`

Expected: all deadline, cache, coverage, authority, and stop-policy tests pass.

- [ ] **Step 8: Commit progress and cache policy**

```bash
git add simple_cc/research_control.py simple_cc/config.py tests/test_research_control.py
git commit -m "feat: track research progress and cache fetches"
```

---

### Task 3: Make Provider and Recovery Retries Deadline-Aware

**Files:**
- Modify: `simple_cc/models.py:43-53`
- Modify: `simple_cc/provider.py:14-24,188-279`
- Modify: `simple_cc/recovery.py:11-54`
- Modify: `simple_cc/telemetry.py:84-206`
- Modify: `tests/fakes.py`
- Modify: `tests/test_config_provider.py`
- Modify: `tests/test_telemetry.py`

**Interfaces:**
- Extends `ChatProvider.create()` with keyword-only `timeout_seconds: float | None = None`, `max_attempts: int | None = None`, and `deadline_monotonic: float | None = None`.
- Produces `ProviderDeadlineExceeded(TimeoutError)`.
- Extends `with_retry()` with `deadline_monotonic`, injectable `clock`/`sleep`, and `max_attempts`.
- `TracingProvider` and `ScriptedProvider` must forward/record all new options without altering default behavior.

- [ ] **Step 1: Add failing provider tests for per-request timeout and retry admission**

Add to `tests/test_config_provider.py`:

```python
def test_provider_clamps_request_timeout_to_absolute_deadline(tmp_path, monkeypatch):
    clock = iter([100.0, 100.0]).__next__
    response = SimpleNamespace(
        id="req-1",
        usage=None,
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="ok", tool_calls=[]),
        )],
    )
    seen = []

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            seen.append(kwargs["timeout"])
            return response

    provider = SiliconFlowProvider(settings(tmp_path), client=Client(), clock=clock)
    provider.create(
        [], "", [], 10,
        timeout_seconds=120,
        max_attempts=1,
        deadline_monotonic=130.0,
    )

    assert seen == [30.0]


def test_provider_does_not_retry_after_deadline(tmp_path, monkeypatch):
    clock = FakeClock(100.0)
    client = FailingRateLimitedClient(on_call=lambda: clock.advance(20.0))
    provider = SiliconFlowProvider(settings(tmp_path), client=client, clock=clock, sleep=lambda _: None)

    with pytest.raises(ProviderDeadlineExceeded):
        provider.create(
            [], "", [], 10,
            timeout_seconds=20,
            max_attempts=4,
            deadline_monotonic=120.0,
        )

    assert client.calls == 1
```

Define the small `FakeClock`, `settings(tmp_path)`, and failing client locally in the test file; do not depend on Task 1 test modules.

- [ ] **Step 2: Add failing tracing/fake-provider forwarding tests**

In `tests/test_telemetry.py`, assert the delegate receives:

```python
provider.create(
    [], "system", [],
    timeout_seconds=30,
    max_attempts=1,
    deadline_monotonic=140.0,
)

assert delegate.kwargs == {
    "timeout_seconds": 30,
    "max_attempts": 1,
    "deadline_monotonic": 140.0,
}
```

In `tests/fakes.py`, record these keys on every `ScriptedProvider.requests` entry so later Agent tests can inspect them.

- [ ] **Step 3: Run focused provider and telemetry tests and verify signature failures**

Run: `python -m pytest tests/test_config_provider.py tests/test_telemetry.py -v`

Expected: new calls fail with unexpected keyword arguments or missing `ProviderDeadlineExceeded`.

- [ ] **Step 4: Extend the provider protocol and concrete provider**

Use this exact protocol tail:

```python
        *,
        enable_thinking: bool | None = None,
        timeout_seconds: float | None = None,
        max_attempts: int | None = None,
        deadline_monotonic: float | None = None,
```

Add optional `clock=time.monotonic` and `sleep=time.sleep` constructor dependencies to `SiliconFlowProvider`. Default attempt count remains four when `max_attempts` is `None`. Before every attempt:

```python
remaining = (
    float("inf")
    if deadline_monotonic is None
    else deadline_monotonic - self.clock()
)
if remaining <= 0:
    raise ProviderDeadlineExceeded("provider deadline exhausted")
request_timeout = min(timeout_seconds or 120.0, remaining)
```

Pass `timeout=request_timeout` to `chat.completions.create()`. Clamp backoff sleep to remaining time and never start a retry at or beyond the deadline. Preserve the actual attempt count in `ProviderRequestError`.

- [ ] **Step 5: Bound the outer recovery loop**

Change `with_retry` to:

```python
def with_retry(
    fn,
    state: RecoveryState,
    *,
    deadline_monotonic: float | None = None,
    max_attempts: int | None = None,
    clock=time.monotonic,
    sleep=time.sleep,
):
```

Default to `config.MAX_RETRIES`; check the deadline before each attempt and before sleeping. Raise `ProviderDeadlineExceeded` rather than sleeping through the deadline.

- [ ] **Step 6: Forward options through tracing and test fakes**

Include the three values in the traced request artifact and `llm_request_started` payload. Forward them to delegates in both traced and untraced paths. Do not include an infinite timestamp; use `None` when no deadline was supplied.

- [ ] **Step 7: Run provider, telemetry, and content-block tests**

Run: `python -m pytest tests/test_config_provider.py tests/test_telemetry.py tests/test_provider_content_blocks.py -v`

Expected: all pass, including the existing default four-attempt test.

- [ ] **Step 8: Commit deadline-aware provider behavior**

```bash
git add simple_cc/models.py simple_cc/provider.py simple_cc/recovery.py simple_cc/telemetry.py tests/fakes.py tests/test_config_provider.py tests/test_telemetry.py
git commit -m "feat: bound provider retries by phase deadlines"
```

---

### Task 4: Add Tool-Free Finalization, One Repair, and Deterministic Fallback

**Files:**
- Modify: `simple_cc/prompts.py`
- Modify: `simple_cc/agent.py:86-92,94-253,256-563,872-877`
- Modify: `simple_cc/benchmark.py:31-38,116-144`
- Modify: `eval/run_task.py:142-203`
- Create: `tests/test_research_finalization.py`
- Modify: `tests/test_evidence_trace.py:254-288`

**Interfaces:**
- Extends `AgentLoopOutcome` with `research_control: dict[str, Any] | None = None`.
- Extends `agent_loop()`, `SourceRuntime`, and `execute_task()` with optional `research_control_config: ResearchControlConfig | None`; `agent_loop()`/`SourceRuntime` also accept `clock: Callable[[], float]`, and the loop boundary accepts explicit `original_question: str | None`.
- Produces `build_research_finalization_prompt(...) -> str`, `build_research_repair_prompt(...) -> str`, `_run_tool_free_model(...) -> ProviderResponse`, and `_fallback_outcome(...) -> AgentLoopOutcome`.

- [ ] **Step 1: Add failing tests for deadline finalization and empty `max_rounds` removal**

Create `tests/test_research_finalization.py` with a local fake clock and these cases:

```python
def test_deadline_forces_one_tool_free_finalization(monkeypatch):
    clock = FakeClock()
    provider = ScriptedProvider([
        ModelResponse("", [ToolCall("s1", "web_search", {"query": "rates"})], "tool_calls"),
        ModelResponse("draft with no valid evidence"),
        ModelResponse("repair still invalid"),
    ])
    settings = ResearchControlConfig(
        research_deadline_seconds=1,
        finalization_deadline_seconds=20,
        repair_deadline_seconds=30,
        hard_deadline_seconds=40,
    )
    handlers = {"web_search": lambda **_: clock.advance(1) or '{"ok": true, "results": []}'}

    outcome = agent_loop(
        [{"role": "user", "content": "Research rates"}],
        {},
        provider=provider,
        tools=[{"name": "web_search", "description": "search", "input_schema": {}}],
        handlers=handlers,
        max_rounds=10,
        memory_enabled=False,
        run_context=FakeRunContext(),
        research_control_config=settings,
        clock=clock,
        original_question="Research rates",
    )

    assert provider.requests[1]["tools"] == []
    assert provider.requests[1]["enable_thinking"] is False
    assert outcome.status == "completed"


def test_max_rounds_flows_to_tool_free_finalizer():
    provider = ScriptedProvider([
        ModelResponse("", [ToolCall("n1", "noop", {})], "tool_calls"),
        ModelResponse("unsupported draft"),
        ModelResponse("unsupported repair"),
    ])

    outcome = agent_loop(
        [{"role": "user", "content": "Research"}],
        {},
        provider=provider,
        tools=[{"name": "noop", "description": "noop", "input_schema": {}}],
        handlers={"noop": lambda: "ok"},
        max_rounds=1,
        memory_enabled=False,
        run_context=FakeRunContext(),
    )

    assert outcome.status == "completed"
    assert outcome.final_text.startswith("INSUFFICIENT_EVIDENCE")
    assert provider.requests[-1]["tools"] == []
```

Use a real lightweight `TraceRecorder`/`RunContext` fixture rather than an incomplete stub if `record_research_evidence()` requires recorder methods.

- [ ] **Step 2: Add failing tests for provider outage, tool-call output, empty output, and one repair**

Add cases asserting all of these return completed fallback:

```python
@pytest.mark.parametrize("final_response", [
    RuntimeError("provider unavailable"),
    ModelResponse(""),
    ModelResponse("", [ToolCall("again", "web_search", {"query": "more"})], "tool_calls"),
])
def test_finalization_failure_returns_deterministic_fallback(final_response):
    ...
    assert outcome.status == "completed"
    assert outcome.final_text.startswith("INSUFFICIENT_EVIDENCE")
```

Update the existing evidence repair assertion so both finalization and repair requests have `tools == []` and the provider receives exactly two writer calls after research stops.

- [ ] **Step 3: Run focused tests and verify current failures**

Run: `python -m pytest tests/test_research_finalization.py tests/test_evidence_trace.py -v`

Expected: deadline/config keywords are unsupported, `max_rounds` remains empty, and current repair requests still expose tools.

- [ ] **Step 4: Add host-owned finalization and repair prompts**

Add functions in `prompts.py` that accept only serializable host state:

```python
def build_research_finalization_prompt(
    *,
    original_question: str,
    research_context: str,
    registered_sources: dict[str, str],
    control_summary: dict[str, Any],
) -> str:
    ...


def build_research_repair_prompt(
    *,
    original_question: str,
    draft: str,
    validation_errors: list[str],
    registered_sources: dict[str, str],
) -> str:
    ...
```

Both prompts must state that no tools are available, only registered URLs may be cited, unsupported claims must be removed or qualified, and `INSUFFICIENT_EVIDENCE` is valid. The repair prompt must not instruct the model to fetch more evidence.

- [ ] **Step 5: Extend `call_llm()` for bounded tool-free calls**

Add keyword-only options and forward them through both retry layers:

```python
def call_llm(
    messages,
    context,
    tools,
    state,
    max_tokens,
    provider=None,
    *,
    enable_thinking=None,
    timeout_seconds=None,
    max_attempts=None,
    deadline_monotonic=None,
):
```

For finalization/repair, use a fresh `RecoveryState`, `tools=[]`, `enable_thinking=False`, `max_attempts=1`, and the controller's phase deadline.

- [ ] **Step 6: Add finalization/fallback helpers and route all recoverable exits through them**

Implement one helper that performs exactly one writer call and rejects empty text or any `tool_use`. Implement one repair call only after `validate_research_final()` errors. Use `deterministic_fallback()` from `research_control.py` for any writer/repair error.

Replace the current final repair `continue` path and final `max_rounds` return. Provider exceptions, exhausted maximum-token recovery, and context recovery failure should set a stop reason and call the same finalization pipeline instead of returning `failed` while the process remains usable.

Capture `original_question` before any compaction. Store controller state outside `messages`.

- [ ] **Step 7: Thread optional controller config through runtime, worker, and benchmark construction**

Add `research_control_config` and `clock` to `SourceRuntime.__init__`; pass them to every `agent_loop()` call. Add `research_control_config: ResearchControlConfig | None = None` to `BenchmarkOptions` and pass it through `build_benchmark_runtime()`. Add the same keyword-only option to `execute_task()` and use it when constructing `BenchmarkOptions`. This is the injection path used by deterministic small-deadline worker tests; production calls omit it and receive the environment-backed defaults.

- [ ] **Step 8: Run finalization, evidence, Agent-loop, and runtime tests**

Run: `python -m pytest tests/test_research_finalization.py tests/test_evidence_trace.py tests/test_agent_loop.py tests/test_agent_loop_source.py tests/test_benchmark_runtime.py -v`

Expected: all pass; no finalization/repair request contains tools; `max_rounds` returns completed text.

- [ ] **Step 9: Commit the finalization state machine**

```bash
git add simple_cc/prompts.py simple_cc/agent.py simple_cc/benchmark.py eval/run_task.py tests/test_research_finalization.py tests/test_evidence_trace.py
git commit -m "feat: force tool-free research finalization"
```

---

### Task 5: Enforce Foreground Tool Timeouts and Integrate Cache/Progress Stops

**Files:**
- Create: `simple_cc/tool_execution.py`
- Modify: `simple_cc/agent.py:565-871`
- Create: `tests/test_tool_execution.py`
- Modify: `tests/test_research_finalization.py`
- Modify: `tests/test_web_research.py`
- Modify: `tests/test_pdf_research.py`

**Interfaces:**
- Produces `ToolCallTimeout(TimeoutError)` and `call_tool_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any`.
- Consumes controller methods from Tasks 1–2 and existing `call_tool_handler()`.
- Cache/progress integration applies only to `web_search`, `web_fetch`, and `pdf_fetch`; deadline admission applies to every foreground handler.

- [ ] **Step 1: Add failing timeout-wrapper tests**

Create `tests/test_tool_execution.py`:

```python
import contextvars
import threading

import pytest

from simple_cc.tool_execution import ToolCallTimeout, call_tool_with_timeout


def test_tool_wrapper_returns_fast_result():
    assert call_tool_with_timeout(lambda: "ok", 0.1) == "ok"


def test_tool_wrapper_times_out_without_blocking_process_exit():
    release = threading.Event()
    with pytest.raises(ToolCallTimeout, match="0.01"):
        call_tool_with_timeout(lambda: release.wait(1), 0.01)
    release.set()


def test_tool_wrapper_propagates_contextvars():
    marker = contextvars.ContextVar("marker", default="missing")
    marker.set("captured")
    assert call_tool_with_timeout(marker.get, 0.1) == "captured"
```

- [ ] **Step 2: Add failing Agent tests for cache hits and stagnation stops**

Add to `tests/test_research_finalization.py`:

```python
def test_repeated_canonical_fetch_invokes_handler_once_and_finalizes():
    calls = []
    provider = ScriptedProvider([
        tool_response("f1", "web_fetch", {"url": "https://EXAMPLE.com/report#one"}),
        tool_response("f2", "web_fetch", {"url": "https://example.com/report"}),
        ModelResponse("unsupported final"),
        ModelResponse("unsupported repair"),
    ])

    outcome = run_research_agent(
        provider,
        handlers={"web_fetch": lambda **args: calls.append(args) or successful_fetch(args["url"])},
    )

    assert len(calls) == 1
    assert outcome.research_control["cache_hits"] == 1
    assert provider.requests[-1]["tools"] == []


def test_two_stagnant_rounds_transition_to_finalization():
    ...
    assert outcome.research_control["stop_reason"] == "stagnation_rounds"


def test_two_domains_one_authority_then_one_stagnant_round_finalize():
    ...
    assert outcome.research_control["coverage_sufficient"] is True
    assert outcome.research_control["stop_reason"] == "coverage_sufficient"
```

Use successful fetch JSON containing the final URL so the same parser used in production can update progress.

- [ ] **Step 3: Run timeout/cache tests and verify failures**

Run: `python -m pytest tests/test_tool_execution.py tests/test_research_finalization.py -v`

Expected: the timeout module is missing and repeated fetches call the handler twice.

- [ ] **Step 4: Implement a daemon, context-preserving foreground timeout wrapper**

Use `contextvars.copy_context()`, `queue.Queue(maxsize=1)`, and `threading.Thread(daemon=True)`. Join only for `timeout_seconds`; if still alive, raise `ToolCallTimeout`. Re-raise handler exceptions in the caller thread. Reject non-positive timeouts before starting the thread.

The timed-out daemon may finish later, so do not use this wrapper to claim transactional cancellation. It exists to ensure a stuck handler cannot prevent phase transition or process exit. Existing handler-specific socket/subprocess timeouts remain defense in depth.

- [ ] **Step 5: Add tool admission and timeout execution to `agent_loop()`**

Before `call_tool_handler()`:

```python
allowed_timeout = controller.admit_call(
    controller.tool_timeout(block.name)
)
if allowed_timeout <= 0:
    controller.transition(ResearchPhase.FINALIZING, "tool_budget_exhausted")
    break

output = call_tool_with_timeout(
    lambda: call_tool_handler(handler, arguments, block.name, capture=capture),
    allowed_timeout,
)
```

On `ToolCallTimeout`, return a structured tool error to the history, record `research_call_rejected`/`tool_error`, and move to finalization if the phase deadline is exhausted.

- [ ] **Step 6: Integrate cache before handler execution and progress after results**

For research tools:

1. Normalize/prepare cutoff arguments first.
2. Run existing pre-tool Hook and permission approval checks on every request, including requests that may hit cache.
3. Ask `FetchCache.lookup()` after approval but before handler execution.
4. On hit, use cached output, increment cache telemetry, and do not call `record_research_evidence()` again.
5. On miss, execute the handler once, store its output classification, then call existing evidence registration.
6. Parse successful search results into candidate URLs.
7. Compare `registered_sources` before/after the tool to decide whether evidence progressed.
8. At the end of the LLM/tool round call `controller.record_research_round(progressed)` and consult `next_action()` before another model call.

When a transient fetch has already used its single retry, return a structured `transient_retry_exhausted` result without network I/O.

- [ ] **Step 7: Preserve Web/PDF native timeout regressions**

Add assertions to existing Web/PDF tests that their internal connection timeouts remain at or below the Agent ceilings. Do not expose hidden timeout fields in public tool schemas in this task; the foreground wrapper is the host-level hard bound.

- [ ] **Step 8: Run tool, Web, PDF, evidence, and finalization tests**

Run: `python -m pytest tests/test_tool_execution.py tests/test_research_finalization.py tests/test_web_research.py tests/test_pdf_research.py tests/test_evidence_trace.py -v`

Expected: all pass; repeated canonical fetch calls execute once; timeout paths still return completed reports.

- [ ] **Step 9: Commit tool bounds and convergence integration**

```bash
git add simple_cc/tool_execution.py simple_cc/agent.py tests/test_tool_execution.py tests/test_research_finalization.py tests/test_web_research.py tests/test_pdf_research.py
git commit -m "feat: bound research tools and stop stalled searches"
```

---

### Task 6: Persist Research-Control Telemetry and Manifest Summary

**Files:**
- Modify: `simple_cc/agent.py:86-92,161-229,282-296,573-835`
- Modify: `simple_cc/telemetry.py:84-206`
- Modify: `eval/run_task.py:142-248`
- Modify: `tests/test_telemetry.py`
- Modify: `tests/test_benchmark_worker.py:58-143,189-340`

**Interfaces:**
- Consumes: `ResearchController.summary() -> dict[str, Any]`.
- Produces: `AgentLoopOutcome.research_control` and terminal manifest key `research_control`.
- Adds trace events: `research_phase_transition`, `research_stop_decision`, `research_call_rejected`, `tool_cache_hit`, `research_progress_updated`, `finalization_started`, `finalization_failed`, `repair_started`, and `fallback_generated`.

- [ ] **Step 1: Add failing worker assertions for completed provider outage and manifest summary**

Change the existing provider-failure test in `tests/test_benchmark_worker.py` to the new reliability contract:

```python
def test_execute_task_provider_failure_publishes_fallback(tmp_path):
    ...
    assert exit_code == 0
    assert manifest["status"] == "completed"
    assert manifest["research_control"]["fallback_used"] is True
    assert manifest["research_control"]["stop_reason"] == "provider_error"
    assert (run_dir / "final_answer.txt").read_text(
        encoding="utf-8"
    ).startswith("INSUFFICIENT_EVIDENCE")
```

Add a successful tiny-deadline case asserting these manifest keys:

```python
assert set(manifest["research_control"]) >= {
    "phase",
    "stop_reason",
    "elapsed_seconds",
    "phase_durations",
    "llm_calls",
    "research_tool_calls",
    "search_calls",
    "fetch_calls",
    "cache_hits",
    "stagnant_rounds",
    "stagnant_tool_calls",
    "fetched_sources",
    "independent_domains",
    "authoritative_sources",
    "coverage_sufficient",
    "finalization_attempts",
    "repair_attempts",
    "fallback_used",
}
```

- [ ] **Step 2: Add failing trace-event assertions**

Read `trajectory.jsonl` and assert an ordered terminal subsequence:

```python
event_types = [row["event_type"] for row in rows]
assert "research_stop_decision" in event_types
assert "research_phase_transition" in event_types
assert "finalization_started" in event_types
assert "fallback_generated" in event_types
assert event_types.index("research_stop_decision") < event_types.index("finalization_started")
```

- [ ] **Step 3: Run telemetry/worker tests and verify missing summary**

Run: `python -m pytest tests/test_telemetry.py tests/test_benchmark_worker.py -v`

Expected: provider failure is still marked failed and the manifest has no `research_control` key.

- [ ] **Step 4: Centralize controller event recording in `agent.py`**

Add a helper:

```python
def record_research_control_event(
    run_context: RunContext | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if run_context is not None:
        run_context.recorder.record(
            event_type,
            payload,
            agent_id=run_context.agent_id,
        )
```

Record transitions exactly once, cache hits without duplicate evidence events, progress after each research round, and finalization/repair/fallback attempts with elapsed time and stop reason. Never put raw secrets or unredacted page content in these new payloads.

- [ ] **Step 5: Attach the serializable summary to every outcome**

Populate `AgentLoopOutcome.research_control` for substantive completion, fallback completion, and unrecoverable failure. `SourceRuntime` keeps recording `final_answer` from the final text and leaves the summary attached to `last_outcome`.

- [ ] **Step 6: Finalize the manifest with controller details**

Change the completed call to:

```python
recorder.finalize(
    "completed",
    {"research_control": outcome.research_control or {}},
)
```

For non-completed system failures, include the same key beside failure fields when available. Do not change atomic staging/publication order.

- [ ] **Step 7: Run telemetry, worker, trace-validation, and metric tests**

Run: `python -m pytest tests/test_telemetry.py tests/test_benchmark_worker.py tests/test_trace.py tests/test_evidence_trace.py tests/test_eval_metrics.py -v`

Expected: all pass; completed manifests contain the summary and trace validation accepts the new event types.

- [ ] **Step 8: Commit telemetry and manifest reporting**

```bash
git add simple_cc/agent.py simple_cc/telemetry.py eval/run_task.py tests/test_telemetry.py tests/test_benchmark_worker.py
git commit -m "feat: report research finalization telemetry"
```

---

### Task 7: Prove Worker Completion Before Supervisor Timeout and Run Regressions

**Files:**
- Create: `tests/fixtures/deadline_worker.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_benchmark_runtime.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: public `execute_task()`, `TaskInput`, `ResearchControlConfig`, and `ScriptedProvider` contracts.
- Produces: a subprocess-level test showing `final_answer.txt` and terminal completed manifest exist before the runner's external timeout.
- Documents environment variables, phase defaults, fallback semantics, and evaluation commands.

- [ ] **Step 1: Add a subprocess fixture that deliberately never volunteers a final answer**

Create `tests/fixtures/deadline_worker.py` that accepts `--task-input`, `--run-dir`, and `--workspace`, loads `TaskInput`, then calls `execute_task()` with:

```python
class EndlessResearchProvider(ScriptedProvider):
    def create(self, *args, **kwargs):
        tools = kwargs["tools"] if "tools" in kwargs else args[2]
        self.requests.append({"tools": list(tools)})
        if not self.requests[-1]["tools"]:
            raise RuntimeError("writer unavailable")
        return ProviderResponse(
            content=[ToolUseBlock(
                id=f"search-{len(self.requests)}",
                name="web_search",
                input={"query": f"query {len(self.requests)}"},
            )],
            stop_reason="tool_use",
        )
```

Use a tiny `ResearchControlConfig` with offsets `0.05`, `0.10`, `0.15`, and `0.30` seconds and pass it through the Task 4 `execute_task(..., research_control_config=settings)` keyword. Monkeypatch benchmark tool tables inside the fixture to a fast `web_search` handler returning empty successful JSON. The finalizer then fails and deterministic fallback must publish before 0.30 seconds plus process overhead.

- [ ] **Step 2: Add a failing runner integration test using the fixture command**

In `tests/test_benchmark_runner.py`, monkeypatch `_worker_command()` for one assignment to invoke the fixture with `sys.executable`. Run `_run_assignment(..., timeout_seconds=2.0)` and assert:

```python
assert result.manifest["status"] == "completed"
assert result.manifest["research_control"]["fallback_used"] is True
assert result.manifest["research_control"]["elapsed_seconds"] < 1.0
assert (result.run_dir / "final_answer.txt").exists()
assert (result.run_dir / "final_answer.txt").read_text(
    encoding="utf-8"
).startswith("INSUFFICIENT_EVIDENCE")
```

- [ ] **Step 3: Run the subprocess integration test and verify it fails before implementation is fully wired**

Run: `python -m pytest tests/test_benchmark_runner.py -k deadline_worker -v`

Expected: failure identifies any missing config propagation, finalizer fallback, manifest summary, or fixture command wiring.

- [ ] **Step 4: Complete only the integration wiring exposed by the failing test**

Keep the production runner command unchanged. The test-only command monkeypatch must still exercise the real `execute_task()`, `SourceRuntime`, `agent_loop()`, atomic final answer publication, and manifest finalization paths. Do not replace `_run_assignment()` with an in-process fake.

- [ ] **Step 5: Document the runtime contract and tuning controls**

Add a README section listing:

```text
SIMPLE_CC_RESEARCH_DEADLINE_SECONDS=1450
SIMPLE_CC_FINALIZATION_DEADLINE_SECONDS=1580
SIMPLE_CC_REPAIR_DEADLINE_SECONDS=1690
SIMPLE_CC_HARD_DEADLINE_SECONDS=1750
SIMPLE_CC_FINALIZATION_TIMEOUT_SECONDS=120
SIMPLE_CC_REPAIR_TIMEOUT_SECONDS=90
SIMPLE_CC_STAGNANT_ROUND_LIMIT=2
SIMPLE_CC_STAGNANT_TOOL_CALL_LIMIT=8
```

Explain that model finalization/repair are tool-free, fallback is considered completed, and disk/bootstrap failure remains failed.

- [ ] **Step 6: Run the complete focused feature suite**

Run:

```bash
python -m pytest tests/test_research_control.py tests/test_tool_execution.py tests/test_research_finalization.py tests/test_config_provider.py tests/test_telemetry.py tests/test_evidence_trace.py tests/test_benchmark_worker.py tests/test_benchmark_runtime.py tests/test_benchmark_runner.py -v
```

Expected: all focused tests pass.

- [ ] **Step 7: Run the full regression suite**

Run: `python -m pytest -q`

Expected: entire existing suite passes with no failures.

- [ ] **Step 8: Run a one-task live smoke evaluation under the real 1,800-second supervisor timeout**

Run:

```bash
python -m eval.run_benchmark --dataset eval/data/financegym_20.jsonl --output-dir eval/runs/qwen36-deadline-smoke --workers 1 --limit 1 --timeout-seconds 1800
```

Expected: one terminal `completed` manifest, one `final_answer.txt`, internal elapsed time below 1,750 seconds, and no tool request after `finalization_started`.

- [ ] **Step 9: Run the 20-task acceptance batch and compare quality/completion separately**

Run:

```bash
python -m eval.run_benchmark --dataset eval/data/financegym_20.jsonl --output-dir eval/runs/qwen36-deadline-financegym20 --workers 2 --timeout-seconds 1800
```

Expected acceptance counts:

```text
completed = 20/20
timed_out = 0/20
final answer files = 20/20
internal elapsed time per task < 1750 seconds
tool calls after finalization_started = 0
repair attempts per task <= 1
```

Report substantive answers, fallback answers, registered sources, domain counts, authoritative source counts, stop reasons, cache hits, and result-layer score as separate metrics; do not treat fallback completion as proof of answer quality.

- [ ] **Step 10: Commit integration proof and documentation**

```bash
git add tests/fixtures/deadline_worker.py tests/test_benchmark_runner.py tests/test_benchmark_runtime.py README.md
git commit -m "test: prove research completes before supervisor timeout"
```

## Completion Review

Before declaring implementation complete:

- Confirm `git diff --check` is clean.
- Confirm the full pytest command passed in the current worktree.
- Inspect one completed substantive trajectory and one fallback trajectory.
- Verify every `llm_request_started` after `finalization_started` has an empty tool schema.
- Verify no `tool_requested` event occurs after `finalization_started`.
- Verify provider and outer recovery attempt counts respect the same phase deadline.
- Verify cache hits do not duplicate source-registration events.
- Verify `max_rounds`, provider outage, finalization timeout, and repair failure all publish a non-empty `final_answer.txt`.
- Compare the 20-task acceptance counts with the previous fixed-compaction run and report quality metrics separately from completion metrics.
