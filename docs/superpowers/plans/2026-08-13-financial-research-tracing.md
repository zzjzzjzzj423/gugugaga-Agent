# Financial Research Agent Evaluation Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an append-only, auditable evaluation trace and a process-isolated benchmark runner in which every FinanceGym task executes in a fresh SimpleCC process and task-exclusive workspace.

**Architecture:** A storage-only `TraceRecorder` owns JSONL durability, manifests, redaction, and immutable artifacts. Context-bound telemetry wraps every provider call and tool execution without relying on mutable `messages`; the benchmark layer launches one `eval/run_task.py` interpreter per task, keeps evaluation files outside the agent workspace, and lets `eval/run_benchmark.py` coordinate bounded concurrency, timeouts, retries, and resume.

**Tech Stack:** Python 3.11+, standard library (`contextvars`, `dataclasses`, `hashlib`, `json`, `subprocess`, `threading`, `uuid`), existing OpenAI-compatible provider, pytest 8+, existing `ddgs`, `trafilatura`, and `pdfplumber` integrations.

## Global Constraints

- Preserve the active `SourceRuntime` and module-level `agent_loop()` path; do not migrate benchmark execution to the compatibility `AgentRuntime` class.
- Every benchmark task gets a new operating-system process, a new `run_id`, and an empty task-exclusive `agent_workspace`.
- `trajectory.jsonl`, artifacts, task input, reference answers, rubrics, and judge outputs stay outside `agent_workspace`.
- Benchmark defaults are memory disabled, cron disabled, team disabled, subagents disabled, and no interactive approval callback; under the current `PermissionPolicy`, this denies `bash`.
- Context compaction may mutate inference history but may never mutate or reconstruct prior trace events.
- Trace events contain observable messages, calls, results, and decisions only; never request or store hidden chain-of-thought.
- Secret redaction happens before serialization. Missing provider usage is `null`, never zero.
- Required benchmark tracing is fail-closed: recorder failure produces `trace_invalid`, not a scoreable answer.
- PIT mode remains exactly `non_strict_live_web`; live search must never be described as strict historical replay.
- Preserve the already implemented public tool name `pdf_fetch`; derived metrics classify it as a document fetch rather than introducing a second `document_fetch` tool.
- Keep all existing CLI behavior and existing tests passing.
- Do not overwrite or clean unrelated dirty-worktree changes.

## File Structure

### New production files

- `simple_cc/trace.py`: event schema, `TraceRecorder`, manifest lifecycle, artifact storage, redaction, active run/model/tool contexts.
- `simple_cc/telemetry.py`: `TracingProvider`, model-call classification, traced tool execution, and exact model-visible request/result artifacts.
- `simple_cc/evidence.py`: cutoff injection, research-result parsing, source registration, citation linkage, and PIT evidence events.
- `simple_cc/benchmark.py`: benchmark-only runtime factory, feature flags, tool filtering, clean-workspace validation, and shutdown checks.
- `simple_cc/eval_metrics.py`: strict trajectory reader and derived process/efficiency/risk metrics.
- `eval/__init__.py`: makes benchmark entry points importable in tests and with `python -m`.
- `eval/run_task.py`: single-task worker; one invocation executes exactly one task.
- `eval/run_benchmark.py`: parent coordinator for selection, concurrency, timeout, retry, resume, and process-tree cleanup.

### New test files

- `tests/test_trace.py`: recorder, redaction, artifact, durability, and terminal-state tests.
- `tests/test_telemetry.py`: provider and tool telemetry tests.
- `tests/test_evidence_trace.py`: cutoff and source-provenance tests.
- `tests/test_benchmark_runtime.py`: feature-flag and workspace-boundary tests.
- `tests/test_benchmark_worker.py`: single-task worker lifecycle and failure tests.
- `tests/test_benchmark_runner.py`: fresh-process, concurrency, timeout, retry, and resume tests.
- `tests/test_eval_metrics.py`: event validation and metric derivation tests.
- `tests/fixtures/isolation_probe.py`: network-free child process used to prove PID and workspace isolation.

### Existing files modified

- `simple_cc/provider.py`: retain usage, provider request ID, and attempt count.
- `simple_cc/agent.py`: inject provider/tools/runtime options, return explicit loop outcome, bind trace lifecycle, and record permissions/compaction/final answer.
- `simple_cc/context.py`: expose compaction metadata without changing normal context behavior.
- `simple_cc/tools.py`: accept internal execution context while preserving handler return strings.
- `simple_cc/web_research.py`: capture raw fetched content as a trace artifact when tracing is active.
- `simple_cc/pdf_research.py`: capture downloaded PDF bytes and extracted page text as trace artifacts.
- `simple_cc/background.py`: propagate `contextvars` into background workers and expose quiescence.
- `simple_cc/subagents.py`: assign child `agent_id` and model-call kind.
- `simple_cc/memory.py`: label memory retrieval/extraction/consolidation model calls.
- `simple_cc/teams.py`: label teammate calls when an experiment explicitly enables teams.
- `simple_cc/__main__.py`: pass explicit runtime dependencies while preserving interactive defaults.
- `tests/fakes.py`: construct provider responses with optional usage/request metadata.
- `tests/test_config_provider.py`: verify provider metadata extraction.
- `tests/test_agent_loop_source.py`: verify traced loop outcomes and compatibility.
- `README.md`: document benchmark commands, directory layout, status meanings, and PIT limitation.

---

### Task 1: Durable trace storage, artifacts, manifests, and redaction

**Files:**
- Create: `simple_cc/trace.py`
- Create: `tests/test_trace.py`

**Interfaces:**
- Produces: `ArtifactRef`, `TraceRecorder`, `TraceWriteError`, `RunContext`, `bind_run_context()`, `current_run_context()`, `redact_value()`, and `supervisor_finalize_manifest()`.
- Consumes: only Python standard-library types and filesystem paths.

- [ ] **Step 1: Write failing recorder and redaction tests**

```python
def test_record_is_immediately_readable_and_monotonic(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    recorder.start_run(task_id="task-1", question="q", cutoff="2025-01-01", metadata={})
    recorder.record("tool_started", {"name": "web_search"})
    rows = [json.loads(line) for line in recorder.trajectory_path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[1]["event_type"] == "tool_started"


def test_artifacts_are_deduplicated_by_sha256(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1")
    first = recorder.store_artifact("same", media_type="text/plain", source="test", suffix=".txt")
    second = recorder.store_artifact("same", media_type="text/plain", source="test", suffix=".txt")
    assert first == second
    assert len(list((tmp_path / "run" / "artifacts").iterdir())) == 1


def test_redaction_happens_before_disk_write(tmp_path):
    recorder = TraceRecorder(tmp_path / "run", run_id="run-1", secrets=["secret-value"])
    recorder.start_run(task_id="task-1", question="q", cutoff=None, metadata={})
    recorder.record("tool_requested", {"api_key": "secret-value", "text": "x secret-value y"})
    disk = recorder.trajectory_path.read_text()
    assert "secret-value" not in disk
    assert "[REDACTED:api_key]" in disk
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_trace.py -v`

Expected: collection fails because `simple_cc.trace` does not exist.

- [ ] **Step 3: Implement the storage interfaces**

Implement these exact public shapes in `simple_cc/trace.py`:

```python
TERMINAL_STATUSES = {
    "completed", "failed", "max_rounds", "cancelled",
    "timed_out", "worker_crashed", "trace_invalid",
}


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    media_type: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunContext:
    recorder: "TraceRecorder"
    run_id: str
    task_id: str
    cutoff: str | None
    agent_id: str = "root"
    parent_span_id: str | None = None

    def child(self, agent_id: str, parent_span_id: str | None = None) -> "RunContext":
        return replace(self, agent_id=agent_id, parent_span_id=parent_span_id)


class TraceWriteError(RuntimeError):
    pass
```

Use a `ContextVar[RunContext | None]` and a token-resetting context manager:

```python
_ACTIVE_RUN: ContextVar[RunContext | None] = ContextVar("simple_cc_active_run", default=None)


@contextmanager
def bind_run_context(context: RunContext):
    token = _ACTIVE_RUN.set(context)
    try:
        yield context
    finally:
        _ACTIVE_RUN.reset(token)


def current_run_context() -> RunContext | None:
    return _ACTIVE_RUN.get()
```

`TraceRecorder.record()` must assign `sequence` while holding one `RLock`, redact recursively, serialize one compact JSON object, append one line, call `flush()`, and then `os.fsync()`. `store_artifact()` must hash bytes first, write through a temporary file plus `os.replace()`, use a path relative to `run_dir`, and reuse an existing same-hash artifact. `finalize()` must atomically replace `manifest.json`, reject unknown statuses, and refuse a second conflicting terminal state except the supervisor transition from `running` to `timed_out`, `worker_crashed`, or `trace_invalid`.

Every event envelope must contain `schema_version`, `run_id`, `task_id`, `sequence`, `event_type`, `timestamp_utc`, monotonic `elapsed_ms`, nullable `span_id`, nullable `parent_span_id`, `agent_id`, and `payload`. `start_run()` writes `manifest.json` first and then the `run_started` event. The manifest must retain the task identity, benchmark/task type, question, cutoff, PIT mode, provider/model, worker PID, agent workspace, isolation flags, maximum rounds, prompt/tool-schema hashes, Git commit/dirty state, start/end times, retry linkage, and status. Fields unknown at start are explicit `null` values.

Use exact redaction markers: `[REDACTED:api_key]`, `[REDACTED:authorization]`, `[REDACTED:cookie]`, `[REDACTED:password]`, and `[REDACTED:configured_secret]`. Match sensitive keys case-insensitively after removing `-` and `_`; replace every configured secret substring in free text.

Add `supervisor_finalize_manifest(run_dir, status, task_metadata, details)`. It runs only after the child process has exited, atomically changes an existing `running` manifest to `timed_out`, `worker_crashed`, or `trace_invalid`, and creates a minimal terminal manifest from parent-owned task metadata if the child died before creating one. It never accepts `completed`.

- [ ] **Step 4: Add crash-line and terminal-state tests**

Append tests proving that an incomplete final JSONL line leaves all earlier lines parseable, `completed` cannot become `failed`, and a forced write exception raises `TraceWriteError` before an answer can be accepted.

- [ ] **Step 5: Run the trace tests**

Run: `python -m pytest tests/test_trace.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the trace core**

```bash
git add simple_cc/trace.py tests/test_trace.py
git commit -m "feat: add durable evaluation trace storage"
```

### Task 2: Provider usage metadata and model-call telemetry

**Files:**
- Modify: `simple_cc/provider.py:18-36,167-213`
- Modify: `tests/fakes.py`
- Modify: `tests/test_config_provider.py`
- Create: `simple_cc/telemetry.py`
- Create: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `RunContext`, `ArtifactRef`, `current_run_context()` from Task 1.
- Produces: `ProviderUsage`, extended `ProviderResponse`, `TracingProvider`, and `model_call_scope()`.

- [ ] **Step 1: Write failing provider metadata tests**

Construct an SDK response whose `usage` has `prompt_tokens=11`, `completion_tokens=7`, `total_tokens=18` and whose response ID is `req-1`. Assert:

```python
assert result.usage == ProviderUsage(11, 7, 18)
assert result.request_id == "req-1"
assert result.attempts == 1
```

Add a second test with no `usage` attribute and assert all three usage fields are `None`, not zero.

- [ ] **Step 2: Run the provider tests and verify failure**

Run: `python -m pytest tests/test_config_provider.py -k usage -v`

Expected: failure because `ProviderResponse` does not retain usage.

- [ ] **Step 3: Extend the provider response without breaking content blocks**

Add:

```python
@dataclass(frozen=True)
class ProviderUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class ProviderResponse:
    content: list[TextBlock | ToolUseBlock]
    stop_reason: str
    usage: ProviderUsage = field(default_factory=lambda: ProviderUsage(None, None, None))
    request_id: str | None = None
    attempts: int = 1
```

In `SiliconFlowProvider.create()`, normalize `_value(response, "usage")`, preserve `_value(response, "id")`, and set `attempts=attempt + 1`. Update `ScriptedProvider` so queued `ModelResponse` objects use the explicit unknown-usage default.

- [ ] **Step 4: Write failing `TracingProvider` tests**

Bind a `RunContext`, call a scripted delegate once, and assert `llm_request_started` and `llm_response` share a span, the response contains token usage and latency, and the stored request artifact contains the exact redacted `system`, `messages`, tool schemas, model, and `max_tokens`. Add an exception test that asserts `llm_error` records the safe exception class, attempts, and latency.

- [ ] **Step 5: Implement context-bound provider telemetry**

In `simple_cc/telemetry.py`, define:

```python
@dataclass(frozen=True)
class ModelCallContext:
    kind: str = "agent"
    parent_span_id: str | None = None


@contextmanager
def model_call_scope(kind: str, parent_span_id: str | None = None):
    token = _MODEL_CALL.set(ModelCallContext(kind, parent_span_id))
    try:
        yield
    finally:
        _MODEL_CALL.reset(token)


class TracingProvider:
    def __init__(self, delegate: ChatProvider):
        self.delegate = delegate

    def create(self, messages, system, tools, max_tokens=8192, model=None):
        run = current_run_context()
        if run is None:
            return self.delegate.create(messages, system, tools, max_tokens, model)
        span_id = f"llm_{uuid.uuid4().hex}"
        request = {"system": system, "messages": messages, "tools": tools,
                   "max_tokens": max_tokens, "model": model}
        request_ref = run.recorder.store_artifact(
            request, media_type="application/json", source="llm_request", suffix=".json"
        )
        run.recorder.record("llm_request_started", {
            "call_kind": _MODEL_CALL.get().kind,
            "model": model,
            "max_tokens": max_tokens,
            "request_artifact": request_ref.as_dict(),
        }, span_id=span_id, parent_span_id=_MODEL_CALL.get().parent_span_id,
           agent_id=run.agent_id)
        started = time.monotonic()
        try:
            response = self.delegate.create(messages, system, tools, max_tokens, model)
        except Exception as error:
            run.recorder.record("llm_error", {
                "exception_class": type(error).__name__,
                "message": str(error),
                "attempts": getattr(error, "attempts", None),
                "latency_ms": (time.monotonic() - started) * 1000,
            }, span_id=span_id, agent_id=run.agent_id)
            raise
        run.recorder.record("llm_response", {
            "stop_reason": response.stop_reason,
            "usage": asdict(response.usage),
            "request_id": response.request_id,
            "attempts": response.attempts,
            "latency_ms": (time.monotonic() - started) * 1000,
            "content": response.content,
        }, span_id=span_id, agent_id=run.agent_id)
        return response
```

Convert content blocks to JSON-safe dictionaries before calling `record()`. Store large response content as an artifact and keep only an artifact reference plus a bounded preview in the event.

- [ ] **Step 6: Run provider and telemetry tests**

Run: `python -m pytest tests/test_config_provider.py tests/test_provider_content_blocks.py tests/test_telemetry.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit provider telemetry**

```bash
git add simple_cc/provider.py simple_cc/telemetry.py tests/fakes.py tests/test_config_provider.py tests/test_telemetry.py
git commit -m "feat: trace model usage and latency"
```

### Task 3: SourceRuntime lifecycle, explicit outcomes, and compaction evidence

**Files:**
- Modify: `simple_cc/agent.py:65-380`
- Modify: `simple_cc/context.py:276-344`
- Modify: `simple_cc/__main__.py:179-206,343-420`
- Modify: `tests/test_agent_loop_source.py`
- Modify: `tests/test_context_recovery.py`

**Interfaces:**
- Consumes: `TraceRecorder`, `RunContext`, `bind_run_context()`, `TracingProvider`, and `model_call_scope()`.
- Produces: `AgentLoopOutcome`; extended `SourceRuntime.run_turn()`; dependency-injected `agent_loop()`.

- [ ] **Step 1: Write failing lifecycle tests**

Add tests for a normal text response, provider exception, forced compaction, and tool-loop exhaustion. Assert exact terminal statuses and required events:

```python
recorder.start_run(
    task_id="t1", question="question", cutoff="2025-01-01", metadata={}
)
answer = runtime.run_turn("question", task_id="t1", cutoff="2025-01-01")
assert answer == "answer"
assert runtime.last_outcome.status == "completed"
assert manifest()["status"] == "running"
assert event_types()[-1] == "final_answer"
```

For recorder failure, assert `TraceWriteError` escapes and no valid final answer is returned. For `max_rounds=1` with a tool-using response, assert `runtime.last_outcome.status == "max_rounds"`; Task 7 verifies worker terminal finalization.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_agent_loop_source.py -k "trace or max_rounds" -v`

Expected: failure because the runtime has no recorder or explicit loop outcome.

- [ ] **Step 3: Make runtime dependencies explicit**

Add:

```python
@dataclass(frozen=True)
class AgentLoopOutcome:
    status: str
    final_text: str
    failure_class: str | None = None
    failure_message: str | None = None
```

Change `agent_loop()` to receive keyword-only `provider`, `tools`, `handlers`, `max_rounds`, `memory_enabled`, and `run_context`. Replace `while True` with `for round_index in range(max_rounds)`. Preserve the existing module globals only as compatibility defaults for direct legacy callers; `SourceRuntime` must pass explicit values.

Change `SourceRuntime.__init__()` to accept:

```python
def __init__(self, provider, permissions=None, approval_callback=None, *,
             recorder=None, tool_definitions=None, tool_handlers=None,
             max_rounds=40, memory_enabled=None):
```

Keep the benchmark-facing turn signature exact:

```python
def run_turn(self, query: str, *, task_id: str | None = None,
             cutoff: str | None = None,
             run_metadata: dict[str, Any] | None = None) -> str:
```

Wrap the provider once with `TracingProvider`. When `task_id` and an already-started recorder are supplied, `run_turn()` creates a `RunContext`, binds it around the whole loop, records `final_answer` or `run_failed`, and saves the explicit result in `runtime.last_outcome`. It does not write a terminal manifest state: the single-task worker owns terminal finalization so resource cleanup is part of success. Interactive calls with no recorder retain the current string-returning behavior.

- [ ] **Step 4: Expose compaction metadata and call kinds**

Add a `CompactionReport` dataclass in `context.py` with method, original/retained message counts, original/retained character estimates, and transcript path. Add optional `on_compaction: Callable[[CompactionReport], None]` and `provider` parameters to `prepare_context()`, `compact_history()`, and `reactive_compact()`.

Wrap summary-provider calls in `model_call_scope("context_compaction")`. In `agent_loop()`, record `context_compaction` immediately when the callback fires. Record manual, proactive, and reactive compaction as distinct methods.

- [ ] **Step 5: Update interactive construction without enabling benchmark behavior**

Make `build_runtime()` pass the provider, current tool tables, configured `max_rounds`, and `config.MEMORY_ENABLED` explicitly. Keep recorder `None`, initialize cron/background/team exactly as before, and keep existing CLI command behavior unchanged.

- [ ] **Step 6: Run loop and context regression tests**

Run: `python -m pytest tests/test_agent_loop_source.py tests/test_context_recovery.py tests/test_cli_source.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit runtime lifecycle tracing**

```bash
git add simple_cc/agent.py simple_cc/context.py simple_cc/__main__.py tests/test_agent_loop_source.py tests/test_context_recovery.py
git commit -m "feat: trace source runtime lifecycle"
```

### Task 4: Tool spans, cutoff enforcement, evidence artifacts, and citation linkage

**Files:**
- Modify: `simple_cc/tools.py:31-37`
- Modify: `simple_cc/agent.py:296-379`
- Modify: `simple_cc/web_research.py:272-end`
- Modify: `simple_cc/pdf_research.py:185-end`
- Create: `simple_cc/evidence.py`
- Create: `tests/test_evidence_trace.py`
- Extend: `tests/test_telemetry.py`
- Extend: `tests/test_web_research.py`
- Extend: `tests/test_pdf_research.py`

**Interfaces:**
- Consumes: active `RunContext`, `TraceRecorder.store_artifact()`, and existing JSON tool outputs.
- Produces: `prepare_research_arguments()`, `ToolCapture`, `bind_tool_capture()`, `capture_tool_artifact()`, `record_research_evidence()`, and `link_final_answer_sources()`.

- [ ] **Step 1: Write failing cutoff and tool-span tests**

Cover these cases:

```python
prepared = prepare_research_arguments(
    "web_search", {"query": "rates"}, required_cutoff="2025-05-01"
)
assert prepared.arguments["cutoff"] == "2025-05-01"
assert prepared.decision == "injected"

with pytest.raises(CutoffMismatch):
    prepare_research_arguments(
        "web_fetch", {"url": "https://example.com", "cutoff": "2025-05-02"},
        required_cutoff="2025-05-01",
    )
```

Run one successful and one denied tool call through `agent_loop()` and assert `tool_requested`, `cutoff_validation`, `permission_decision`, `tool_started`, and exactly one `tool_result` or `tool_error` share the tool-use ID/span. Add web and PDF cases proving explicitly post-cutoff/conflicting-date source bodies are absent from both `artifacts/` and every JSONL payload.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_evidence_trace.py tests/test_telemetry.py -k tool -v`

Expected: failure because cutoff injection and tool spans do not exist.

- [ ] **Step 3: Implement internal tool capture without changing schemas**

Add to `telemetry.py`:

```python
@dataclass
class ToolCapture:
    recorder: TraceRecorder
    span_id: str
    artifacts: list[ArtifactRef] = field(default_factory=list)

    def store(self, content, *, media_type: str, source: str, suffix: str) -> ArtifactRef:
        ref = self.recorder.store_artifact(
            content, media_type=media_type, source=source, suffix=suffix
        )
        self.artifacts.append(ref)
        return ref
```

Expose a context manager backed by `ContextVar[ToolCapture | None]` and a `capture_tool_artifact()` helper that is a no-op when no trace is active. Change `call_tool_handler()` to accept keyword-only `capture: ToolCapture | None = None`, bind it only during handler execution, and preserve the existing string result and TypeError behavior.

- [ ] **Step 4: Capture raw web and PDF evidence**

Do not capture raw content immediately after download. In `web_fetch()`, call `capture_tool_artifact(html, media_type=content_type, source=final_url, suffix=".html" or ".txt")` only after date-conflict and post-cutoff checks accept the page. In `pdf_fetch()`, capture `data` as `application/pdf` only after PDF metadata date checks accept the document; then capture the exact accepted page array returned to the model as UTF-8 JSON. Rejected post-cutoff/conflicting-date bodies may exist transiently in memory but must never enter trace artifacts or model-visible output. Never insert filesystem paths into model-visible tool output.

- [ ] **Step 5: Enforce one runtime cutoff and parse research evidence**

In `evidence.py`, define:

```python
RESEARCH_TOOLS = {"web_search", "web_fetch", "pdf_fetch"}


@dataclass(frozen=True)
class PreparedToolArguments:
    arguments: dict[str, Any]
    decision: str
    supplied_cutoff: str | None


class CutoffMismatch(ValueError):
    pass
```

`prepare_research_arguments()` must copy model arguments, inject the run cutoff when absent, accept an identical cutoff, reject a different cutoff before dispatch, and return `not_required` when the run cutoff is `None`.

`record_research_evidence()` must parse JSON output and emit:

- search outcome, normalized query, candidate URLs, and `snippet_only=true` for `web_search`;
- `source_registered` with deterministic `source_id`, canonical URL, publication/date status, cutoff, page range, model-visible output artifact, and raw artifact references for successful fetch/PDF calls;
- structured failure code for `post_cutoff`/`published_after_cutoff`, `date_conflict`, `unknown_date`, unsafe URL, timeout, unsupported type, and extraction failure.

Search snippets are candidates, not fetched evidence. Canonicalize URLs by lowercasing scheme/host, removing fragments, removing default ports, sorting query parameters, and preserving path case. A registered source ID is `src_` followed by the first 16 hexadecimal characters of `sha256(canonical_url.encode("utf-8"))`.

- [ ] **Step 6: Integrate tool tracing into the active source loop**

For every tool-use block:

1. record original `tool_requested` arguments;
2. run cutoff preparation and record `cutoff_validation`;
3. record hook and permission decisions before execution;
4. create one `ToolCapture` and record `tool_started` with normalized arguments;
5. call the handler;
6. store the exact model-visible output as an artifact;
7. emit `tool_result` or `tool_error` with latency and artifact references;
8. call `record_research_evidence()`.

On cutoff mismatch, do not call the tool; return a structured tool result with error code `cutoff_mismatch` and record the rejection.

- [ ] **Step 7: Link final-answer citations to fetched evidence**

Implement `link_final_answer_sources(final_text, registered_sources)` to extract HTTP(S) URLs and Markdown-link targets, canonicalize them, and return `cited_urls`, `matched_source_ids`, and `unmatched_citations`. Add these fields to the `final_answer` event. Do not treat a domain name with no URL as a machine-verifiable citation.

- [ ] **Step 8: Run research and telemetry tests**

Run: `python -m pytest tests/test_evidence_trace.py tests/test_telemetry.py tests/test_web_research.py tests/test_pdf_research.py tests/test_agent_loop_source.py -v`

Expected: all tests pass without network access.

- [ ] **Step 9: Commit evidence tracing**

```bash
git add simple_cc/tools.py simple_cc/agent.py simple_cc/telemetry.py simple_cc/evidence.py simple_cc/web_research.py simple_cc/pdf_research.py tests/test_evidence_trace.py tests/test_telemetry.py tests/test_web_research.py tests/test_pdf_research.py
git commit -m "feat: trace tool evidence and cutoff decisions"
```

### Task 5: All-in accounting for memory, compaction, subagents, background work, and teams

**Files:**
- Modify: `simple_cc/memory.py:458-471,640-690`
- Modify: `simple_cc/context.py:285-302`
- Modify: `simple_cc/subagents.py:96-138`
- Modify: `simple_cc/background.py:91-158`
- Modify: `simple_cc/teams.py`
- Extend: `tests/test_telemetry.py`
- Extend: `tests/test_background_cron_source.py`
- Extend: `tests/test_hooks_subagents.py`
- Extend: `tests/test_teams_source.py`

**Interfaces:**
- Consumes: `model_call_scope()`, `bind_run_context()`, active run/tool context, and existing provider/tool interfaces.
- Produces: trace `call_kind` values `agent`, `context_compaction`, `memory_retrieval`, `memory_extraction`, `memory_consolidation`, `subagent`, and `teammate`; `background_is_quiescent()`.

- [ ] **Step 1: Write failing call-kind and context-propagation tests**

Assert that memory selector/extractor calls are not counted as core-agent calls, a subagent receives a distinct `agent_id`, and a background handler sees the same `run_id` after crossing the thread boundary. Assert completed background events cannot be injected into a later run context.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_telemetry.py tests/test_background_cron_source.py tests/test_hooks_subagents.py -k trace -v`

Expected: failure because auxiliary calls have no classification or propagated context.

- [ ] **Step 3: Label auxiliary provider calls**

Wrap the relevant calls with exact kinds:

```python
with model_call_scope("memory_retrieval"):
    selected = _json_array(self._call(prompt, 300))

with model_call_scope("memory_extraction"):
    values = _json_array(self._call(prompt, 1_000))

with model_call_scope("subagent"):
    response = client.create(
        messages, subagent_system_prompt(), SUB_TOOLS,
        config.DEFAULT_MAX_TOKENS, model=MODEL,
    )
```

Use `memory_consolidation` for consolidation and `context_compaction` for summaries. In subagents and teammates, bind `run.child(agent_id=f"subagent:{id}")` or `run.child(agent_id=f"teammate:{name}")` around the whole child loop.

- [ ] **Step 4: Propagate trace context into background threads**

Capture `contextvars.copy_context()` in `start_background_task()` before constructing the thread, then run the existing worker body through `copied_context.run(worker_body)`. Add:

```python
def background_is_quiescent() -> bool:
    with background_lock:
        return not any(
            task.get("thread") is not None and task["thread"].is_alive()
            for task in background_tasks.values()
        )
```

Record background submission, completion, cancellation, and latency with the originating tool span as parent.

- [ ] **Step 5: Run auxiliary-accounting tests**

Run: `python -m pytest tests/test_telemetry.py tests/test_background_cron_source.py tests/test_hooks_subagents.py tests/test_teams_source.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit all-in accounting**

```bash
git add simple_cc/memory.py simple_cc/context.py simple_cc/subagents.py simple_cc/background.py simple_cc/teams.py tests/test_telemetry.py tests/test_background_cron_source.py tests/test_hooks_subagents.py tests/test_teams_source.py
git commit -m "feat: correlate auxiliary agent activity"
```

### Task 6: Benchmark-only clean runtime factory

**Files:**
- Create: `simple_cc/benchmark.py`
- Create: `tests/test_benchmark_runtime.py`

**Interfaces:**
- Consumes: `Settings`, `SourceRuntime`, `TraceRecorder`, `TracingProvider`, tool tables, and shutdown functions.
- Produces: `BenchmarkOptions`, `BenchmarkSession`, `build_benchmark_runtime()`, `validate_clean_workspace()`.

- [ ] **Step 1: Write failing clean-runtime tests**

Test that a non-empty workspace is rejected, the run directory cannot equal or be inside the workspace, benchmark tools omit cron/team/task tools by default, memory is disabled even if the interactive environment enables it, and no cron/team thread starts.

Use these forbidden default tool names:

```python
FORBIDDEN = {
    "task", "schedule_cron", "list_crons", "cancel_cron",
    "spawn_teammate", "send_message", "check_inbox", "request_shutdown",
    "request_plan", "review_plan",
}
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_benchmark_runtime.py -v`

Expected: collection fails because `simple_cc.benchmark` does not exist.

- [ ] **Step 3: Implement benchmark options and path validation**

```python
@dataclass(frozen=True)
class BenchmarkOptions:
    memory_enabled: bool = False
    cron_enabled: bool = False
    team_enabled: bool = False
    subagent_enabled: bool = False
    max_rounds: int = 40


def validate_clean_workspace(run_dir: Path, workspace: Path) -> None:
    run_dir = run_dir.resolve()
    workspace = workspace.resolve()
    workspace.relative_to(run_dir)
    if workspace == run_dir:
        raise ValueError("agent workspace must not equal run directory")
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("agent workspace must be empty")
```

Also reject symlinks in the workspace ancestor chain under `run_dir`, create the workspace only after validation, and verify that `manifest.json`, `trajectory.jsonl`, `artifacts/`, task input, and output paths are siblings of—not children of—`agent_workspace`.

- [ ] **Step 4: Build a benchmark session without interactive services**

The worker calls `validate_clean_workspace()` before `Settings.from_env()` creates `.simple_cc`. `build_benchmark_runtime()` then configures that prevalidated unique workspace, wraps the supplied provider, filters tool definitions and handlers from the same allowlist, creates `SourceRuntime(memory_enabled=False, approval_callback=None)`, and does not call `initialize_cron()`, `set_team_provider()`, or start CLI polling threads.

`BenchmarkSession.close(timeout)` must call existing background/team/cron shutdown functions defensively and return a frozen result containing `stopped` and `live_resources`. A non-quiescent close prevents `completed` finalization.

- [ ] **Step 5: Run benchmark-runtime tests**

Run: `python -m pytest tests/test_benchmark_runtime.py tests/test_tools_permissions_hooks.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the benchmark runtime factory**

```bash
git add simple_cc/benchmark.py tests/test_benchmark_runtime.py
git commit -m "feat: add isolated benchmark runtime"
```

### Task 7: Single-task worker with fail-closed finalization

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/run_task.py`
- Create: `tests/test_benchmark_worker.py`

**Interfaces:**
- Consumes: `build_benchmark_runtime()`, `TraceRecorder`, `Settings`, `SiliconFlowProvider`.
- Produces: `TaskInput`, `load_task_input()`, `execute_task()`, and a one-task CLI.

- [ ] **Step 1: Write failing worker validation and lifecycle tests**

Test valid input, missing `task_id`, invalid cutoff format, mismatched workspace/run directory, normal completion, provider failure, recorder failure, and resource-close failure. Use `ScriptedProvider`; no worker unit test may access the network.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_benchmark_worker.py -v`

Expected: collection fails because `eval.run_task` does not exist.

- [ ] **Step 3: Implement strict task input**

```python
@dataclass(frozen=True)
class TaskInput:
    run_id: str
    task_id: str
    question: str
    cutoff: str | None
    benchmark: str
    task_type: str
    retry_of_run_id: str | None = None
```

`load_task_input()` accepts only one JSON object, rejects empty IDs/questions, validates cutoff with `date.fromisoformat()` plus round-trip equality, and ignores neither malformed nor duplicate fields.

- [ ] **Step 4: Implement one-task execution and atomic answer output**

Define:

```python
def execute_task(task: TaskInput, run_dir: Path, workspace: Path,
                 provider: ChatProvider) -> int:
```

It must create and start a recorder, record worker PID and isolation flags, build one benchmark session, execute exactly one `run_turn()`, inspect `runtime.last_outcome`, and close task-owned resources before choosing a terminal status. Only a completed loop plus quiescent cleanup may record `run_completed`, atomically write `final_answer.txt`, and finalize `completed`. A non-quiescent close finalizes `failed` with `failure_class="resource_leak"`; a failed or max-round loop uses its corresponding status. If tracing fails, do not write a valid final answer.

Build manifest metadata before `start_run()` with exact keys:

```python
metadata = {
    "benchmark": task.benchmark,
    "task_type": task.task_type,
    "pit_mode": "non_strict_live_web",
    "provider": "siliconflow",
    "model": settings.model,
    "worker_pid": os.getpid(),
    "agent_workspace": str(workspace.resolve()),
    "isolation": {
        "one_task_per_process": True,
        "memory_enabled": False,
        "cron_enabled": False,
        "team_enabled": False,
        "subagent_enabled": False,
        "interactive_approval_enabled": False,
    },
    "max_rounds": settings.max_rounds,
    "prompt_sha256": prompt_sha256,
    "tool_schema_sha256": tool_schema_sha256,
    "git_commit": git_commit,
    "git_dirty": git_dirty,
    "retry_of_run_id": task.retry_of_run_id,
}
```

Hash canonical UTF-8 JSON with sorted keys and compact separators. Obtain Git metadata using argument-list subprocess calls from the repository root; if Git metadata is unavailable, record `git_commit=null`, `git_dirty=null`, and a non-secret `metadata_warning` instead of failing the task.

The CLI accepts only `--task-input`, `--run-dir`, and `--workspace`, constructs the real provider from environment-backed `Settings`, and exits after one task. It must not expose a multi-task loop.

- [ ] **Step 5: Run worker tests**

Run: `python -m pytest tests/test_benchmark_worker.py tests/test_benchmark_runtime.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the single-task worker**

```bash
git add eval/__init__.py eval/run_task.py tests/test_benchmark_worker.py
git commit -m "feat: add fail-closed benchmark worker"
```

### Task 8: Concurrent parent runner, fresh processes, timeout, resume, and retry

**Files:**
- Create: `eval/run_benchmark.py`
- Create: `tests/fixtures/isolation_probe.py`
- Create: `tests/test_benchmark_runner.py`

**Interfaces:**
- Consumes: task JSONL, the worker CLI from Task 7, and terminal manifest statuses.
- Produces: `RunnerOptions`, `RunAssignment`, `allocate_run()`, `launch_assignment()`, `run_benchmark()`.

- [ ] **Step 1: Write failing isolation and concurrency tests**

Use `tests/fixtures/isolation_probe.py` as the child command. It writes its PID, workspace, and whether a marker from another task is visible. Launch two tasks with `workers=2` and assert different PIDs, different absolute workspaces, and no cross-task marker visibility.

Add tests for timeout, nonzero worker exit, malformed/missing manifest, completed-task resume, incomplete-task retry, and retry linkage. Assert retries always have new run IDs and `retry_of_run_id` points to the prior attempt.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_benchmark_runner.py -v`

Expected: collection fails because `eval.run_benchmark` does not exist.

- [ ] **Step 3: Implement assignment and immutable run allocation**

```python
@dataclass(frozen=True)
class RunnerOptions:
    dataset: Path
    output_dir: Path
    workers: int = 2
    timeout_seconds: float = 900.0
    limit: int | None = None
    task_ids: frozenset[str] = frozenset()
    resume: bool = False


@dataclass(frozen=True)
class RunAssignment:
    task: dict[str, Any]
    run_id: str
    run_dir: Path
    workspace: Path
    task_input: Path
    retry_of_run_id: str | None
```

`allocate_run()` uses `uuid.uuid4()`, creates `run_dir`, `agent_workspace`, and an atomic `task_input.json` outside the workspace. It never deletes or reuses a prior run directory.

- [ ] **Step 4: Launch one fresh interpreter per assignment**

Use an argument list with `shell=False`:

```python
command = [
    sys.executable, "-m", "eval.run_task",
    "--task-input", str(assignment.task_input),
    "--run-dir", str(assignment.run_dir),
    "--workspace", str(assignment.workspace),
]
```

Use `CREATE_NEW_PROCESS_GROUP` on Windows and `start_new_session=True` on POSIX. On timeout, terminate the process tree (`taskkill /PID <pid> /T /F` on Windows; `os.killpg()` with `SIGTERM`, then `SIGKILL`, on POSIX), wait for exit, and call `supervisor_finalize_manifest()` with `timed_out`. A nonzero exit with no terminal manifest calls the same utility with `worker_crashed`. Never synthesize `completed`.

- [ ] **Step 5: Add bounded concurrency and resume**

Use `ThreadPoolExecutor(max_workers=options.workers)` only in the parent; each submitted function must launch a new child interpreter. Validate `workers >= 1` and `timeout_seconds > 0`. Resume skips a task only when its latest manifest is valid and exactly `completed`; incomplete, crashed, timed-out, and trace-invalid attempts produce a new assignment linked to the prior run.

Expose `--dataset`, `--output-dir` (default `eval/runs`), repeatable `--task-id`, `--limit`, `--workers` (default 2), `--timeout-seconds` (default 900), and `--resume`. Print one compact JSON summary containing counts by terminal status and return nonzero when any selected task lacks a completed attempt.

- [ ] **Step 6: Run runner tests**

Run: `python -m pytest tests/test_benchmark_runner.py -v`

Expected: all tests pass and the isolation test reports two distinct PIDs.

- [ ] **Step 7: Commit the parent runner**

```bash
git add eval/run_benchmark.py tests/fixtures/isolation_probe.py tests/test_benchmark_runner.py
git commit -m "feat: run benchmark tasks in isolated processes"
```

### Task 9: Strict trajectory reader and derived metrics

**Files:**
- Create: `simple_cc/eval_metrics.py`
- Create: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `manifest.json`, `trajectory.jsonl`, and artifact references.
- Produces: `read_trajectory()`, `validate_trace()`, `derive_metrics()`, `RunMetrics`.

- [ ] **Step 1: Write failing reader and metric tests**

Cover sequential valid events, a truncated last line, an invalid middle line, sequence gaps, wrong run ID, missing terminal event, unknown usage, repeated queries/URLs, independent domains, cutoff failures, core/all-in tokens, and latency totals.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_eval_metrics.py -v`

Expected: collection fails because `simple_cc.eval_metrics` does not exist.

- [ ] **Step 3: Implement strict reading and validation**

```python
@dataclass(frozen=True)
class RunMetrics:
    status: str
    trace_valid: bool
    total_duration_ms: float | None
    model_calls: int
    tool_calls: int
    searches: int
    fetches: int
    documents: int
    tool_failures: int
    repeated_query_rate: float | None
    repeated_url_rate: float | None
    independent_domains: int
    unknown_date_sources: int
    rejected_post_cutoff: int
    core_prompt_tokens: int | None
    core_completion_tokens: int | None
    all_in_prompt_tokens: int | None
    all_in_completion_tokens: int | None
```

`read_trajectory()` may ignore only one malformed unterminated final line and must mark the run incomplete. Any malformed complete line, duplicate/gapped sequence, inconsistent run ID, missing required field, or artifact hash mismatch makes the trace invalid.

- [ ] **Step 4: Derive metrics without inspecting runtime messages**

Count from events only. Treat provider usage as unknown if any included successful call lacks the relevant usage field; never add `None` as zero. Compute core usage from `call_kind="agent"`; all-in usage includes every call kind. Normalize repeated queries after whitespace/case folding and repeated URLs using the evidence canonicalizer. Count independent registered source hostnames, not search-result snippets.

Accept an optional versioned pricing JSON object; calculate currency cost only when every used model has dated input/output rates. Otherwise return token totals and `cost=None`.

- [ ] **Step 5: Run metric tests**

Run: `python -m pytest tests/test_eval_metrics.py tests/test_trace.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit metric derivation**

```bash
git add simple_cc/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat: derive evaluation metrics from traces"
```

### Task 10: End-to-end acceptance, documentation, and full regression

**Files:**
- Modify: `README.md`
- Extend: `tests/test_benchmark_worker.py`
- Extend: `tests/test_benchmark_runner.py`
- Extend: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: all interfaces from Tasks 1-9.
- Produces: documented benchmark workflow and a complete offline acceptance test.

- [ ] **Step 1: Write one offline end-to-end acceptance test**

Run two probe-backed benchmark tasks concurrently, and run one scripted-provider worker through a search → fetch → final-answer trajectory. Assert:

- unique process IDs and workspaces;
- no task sees another task's marker, memory, transcript, mailbox, or output;
- trace/reference/judge paths are rejected by agent file tools;
- manifest, trajectory, final answer, and referenced artifacts exist;
- model and tool spans have durations;
- cutoff is injected and mismatches are rejected;
- fetched evidence and final citations link by source ID;
- derived metrics use only trace events;
- PIT mode is `non_strict_live_web`.

- [ ] **Step 2: Run the acceptance test and route failures back to their owning task**

Run: `python -m pytest tests/test_benchmark_worker.py tests/test_benchmark_runner.py tests/test_eval_metrics.py -v`

Expected: all tests pass. If this command fails, return to the task that owns the failing interface, add the smallest regression test there, make it pass, and rerun this acceptance command.

- [ ] **Step 3: Document exact commands and status semantics**

Add these user-facing examples to `README.md`:

```powershell
python -m eval.run_benchmark --dataset eval/data/financegym_20.jsonl --workers 2
python -m eval.run_benchmark --dataset eval/data/benchmark_400_public.jsonl --workers 4 --resume
```

Document the run-directory tree, one-task-per-process guarantee, benchmark-default disabled features, terminal statuses, retry behavior, token unknown semantics, and the statement that `before:` is only a live-search hint and does not create strict PIT replay.

- [ ] **Step 4: Run formatting-independent static checks**

Run: `python -m compileall simple_cc eval tests`

Expected: exit code 0.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -v`

Expected: all tests pass; no test requires live network access.

- [ ] **Step 6: Inspect the final diff for secrets and unrelated files**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors, no `.env`, API keys, benchmark run outputs, or unrelated user files staged.

- [ ] **Step 7: Commit documentation and acceptance coverage**

```bash
git add README.md tests/test_benchmark_worker.py tests/test_benchmark_runner.py tests/test_eval_metrics.py
git commit -m "docs: document financial agent evaluation tracing"
```

## Final Verification Checklist

- [ ] Run `python -m pytest -v` and retain the final pass count.
- [ ] Run a two-task offline isolation probe and retain both child PIDs and workspace paths.
- [ ] Inspect one completed run and verify every artifact reference exists and matches its SHA-256.
- [ ] Kill one worker after at least one event and verify prior JSONL lines remain readable and the run is not `completed`.
- [ ] Verify a benchmark worker cannot read a sibling `reference_answer.json` through `read_file` or `glob`, and that its shell attempt is denied because no approval callback exists.
- [ ] Verify a post-cutoff fetch is rejected and its body never appears in a model-visible artifact.
- [ ] Verify missing provider usage yields metric `null`, not zero.
- [ ] Verify manifest feature flags show memory, cron, team, subagent, and interactive approval disabled by default.
- [ ] Verify completed retries do not overwrite prior attempts and carry `retry_of_run_id`.
- [ ] Verify README states `non_strict_live_web` and does not claim strict historical replay.
