# Financial Research Agent Append-Only Evaluation Trace Design

**Date:** 2026-08-13  
**Status:** Approved approach; written specification awaiting user review  
**Repository:** `E:\AgentLearnProject\simple_cc`

## 1. Purpose

Add an append-only evaluation trace to Simple CC so a financial research run can be scored after completion without relying on the agent's mutable conversation context.

The trace must preserve the observable evidence needed to evaluate four rubric layers:

1. result quality;
2. research process and evidence provenance;
3. efficiency and resource consumption;
4. safety and point-in-time risk controls.

The current CLI uses `SourceRuntime` and the module-level `agent_loop()` in `simple_cc/agent.py`. This is the primary integration path. The later compatibility `AgentRuntime` class is outside the first implementation scope unless a test or active caller requires equivalent instrumentation.

## 2. Goals

The implementation will:

- create one durable run directory for every benchmark task;
- execute every benchmark task in a fresh operating-system process with a task-exclusive workspace;
- append events as model and tool operations happen;
- preserve events independently of context budgeting, summarization, and compaction;
- record the final answer, search queries, fetched URLs, fetched contents, document contents, tool inputs and outputs, timestamps, durations, token usage, errors, and permission decisions;
- bind every event to a `run_id` and benchmark `task_id`;
- preserve large evidence as immutable artifacts referenced from JSONL events;
- enforce a single task cutoff across all search and fetch calls;
- identify the current web policy as `non_strict_live_web` rather than claiming strict historical replay;
- redact secrets before data reaches the trace store;
- remain usable after abnormal termination, including a partially written run.

## 3. Non-goals

The first implementation will not:

- store or request hidden chain-of-thought; the trace contains only observable model output, tool calls, tool results, and system decisions;
- turn live web search into a strict historical corpus;
- assign final rubric scores inside the agent runtime;
- replace the agent's existing conversation history or context compaction;
- introduce OpenTelemetry or an external tracing service;
- combine the four rubric layers into a single weighted score.

## 4. Chosen architecture

### 4.1 Append-only event source

A new `TraceRecorder` writes one JSON object per line to `trajectory.jsonl`. Each event is written when the operation occurs. The recorder never reconstructs a run by exporting `messages` after the turn.

This makes `trajectory.jsonl` the evaluation source of truth. The conversation history remains an inference mechanism and may be compacted without affecting the trace.

### 4.2 Run directory

Each benchmark execution uses this structure:

```text
eval/runs/{run_id}/
  manifest.json
  trajectory.jsonl
  final_answer.txt
  artifacts/
    web_0001.txt
    document_0002.pdf
    document_0002.extracted.txt
    tool_0003.json
  agent_workspace/
    .simple_cc/
```

`run_id` is a collision-resistant UUID. Human-readable timestamps may appear in directory indexes but must not be the only identifier.

`agent_workspace/` is the only part of the run directory visible to agent file tools. The trace, final answer, artifacts, task dataset, reference answers, rubrics, and judge outputs remain outside that workspace. The benchmark permission policy denies shell execution without an approval callback, so the agent cannot use `bash` to bypass the workspace boundary.

### 4.3 Benchmark process isolation

The parent benchmark runner never executes two tasks through the same imported Simple CC runtime. It starts one fresh Python worker process per task, and each worker handles exactly one task before exiting. Process isolation resets module-level state, including provider bindings, `config` paths, memory stores, cron queues, background-task registries, subagent globals, team registries, hooks, counters, and conversation messages.

Each worker receives a unique absolute `run_dir` and `agent_workspace`. Two workers must never share either path. The parent may run several workers concurrently, but concurrency changes throughput only; it does not change the one-task-per-process rule.

Benchmark defaults are:

- memory disabled;
- cron and durable scheduled tasks disabled;
- team tools and cross-task mailboxes disabled;
- no interactive approval callback, which denies `bash` under the current permission policy;
- background or subagent work, when explicitly enabled by an experiment configuration, remains owned by that task worker and must finish or be cancelled before the worker reports completion.

The worker records these feature flags in `manifest.json`. A worker that cannot stop its owned asynchronous work is terminated by the parent and classified as failed or cancelled rather than completed. The parent also cleans up the worker process tree so detached descendants cannot survive into the next task.

### 4.4 New module

Create `simple_cc/trace.py` with:

```python
class TraceRecorder:
    def start_run(self, *, task_id, question, cutoff, metadata): ...
    def record(self, event_type, payload, *, span_id=None, parent_span_id=None): ...
    def store_artifact(self, content, *, media_type, source, suffix): ...
    def finish_run(self, *, status, final_answer=None, error=None): ...

class NullTraceRecorder:
    ...
```

`NullTraceRecorder` keeps ordinary interactive use backward compatible. Benchmark mode uses a required recorder and fails the run if durable trace writing fails.

## 5. Trace schema

### 5.1 Common event envelope

Every JSONL row contains:

```json
{
  "schema_version": "1.0",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_id": "69f904b728538874c086db16",
  "sequence": 12,
  "event_id": "b40d3de1-4c65-45e6-94af-61799d7347d1",
  "event_type": "tool_result",
  "timestamp_utc": "2026-08-13T08:21:31.274Z",
  "elapsed_ms": 8421.37,
  "span_id": "tool_call_03",
  "parent_span_id": "llm_call_02",
  "agent_id": "root",
  "payload": {}
}
```

Rules:

- `sequence` is assigned under a lock and increases monotonically within a run;
- `timestamp_utc` is a wall-clock audit timestamp;
- `elapsed_ms` is calculated from a monotonic run clock;
- `span_id` connects start/result/error events;
- `parent_span_id` represents causal nesting;
- concurrent tool or subagent events share the same run but retain distinct spans and `agent_id` values.

Each event is serialized to a single line, flushed, and `fsync`ed before `record()` returns in required benchmark mode. An incomplete final line after a process crash is ignored by the reader and marks the run incomplete.

### 5.2 Required event types

The runtime records:

| Event | Required payload |
|---|---|
| `run_started` | question, cutoff, PIT mode, task metadata |
| `user_prompt` | exact benchmark prompt after secret redaction |
| `llm_request_started` | call kind, model, maximum tokens, prompt/tool-schema hashes |
| `llm_response` | observable text, tool calls, stop reason, usage, latency, provider request ID |
| `llm_error` | exception class, safe message, attempts, latency |
| `tool_requested` | tool call ID, tool name, original arguments |
| `cutoff_validation` | required cutoff, supplied cutoff, normalized cutoff, decision |
| `permission_decision` | allow/deny, reason, whether human approval was involved |
| `tool_started` | tool name, normalized arguments |
| `tool_result` | success state, preview or artifact reference, latency |
| `tool_error` | structured error code, safe message, latency |
| `source_registered` | source ID, canonical URL, publication status, artifact reference |
| `context_compaction` | method, original counts/sizes, retained counts/sizes, transcript path |
| `final_answer` | exact final answer and final-answer artifact |
| `run_completed` | total duration and derived count summary |
| `run_failed` | failure class, stage, safe message and duration |

Retries produce separate model-call spans at the agent-call level. Provider-internal attempts are reported through an `attempts` field. Token totals include every successful provider response for which the provider returns usage. Failed provider attempts with unavailable usage are counted as attempts but reported with unknown usage rather than assumed to be zero.

## 6. Manifest

`manifest.json` is written at run start and finalized at run end. It contains:

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "task_id": "...",
  "benchmark": "financegym",
  "task_type": "research_analysis",
  "question": "...",
  "cutoff": "2025-08-05",
  "pit_mode": "non_strict_live_web",
  "provider": "siliconflow",
  "model": "...",
  "worker_pid": 12345,
  "agent_workspace": ".../eval/runs/{run_id}/agent_workspace",
  "isolation": {
    "one_task_per_process": true,
    "memory_enabled": false,
    "cron_enabled": false,
    "team_enabled": false,
    "interactive_approval_enabled": false
  },
  "max_rounds": 40,
  "prompt_sha256": "...",
  "tool_schema_sha256": "...",
  "git_commit": "...",
  "git_dirty": true,
  "started_at": "...",
  "ended_at": null,
  "status": "running"
}
```

On success, `status` becomes `completed`. Expected terminal alternatives are `failed`, `max_rounds`, `cancelled`, `timed_out`, `worker_crashed`, and `trace_invalid`.

Manifest updates use a temporary file and atomic replace so readers never observe partially serialized JSON. The worker owns manifest updates while it is alive. After the worker exits, the parent may finalize a still-running manifest as `timed_out` or `worker_crashed`; it must never synthesize `completed`.

## 7. Runtime integration

### 7.1 `SourceRuntime`

Change `SourceRuntime.__init__` to accept a recorder factory or recorder. Change the benchmark-facing method to:

```python
def run_turn(
    self,
    query: str,
    *,
    task_id: str | None = None,
    cutoff: str | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> str:
    ...
```

Ordinary CLI calls may omit the new keyword arguments. A benchmark run requires `task_id` and a trace recorder.

### 7.2 `agent_loop()`

Pass a `RunContext` into `agent_loop()` containing:

- recorder;
- run ID;
- task ID;
- task cutoff;
- active agent ID.

Instrument the boundaries around:

- `call_llm()`;
- permission evaluation;
- `call_tool_handler()`;
- background task submission and completion;
- compaction;
- normal final return;
- every exception and forced stop.

Instrumentation must not use the post-compaction `messages` list as the primary data source.

### 7.3 Failure policy

In benchmark mode, a trace write failure makes the run `trace_invalid` and stops evaluation. Producing a model answer without an auditable trace is not a valid benchmark completion.

In interactive mode, tracing may be disabled with `NullTraceRecorder`.

## 8. Provider usage and timing

Extend `ProviderResponse` in `simple_cc/provider.py` with:

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
    usage: ProviderUsage
    model: str | None
    provider_request_id: str | None
    attempts: int
```

The SiliconFlow adapter reads `response.usage`, `response.model`, and `response.id`. Missing provider fields remain `null`; they are never converted to zero.

Each model invocation is tagged with a `call_kind`:

- `agent`;
- `context_summary`;
- `memory_retrieval`;
- `memory_extraction`;
- `subagent`.

The evaluator reports core-agent usage and all-in usage separately.

## 9. Tool events and evidence artifacts

### 9.1 Generic tools

For every tool call, retain:

- tool call ID;
- tool name;
- original arguments;
- normalized arguments after runtime enforcement;
- permission result;
- start and end timestamps;
- latency;
- success/error status;
- output byte and character sizes;
- output preview;
- artifact reference when the output exceeds the inline threshold.

The recorder observes the output without changing what the agent receives.

### 9.2 Web search

Record the normalized query, result limit, cutoff, result URLs, provider backend, result count, latency, and structured failure.

Search snippets are candidates rather than verified evidence. A URL becomes a registered evidence source only after successful fetch or document processing.

### 9.3 Web fetch

For successful fetches, copy the full returned `content` into an artifact for the trace representation and retain:

- canonical final URL;
- title;
- publication date and `date_status`;
- cutoff and PIT mode;
- truncation state;
- content media type;
- SHA-256;
- artifact relative path;
- a bounded preview.

This artifact extraction does not remove or shorten the content delivered to the agent. Context budgeting may later compact the agent's copy, while the trace artifact remains immutable.

Repeated fetches of identical canonical URL and identical content hash reference the existing artifact while still generating a separate tool event. This permits duplicate-fetch scoring without duplicating storage.

### 9.4 Financial documents

Add a separate `document_fetch` tool rather than broadening HTML behavior invisibly. It uses the existing public-URL and redirect safety checks, accepts bounded PDF responses, stores the original PDF, extracts text with page boundaries, and stores the extracted representation.

Its result contains URL, publication status, document hash, page count, pages exposed to the model, extraction method, and artifact references. Unsupported or failed extraction remains a structured tool error.

The first version supports text-bearing PDFs. OCR for scanned PDFs is a later capability and must be reported as unsupported rather than silently returning empty content.

## 10. Cutoff and PIT enforcement

The benchmark runner sets one task cutoff in `RunContext`. Before dispatching `web_search`, `web_fetch`, or `document_fetch`, the runtime applies:

1. if the tool omits cutoff, inject the task cutoff;
2. if the supplied cutoff equals the task cutoff, accept it;
3. if it differs, reject the call with `cutoff_mismatch`;
4. record a `cutoff_validation` event in all cases.

The existing rules remain:

- explicitly post-cutoff pages are rejected;
- conflicting explicit dates are rejected;
- unknown-date pages may be returned only with `date_status=unknown` and a warning;
- live search remains `non_strict_live_web`.

Therefore a run may be evaluated for cutoff discipline, but it must not be described as a strict historical replay or official strict-PIT FinanceGym reproduction.

## 11. Source identity and citations

Successful fetched sources receive deterministic run-local IDs in first-seen order: `S1`, `S2`, and so on. `source_registered` links the ID to canonical URL and content artifact.

The tool response shown to the agent includes the source ID. The system prompt instructs the agent to cite claims with source IDs and include the corresponding URL. This creates an observable chain:

```text
final claim -> source ID/URL -> successful tool result -> immutable artifact
```

The judge may still assign `unknown` when a claim-to-source relationship cannot be established.

## 12. Secret handling

All trace fields pass through a redactor before serialization. At minimum it redacts:

- API keys, access tokens, passwords, and secrets;
- authorization and cookie headers;
- URL credentials;
- configured sensitive environment values.

Redaction is explicit:

```json
{
  "value": "[REDACTED]",
  "redaction_reason": "api_key"
}
```

The runtime does not log hidden reasoning or raw SDK request headers. Public financial source material is preserved because removing it would break evidence evaluation. Private benchmark deployments may add a configurable PII policy.

## 13. Benchmark runner

Create two entry points:

- `eval/run_benchmark.py`: parent coordinator that reads tasks, allocates run directories, controls concurrency, starts workers, enforces wall-clock limits, and collects exit metadata;
- `eval/run_task.py`: single-task worker that configures one workspace, creates one runtime, executes one question, closes task-owned resources, and exits.

The parent passes task data to the worker through a task-specific input file outside `agent_workspace/`. The benchmark question must not be embedded in a shell command string. Conceptually, the worker performs:

```python
config.configure_workspace(agent_workspace)
runtime = build_benchmark_runtime(
    workspace=agent_workspace,
    recorder=recorder,
    memory_enabled=False,
    cron_enabled=False,
    team_enabled=False,
)
runtime.run_turn(
    task["question"],
    task_id=task["task_id"],
    cutoff=task.get("cutoff"),
    run_metadata={
        "benchmark": "financegym",
        "task_type": task.get("task_type", "research_analysis"),
    },
)
```

The parent starts a fresh interpreter for every task, for example `python eval/run_task.py --task-input ... --run-dir ... --workspace ...`. A reusable `ProcessPoolExecutor` worker is not sufficient unless it is configured and verified to replace the process after every task.

The runner supports selecting task IDs, limiting task count, choosing the output directory, setting `--workers`, setting a per-task timeout, and resuming by skipping runs already marked `completed`. Initial production runs should default to a conservative worker count, such as two, because model and search-provider rate limits are shared external constraints. It never treats an incomplete or `trace_invalid` directory as completed.

Worker startup validates that:

1. `run_dir` and `agent_workspace` are unique to the task;
2. `agent_workspace` is a child of `run_dir` but trace and evaluation files are not children of `agent_workspace`;
3. no prior memory, task, mailbox, transcript, scheduled-task, or tool-output state exists in the workspace;
4. the worker has exactly one task assignment;
5. benchmark feature flags match the manifest.

The worker does not delete or reuse a prior workspace. On resume, an already completed run is immutable; an incomplete retry receives a new `run_id`, with `retry_of_run_id` linking it to the prior attempt.

## 14. Derived metrics

The trace writer records facts; a separate reader derives:

- total duration;
- model-call and tool-call counts;
- search/fetch/document counts;
- success and failure rates;
- repeated query and repeated URL rates;
- successfully fetched independent domains;
- unknown-date and rejected post-cutoff counts;
- prompt, completion, and total tokens;
- core and all-in token usage;
- cost when an explicit versioned pricing table is supplied.

Pricing is not embedded as an unversioned runtime constant. If pricing is unavailable, dollar cost is unknown while token counts remain valid.

## 15. Rubric observability

The design provides these evidence mappings:

| Rubric layer | Trace evidence |
|---|---|
| Result | `final_answer`, sources, source artifacts, task metadata |
| Process | ordered search/fetch/document/tool events and outcomes |
| Efficiency | monotonic durations, calls, retries, tokens and artifacts |
| Risk | cutoff validation, date decisions, permissions, URL safety and redactions |

`unknown` remains a legitimate rubric value when the necessary semantic evidence is absent. Missing telemetry caused by trace failure invalidates the run rather than producing a deceptively high score with many unknowns.

## 16. Tests

Add focused tests for:

1. monotonic sequence numbers and valid JSONL;
2. immediate event durability before run completion;
3. trace survival across context compaction;
4. normal, failed, cancelled, and max-round terminal statuses;
5. model usage extraction and missing-usage handling;
6. model and tool latency recording with a controllable clock;
7. automatic cutoff injection and mismatch rejection;
8. post-cutoff, date-conflict, and unknown-date trace events;
9. large-output artifact creation, hashing, and deduplication;
10. PDF storage and page-preserving text extraction;
11. secret redaction in arguments, outputs, errors, and manifests;
12. concurrent event ordering;
13. final-answer and source linkage;
14. benchmark resume behavior;
15. recorder failure producing `trace_invalid` rather than a valid scoreable run;
16. two concurrent tasks receiving different process IDs and workspace paths;
17. memory, messages, files, cron jobs, mailboxes, background results, and subagent/team state from task A being unavailable to task B;
18. trace, reference answer, rubric, and judge files being inaccessible through agent file tools;
19. worker timeout and abnormal exit never producing `completed`;
20. worker cleanup stopping task-owned asynchronous work and descendant processes;
21. retry creating a new run directory linked by `retry_of_run_id` instead of reusing dirty state.

All network tests use fixtures or monkeypatching. No default unit test depends on live web access.

## 17. Acceptance criteria

The tracing feature is complete when:

- a normal benchmark task always produces a manifest, trajectory, final answer, and referenced artifacts;
- every task runs under a unique process ID and task-exclusive `agent_workspace`;
- no task can observe another task's conversation, memory, files, queues, background results, subagents, mailboxes, transcripts, or evaluation outputs;
- only `agent_workspace` is visible to agent file tools; gold answers, rubrics, traces, artifacts, and judge outputs are not;
- killing a run after any completed event leaves every prior line readable and durable;
- context compaction cannot remove or rewrite prior trace events;
- every model call reports duration and usage fields, with unavailable values explicitly null;
- every tool call has requested, started, and result/error evidence linked by span;
- every web research tool call uses the task cutoff or is rejected;
- fetched web and PDF evidence can be reconstructed from artifact hashes and paths;
- sensitive configured values do not appear in the trace files;
- rubric feature extraction can calculate process, efficiency, and risk statistics without inspecting mutable in-memory messages;
- the run clearly reports `non_strict_live_web` and does not claim strict PIT.

## 18. Implementation order

1. event schema, recorder, manifest, redaction, and tests;
2. `SourceRuntime` and `agent_loop()` lifecycle instrumentation;
3. provider usage and model-call timing;
4. tool timing, cutoff enforcement, and source registration;
5. evidence artifact extraction and deduplication;
6. PDF `document_fetch` support;
7. single-task worker, process-isolated benchmark runner, cleanup, resume, and isolation tests;
8. derived-metric reader;
9. end-to-end trace and rubric-observability tests.
