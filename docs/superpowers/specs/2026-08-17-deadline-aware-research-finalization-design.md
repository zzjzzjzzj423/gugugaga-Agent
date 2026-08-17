# Deadline-Aware Financial Research Finalization Design

## Purpose

`simple_cc` is treated as a financial-research Agent. Its primary reliability requirement is that every run that remains alive and can write its output directory produces a report before the external 1,800-second benchmark timeout. A standards-compliant `INSUFFICIENT_EVIDENCE` report is an acceptable terminal result when model or evidence failures prevent a substantive answer.

The current research loop only finalizes when the model voluntarily stops calling tools. On `max_rounds`, provider failure, or an external timeout, the run can end without a final answer. Successful context preservation makes this worse by allowing the model to continue searching for longer. This design adds a lightweight, host-owned phase controller that solves the immediate timeout problem while exposing boundaries that can later become a complete staged workflow.

## Goals

- Finish internally by 1,750 seconds so the benchmark has 50 seconds to persist artifacts and exit before its 1,800-second kill point.
- Make the transition from research to finalization irreversible.
- Remove every tool from finalization and repair requests.
- Bound individual LLM and tool calls by the current phase deadline.
- Reuse successful URL fetches within one run.
- Stop research after sustained lack of new evidence or after sufficient evidence stops improving.
- Return a deterministic `INSUFFICIENT_EVIDENCE` report when model-based finalization cannot safely complete.
- Keep policy state outside the conversation so compaction cannot erase the phase, deadlines, cache, or progress counters.
- Structure the controller so its phases can later map to workflow nodes without rewriting deadline, progress, cache, and fallback logic.

## Non-Goals

- Introducing LangGraph or another workflow framework now.
- Adding Supervisor or parallel Researcher agents.
- Adding a cross-run or persistent Web cache.
- Hard-blocking semantically similar searches using embeddings.
- Automatically deriving factual conclusions for deterministic fallback output.
- Optimizing for the 1,200-second efficiency score in this change.

## Architecture

Add `simple_cc/research_control.py` as a pure control module. It owns no provider, tool, filesystem, or recorder implementation.

```text
ResearchController
├── ResearchPhase
├── ResearchAction
├── ResearchControlConfig
├── ResearchProgress
├── DeadlinePolicy
├── StopPolicy
├── FetchCache
└── deterministic_fallback
```

The phases are:

```text
RESEARCHING → FINALIZING → REPAIRING → DONE
```

Allowed actions are:

```text
CONTINUE_RESEARCH
FINALIZE_WITHOUT_TOOLS
REPAIR_WITHOUT_TOOLS
RETURN_FALLBACK
DONE
```

`agent_loop()` remains the executor. Before each LLM call and tool call it asks the controller whether the call fits in the current phase. After research tool results it updates progress. At every loop boundary it asks for the next action. Once the controller leaves `RESEARCHING`, no transition may return to that phase.

Future workflow nodes can use the same boundaries:

```text
RESEARCHING       → Research node
ResearchProgress  → Evidence Reducer state
FINALIZING        → Final Writer node
REPAIRING         → Validator/Repair node
FetchCache        → run-scoped workflow resource
```

## File Responsibilities

- `simple_cc/research_control.py`: phases, actions, deadlines, progress, stop decisions, URL cache, authority classification, and deterministic fallback formatting.
- `simple_cc/agent.py`: integrates the controller, constructs finalization and repair messages, executes provider/tool calls, validates final text, records transitions, and returns `AgentLoopOutcome`.
- `simple_cc/provider.py`: supports a caller-supplied absolute deadline, per-attempt timeout, and maximum attempts. Retries never start unless they fit before the supplied deadline.
- `simple_cc/evidence.py`: continues to canonicalize URLs, register successfully fetched sources, link citations, and validate final answers.
- `simple_cc/config.py`: exposes the timing, call, stagnation, evidence coverage, and authority-domain defaults.
- `simple_cc/web_research.py`: retains network and content extraction behavior. It does not own a global cache.
- `simple_cc/telemetry.py` and the existing recorder interface: record phase, stop, cache, progress, finalization, repair, and fallback events.
- `eval/run_task.py`: includes final research-control metrics in the manifest and writes every completed fallback or substantive answer through the existing completed path.

## Deadline Policy

The default budget begins when `agent_loop()` starts and uses `time.monotonic()` through an injectable clock.

| Phase | Time window | Rule |
|---|---:|---|
| Research | 0–1,450 seconds | Research LLM and tool calls are allowed only when their bounded timeout fits. |
| Finalization | 1,450–1,580 seconds | One tool-free model attempt, with at most 120 seconds for the whole attempt. |
| Repair | 1,580–1,690 seconds | One tool-free repair attempt, with at most 90 seconds for the whole attempt. |
| Deterministic fallback | At or after 1,690 seconds | No model or tool call; build the report immediately. |
| Internal completion | Before 1,750 seconds | Return `completed`; reserve 50 seconds for persistence and process exit. |
| External timeout | 1,800 seconds | Benchmark kill point, outside Agent control. |

Deadlines are absolute monotonic timestamps, not independent relative timeouts. A call receives the lesser of its configured timeout and the phase's remaining time. A provider retry may start only if it fits before the same absolute deadline. The current unconditional four attempts at 120 seconds each must not be possible when a phase deadline is present.

Default call ceilings are:

- `web_search`: 15 seconds.
- `web_fetch`: 30 seconds.
- `pdf_fetch`: 60 seconds.
- Other foreground tools: 120 seconds.
- Finalization LLM: one attempt, 120 seconds total.
- Repair LLM: one attempt, 90 seconds total.

Background work may not be launched if its declared maximum duration does not fit before the research deadline. Entering `FINALIZING` disables all new foreground and background tools.

The following conditions all transition to `FINALIZING` rather than returning an empty or failed outcome:

- Research deadline reached.
- `max_rounds` reached.
- Provider retry budget exhausted during research.
- A new research call cannot safely fit before the research deadline.
- Search stagnation threshold reached.
- Sufficient evidence followed by one stagnant round.
- Research LLM/tool exception from which the process can still recover.
- Maximum-token recovery exhausted.

## Run-Scoped Fetch Cache

The cache key is `(tool_name, canonical_url, cutoff)`. URL normalization reuses `evidence.canonicalize_url()`.

- Successful `web_fetch` and `pdf_fetch` results remain cached for the current run.
- Deterministic failures such as `unsafe_url`, `post_cutoff`, and `date_conflict` remain cached for the current run.
- Transient network, DNS, timeout, and provider failures are not permanent cache entries, but a canonical URL receives at most one retry after its first transient failure.
- A cache hit returns the original tool result without network I/O.
- A cache hit records `tool_cache_hit` but does not create a duplicate artifact or duplicate source registration.
- Cache lifetime ends with the current `agent_loop()` invocation.

Exact normalized search queries may reuse their previous search result or return a controlled duplicate-query result. The first implementation does not hard-block semantically similar queries.

## Research Progress and Stop Policy

`ResearchProgress` stores:

- Research rounds and LLM call count.
- Total research tool, search, and fetch call counts.
- Candidate URLs observed in search results.
- Successfully fetched canonical URLs.
- Independent source domains.
- Authoritative source URLs.
- Consecutive stagnant research rounds.
- Consecutive stagnant research tool calls.
- Cache hit and transient retry counts.
- The terminal stop reason.

A meaningful evidence advance is at least one of:

- A new canonical URL is fetched successfully.
- A new independent domain is added.
- A new authoritative source is added.
- A new source is successfully registered as citable evidence.

New search snippets alone do not count as evidence progress. Cache hits do not count as progress.

The default stop conditions are:

- Two consecutive research rounds without meaningful evidence progress.
- Eight consecutive research tool calls without meaningful evidence progress.
- An exact normalized query is repeated.
- A cached URL is requested repeatedly without any other progress.
- Evidence coverage is sufficient and the following research round adds no meaningful evidence.

Evidence coverage is sufficient when all are true:

```text
successfully fetched sources >= 2
independent domains >= 2
authoritative sources >= 1
```

`AuthorityPolicy` uses configurable exact domains and domain suffixes for government and regulatory bodies, central banks, exchanges, company investor-relations/official announcement pages, official statistical agencies, and international financial organizations. The policy is deterministic and inspectable. It does not call an LLM.

If coverage is insufficient when a stop or deadline condition fires, the Agent still finalizes with available evidence. The validator and deterministic fallback decide how much can safely be returned.

## Tool-Free Finalization and Repair

The finalization request uses no tool definitions and disables thinking where supported. It receives:

- The original user question captured outside compacted history.
- The compacted current research context.
- Successfully registered source URLs and identifiers.
- Evidence coverage statistics.
- Unresolved evidence gaps that the host can state safely.
- The reason research stopped.

The prompt requires the model to write the best supportable financial report, cite only registered URLs, state uncertainty, and avoid invented facts, figures, dates, and sources. Asking for or emitting a tool call is treated as finalization failure.

The first final answer passes through `validate_research_final()`. A validation failure transitions once to `REPAIRING`. The repair request contains the draft, exact validation errors, registered URLs, and remaining time. It also has no tools. A second validation failure transitions directly to deterministic fallback.

## Deterministic Fallback

Fallback never calls a model. It returns `AgentLoopOutcome("completed", report)` with a stable report containing:

- `INSUFFICIENT_EVIDENCE`.
- The original research question.
- A conservative research-status section.
- Successfully verified source URLs.
- Unresolved evidence gaps.
- The stop or failure reason.

The host must not synthesize factual conclusions from arbitrary tool output. If no safe conclusion is available, the research-status section says only what was attempted and what evidence was registered.

Fallback is used when finalization or repair times out, errors, returns empty text, emits tool calls, fails validation, or reaches the fallback deadline. Research-stage provider failures should first transition to finalization; if the provider remains unavailable, fallback still creates a report.

Only failures that prevent output creation itself remain `failed`, such as process bootstrap failure, inability to create the run directory, or disk write failure.

## Telemetry and Manifest Data

Add recorder events:

- `research_phase_transition`
- `research_stop_decision`
- `research_call_rejected`
- `tool_cache_hit`
- `research_progress_updated`
- `finalization_started`
- `finalization_failed`
- `repair_started`
- `fallback_generated`

The final manifest records phase durations, stop reason, cache hits, stagnant counts, evidence coverage counts, finalization/repair attempts, whether fallback was used, and internal elapsed time. Existing tool and source telemetry remains intact.

## Testing Strategy

Add `tests/test_research_control.py` for pure policy tests with an injected fake monotonic clock, and `tests/test_research_finalization.py` for Agent integration using `ScriptedProvider` and fake handlers. Extend provider, Web research, evidence trace, run-task, and benchmark tests where their existing boundary is the correct location.

Required cases include:

- The 1,450-second research deadline causes an irreversible finalization transition.
- Finalization and repair requests contain `tools=[]`.
- `max_rounds` produces a report instead of an empty outcome.
- Provider retries cannot cross an absolute phase deadline.
- LLM and tool timeouts are clamped to remaining phase time.
- The same canonical URL invokes the underlying fetch handler once.
- Deterministic fetch failures are cached; transient failures receive at most one retry.
- Two stagnant rounds or eight stagnant research tool calls finalize.
- Sufficient coverage followed by one stagnant round finalizes.
- A failed finalization receives at most one repair.
- Failed repair returns a completed `INSUFFICIENT_EVIDENCE` report.
- A completely unavailable provider still yields a completed fallback report.
- A benchmark integration test with tiny injected deadlines writes `final_answer.txt` before its external timeout.

## Acceptance Criteria

For the 20-task FinanceGym batch under an 1,800-second external timeout:

```text
completed = 20/20
timed_out = 0/20
final answer files = 20/20
internal elapsed time per task < 1,750 seconds
tool calls during finalization and repair = 0
repair attempts per task <= 1
```

Quality is reported separately rather than hidden by completion results. The batch comparison must include substantive report count, deterministic fallback count, registered sources, independent domains, authoritative source count, result-layer score, stop-reason distribution, and cache hits.

The implementation is complete only when the focused tests and full existing test suite pass, and a small-deadline end-to-end benchmark simulation proves that the worker writes a final answer before its supervisor timeout.
