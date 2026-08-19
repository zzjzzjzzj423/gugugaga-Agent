# Routed Research Workflow Design

**Date:** 2026-08-19

**Status:** Approved in conversation; user requested that implementation planning begin without a separate document-review round.

## 1. Goal

Split the runtime into two explicitly routed task paths:

- ordinary tasks keep the existing agent loop without financial-research gating;
- research tasks run a bounded plan, research, research-gate, writing, and writing-gate workflow.

The research workflow must choose one of three fixed research ranks, permit at most one supplemental-research attempt, permit at most one writing repair, and apply the existing evidence/citation gate only after research writing.

## 2. Scope

This change covers task routing, stage-specific prompts, fixed research ranks, evidence records, research sufficiency evaluation, writing-only finalization, dynamic final validation, retry limits, tracing, and benchmark propagation of `task_type`.

This change does not introduce parallel subagents, teammate orchestration, a new workflow dependency, or network-dependent tests. “Research collaboration” means an orchestrated set of research directions in this version. The research executor remains replaceable so directions may be parallelized later without changing the outer workflow contract.

## 3. Current State and Problem

`SourceRuntime.run_turn()` currently receives `run_metadata` but discards it. The benchmark worker records `task_type` in the manifest, yet the runtime never uses it for routing.

The current evidence gate runs whenever `run_context` exists. Consequently, every recorded task is treated as financial research. The system prompt is also globally specialized for financial research. This couples tracing to research semantics and prevents ordinary recorded tasks from using the existing loop normally.

The current gate enforces two registered sources, two independent domains, at least one fetched citation, and no unfetched citation. It may ask the same tool-enabled loop to repair once. The new design moves this validation to the research workflow after a separate writing stage and makes its quantitative thresholds depend on the selected rank.

## 4. Architectural Decision

Add a thin workflow orchestrator around the retained `agent_loop()` executor.

```text
upstream task_type
├── normal / unknown / missing
│   └── existing agent_loop with ordinary-task prompt and no evidence gate
└── research / research_analysis
    ├── tool-free research planning
    ├── bounded research execution
    ├── hybrid research gate
    ├── optional supplemental research (at most once)
    ├── tool-free writing
    ├── deterministic writing gate
    ├── optional tool-free rewrite (at most once)
    └── completed report or INSUFFICIENT_EVIDENCE
```

The orchestrator owns stage transitions, budgets, retry counters, and terminal behavior. The retained loop continues to own model/tool turns, permission checks, context management, compaction, provider recovery, memory behavior, and tool tracing.

## 5. Task Routing

Upstream explicitly marks task type. The runtime does not ask an LLM to classify the task.

Normalize task types as follows:

- `research` and `research_analysis` select the research workflow;
- `normal`, unknown values, and a missing value select the ordinary path.

The default is deliberately ordinary so an absent or novel metadata value cannot accidentally trigger an expensive research run.

`SourceRuntime.run_turn()` must stop discarding `run_metadata`. Benchmark execution must pass `task.task_type` through the runtime boundary. Interactive callers may omit it and therefore use the ordinary path.

## 6. Prompt Profiles

Replace the global financial-research identity with stage-specific prompt profiles:

- the ordinary profile describes a general workspace agent and preserves the current loop capabilities;
- the research-planning profile requests strict structured output and exposes no tools;
- the research-execution profile includes the financial-research contract, cutoff, selected rank, source requirements, directions, remaining rounds, and any gate gaps;
- the research-gate profile evaluates direction coverage and authority only from registered evidence;
- the writing profile receives the question, cutoff, plan, evidence packet, and unresolved research gaps, and exposes no tools;
- the rewrite profile additionally receives deterministic writing-gate errors and exposes no tools.

Ordinary tasks are not subject to research source quotas, citation validation, or research-only cutoff enforcement. Research-tool cutoff injection remains active inside research execution.

## 7. Fixed Research Ranks

The planner selects one rank; it does not invent numeric budgets.

| Rank | Maximum research rounds | Distinct sources | Authoritative sources | Research directions |
|---|---:|---:|---:|---:|
| `light` | 10 | 2 | 1 | 1 |
| `standard` | 20 | 3 | 1 | 2 |
| `deep` | 30 | 4 | 2 | 3 |

The maximum research rounds are a single total budget shared by initial and supplemental research. Planner calls, gate calls, writing calls, and rewrite calls do not consume research rounds.

The rank table is code-owned and immutable at runtime. The plan stores the selected rank and the resolved values for tracing and convenient use, but resolved values must always originate from this table.

## 8. Research Planning

The planner makes one tool-free LLM call and requests an object with this logical shape:

```json
{
  "rank": "standard",
  "directions": ["direction one", "direction two"],
  "reason": "brief rank-selection rationale"
}
```

Validation requires:

- rank is exactly `light`, `standard`, or `deep`;
- direction count exactly matches the selected rank;
- every direction is a non-empty string;
- directions are unique after whitespace and case normalization.

Malformed, incomplete, or semantically invalid planner output is not retried. It falls back to `standard` with two deterministic generic directions: primary facts and first-party evidence; impact, risk, and independent corroboration. The trace records the raw output, validation errors, fallback decision, and normalized plan.

Provider-level transient retry behavior remains the responsibility of the existing provider/recovery layer. An unrecoverable provider failure retains the runtime’s normal failed outcome semantics.

## 9. Evidence Model

Replace the URL-to-source-ID-only research registry with a structured evidence record while preserving URL canonicalization and deterministic source IDs.

Each successfully fetched source records at least:

- source ID;
- canonical URL;
- independent domain;
- title when available;
- bounded model-visible content or summary used by later stages;
- publication date and date status when available;
- cutoff;
- source tool name;
- artifact references already produced by tracing;
- authority decision and reason after research-gate evaluation.

Search snippets remain leads and never become evidence records. Failed fetches, rejected post-cutoff material, date conflicts, and unparseable results do not count toward source quotas.

Initial and supplemental research share one evidence registry. Canonical URLs are deduplicated. The writer receives an explicit bounded evidence packet derived from this registry so writing does not depend on mutable or compacted conversation history.

## 10. Research Execution

The research executor reuses the retained agent loop with a research-stage prompt and research-capable tool table. It works through the plan’s directions and may stop before consuming its rank budget.

The workflow tracks actual research rounds consumed. If the first research gate fails and rounds remain, it invokes supplemental research exactly once with:

- the original question and cutoff;
- the original plan and directions;
- the shared evidence registry;
- the research-gate gaps;
- the number of remaining research rounds.

Supplemental research cannot reset or exceed the rank’s total round budget. If no rounds remain, the workflow records budget exhaustion and proceeds as if the single supplemental opportunity were unavailable.

Research-stage terminal text is treated as research notes, not as the user-facing final answer.

## 11. Research Gate

The research gate is hybrid.

Code performs hard checks for:

- successfully registered distinct source count;
- independent domain count, using the rank’s distinct-source target as the domain target.

A tool-free LLM call evaluates:

- whether every planned direction is adequately covered by registered evidence;
- which registered sources are authoritative for this question;
- a reason for every authority decision;
- evidence-backed gaps that supplemental research or writing must address.

Government bodies, regulators, exchanges, official company disclosures, filings, and original data publishers normally qualify as authoritative. News republication and aggregation normally do not. Authority is contextual and therefore evaluated from the question, URL, title, and bounded evidence content.

The gate must return only source IDs that exist in the evidence registry. Unknown IDs, invalid JSON, missing direction decisions, or missing authority reasons make the semantic gate fail closed. Code combines the hard and semantic results; passage requires all directions covered, the source and domain targets met, and the authoritative-source target met.

The gate runs after initial research and again after supplemental research when supplemental research occurs. A second failure never triggers a third research execution. Its gaps are passed to writing.

## 12. Writing

Writing is a distinct, tool-free stage. Its input consists of:

- original question;
- cutoff;
- normalized research plan and fixed rank requirements;
- structured evidence packet;
- authority decisions;
- remaining research-gate gaps.

The writer must answer from supplied evidence, cite exact fetched URLs, distinguish fact from inference, and disclose unresolved limitations. It cannot search, fetch, call shell commands, or mutate files.

## 13. Writing Gate

Move the current research final-answer gate out of the generic loop and run it only after research writing.

The deterministic gate requires:

- registered distinct sources greater than or equal to the rank target;
- independent domains greater than or equal to the rank’s distinct-source target;
- research-gate-confirmed authoritative sources greater than or equal to the rank target;
- at least one final-answer citation linked to a fetched registered source;
- no final-answer URL that was not fetched and registered.

The first writing-gate failure invokes one tool-free rewrite with the exact gate errors and the same evidence packet. It never returns to research. A second failure returns:

```text
INSUFFICIENT_EVIDENCE

Research finalization failed:
- <specific validation error>
```

Research retry and writing retry counters are independent. Each permits at most one retry.

If the second research gate failed only on semantic direction coverage while all deterministic writing checks pass, the writer may still produce a completed report that explicitly discloses the limitation. Failure of deterministic source, authority, or citation requirements eventually produces the controlled insufficient-evidence result.

## 14. Outcomes and Error Handling

Ordinary tasks preserve existing `completed`, `failed`, `max_rounds`, and recovery behavior.

Research workflow outcomes follow these rules:

- unrecoverable provider or runtime exceptions return the existing failed outcome with a safe failure class and message;
- malformed planner output uses the deterministic standard fallback;
- malformed research-gate output fails the gate and produces actionable gaps;
- exhausted research rounds prevent further research but do not prevent writing;
- malformed or failed writing output is processed by the one rewrite opportunity when possible;
- a deterministic second writing-gate failure returns `INSUFFICIENT_EVIDENCE` as a completed, controlled result, matching current benchmark semantics.

No stage may create an unbounded loop. Phase transitions are forward-only except for the single supplemental-research and single rewrite edges.

## 15. Tracing

Add phase-aware events sufficient to reconstruct the workflow without inspecting in-memory messages:

- task routing decision and normalized task type;
- planner request/result, raw output artifact, normalized plan, and fallback reason;
- research attempt start/end, round budget before/after, and directions;
- evidence registration with structured metadata;
- research-gate hard result, semantic result, authority decisions, gaps, and pass/fail decision;
- supplemental-research admission or skip reason;
- writing and rewrite attempt start/end;
- writing-gate errors and pass/fail decision;
- final workflow outcome and terminal reason.

Existing model, tool, cutoff, permission, source, artifact, and final-answer tracing remains intact. Sensitive-value redaction applies to all new payloads.

## 16. File Boundaries

Create `simple_cc/research_models.py` for fixed ranks and immutable plan, evidence, gate, budget, and workflow-result data structures.

Create `simple_cc/research_workflow.py` for planner parsing, orchestration, research-gate coordination, writing, retry transitions, and phase tracing.

Modify `simple_cc/agent.py` to accept an injected prompt/profile and gate-free execution mode, report consumed rounds/research notes to the orchestrator, preserve the ordinary path, and route from `SourceRuntime.run_turn()`.

Modify `simple_cc/evidence.py` to build structured evidence records and validate a final report against dynamic rank requirements and authority decisions while retaining canonicalization and citation linkage.

Modify `simple_cc/prompts.py` to provide ordinary and stage-specific research prompts.

Modify `eval/run_task.py` and the benchmark/runtime boundary so `task_type` reaches `SourceRuntime.run_turn()` instead of stopping at manifest metadata.

Add focused workflow and model tests. Update existing evidence-gate tests so the generic loop is no longer expected to repair or reject ordinary recorded outputs.

## 17. Testing

All tests use scripted providers and local fake tools. Default tests perform no network access.

Required coverage includes:

1. ordinary tasks call only the retained loop and never call planner, research gate, writer, or writing gate;
2. `research` and `research_analysis` route to research while missing and unknown values route to ordinary;
3. each fixed rank resolves to its exact four configured values;
4. invalid rank, incorrect direction count, blank directions, repeated directions, and invalid JSON use the deterministic standard fallback;
5. initial research gate passage skips supplemental research;
6. initial failure admits at most one supplemental attempt and shares evidence and the total round budget;
7. no remaining rounds skip supplemental research and proceed to writing;
8. semantic gate cannot mark an unregistered source authoritative;
9. a second research-gate failure proceeds to writing with gaps;
10. writing and rewrite provider requests expose an empty tool list;
11. dynamic source, domain, authority, matched-citation, and unmatched-citation rules each have passing and failing tests;
12. writing-gate failure permits one rewrite and a second failure returns deterministic `INSUFFICIENT_EVIDENCE`;
13. phase trace order, budget accounting, retry reasons, and final reason are reconstructable;
14. cutoff injection, evidence registration, source linkage, ordinary loop behavior, and benchmark worker behavior remain covered by regression tests.

## 18. Acceptance Criteria

The change is complete when:

- upstream `task_type` deterministically selects ordinary or research execution;
- ordinary recorded tasks no longer receive financial-research prompts or evidence gating;
- a research task produces one valid fixed-rank plan or the documented fallback;
- research never exceeds 10, 20, or 30 total research rounds according to rank;
- research is retried at most once and writing is retried at most once;
- initial and supplemental research share deduplicated structured evidence;
- authority decisions reference only registered sources and contain reasons;
- writing has no tool access;
- the migrated writing gate uses dynamic rank thresholds and existing citation-linkage protections;
- all new and affected tests pass without live network access;
- traces expose every routing, phase, budget, gate, retry, and terminal decision.
