# Hardening E: research stage isolation and writing completion trace

## Baseline and scope

- Verified clean linked-worktree HEAD before edits: `33a12d7d69ccdb80f00a769c65bf80809c752b7a`.
- Production files changed: `simple_cc/agent.py`, `simple_cc/research_workflow.py`, and (after controller approval) `simple_cc/prompts.py`.
- Tests changed: `tests/test_agent_loop_source.py`, `tests/test_research_workflow.py`.
- The accepted scanner limitation for a literal embedded `http(s)://` inside one URL path/query was not touched.

## Root-cause evidence

### Research execution

`SourceRuntime.run_turn()` built the research executor but passed `self.tool_definitions` and `self.tool_handlers` unchanged to `agent_loop()`. With the default runtime those tables include `bash`, file mutation, task/team, cron, and background-capable tools. `agent_loop()` also evaluated the special background branch by the provider-returned name, so a hallucinated `bash` call could reach `start_background_task()` even when it should not have existed in the research stage.

The prompt had a separate mismatch: `research_execution_prompt()` delegated to `assemble_system_prompt()`, whose tools section was a hard-coded complete tool list. The runtime state could therefore not make the research-stage claim match an isolated executor.

### Writing trace

`ResearchWorkflow._run_forward()` emitted `writing_attempt_started` / `writing_repair_started`, called `write()` / `rewrite()`, and moved directly to `writing_gate`. There was no code-owned event at provider-call completion and no bounded failed-attempt trace before re-raising provider errors.

## Implementation

### `simple_cc/agent.py`

- Imports the existing `simple_cc.evidence.RESEARCH_TOOLS` as the single allowlist source.
- Builds a deterministic definition/handler key intersection from the current runtime; it never reinjects a missing global tool.
- Passes that same intersection to the research `agent_loop()` and its prompt.
- Enables the explicit research-isolated execution policy before cutoff processing, permissions, hooks, background dispatch, or handler lookup. Hallucinated non-stage tools receive a synchronous `tool_not_available` result and cannot execute.
- Leaves the ordinary route on the complete runtime definitions/handlers and the default non-strict loop behavior.

Call chain after the change:

`SourceRuntime.run_turn(research)` -> `_research_tool_view(runtime tables)` -> `research_execution_prompt(tool_names)` + `agent_loop(tools, handlers, execution_policy=RESEARCH_ISOLATED)` -> foreground-only `web_search` / `web_fetch` / `pdf_fetch` configured intersection -> research gate.

### `simple_cc/prompts.py`

- Adds an optional explicit `tool_names` override to prompt assembly.
- Research execution supplies the isolated names; ordinary/default callers retain the existing complete tools text.

### `simple_cc/research_workflow.py`

- Adds `writing_attempt_finished` immediately after each successful writing provider call and before its independent `writing_gate`.
- Emits one event for attempt 1, and an attempt 2 event only when rewrite actually starts.
- On non-trace provider errors, records a failed event best-effort and re-raises the identical provider exception.
- Bounds and whitespace-normalizes failure class/message (128/512 characters); the event contains only `attempt`, `repair`, `status`, `failure_class`, and `failure_message`.
- Preserves `TraceWriteError` identity without wrapping or swallowing; a trace failure while writing the new completion event also stops before the writing gate.

## TDD evidence

- RED: corrected focused selection produced `8 failed, 1 passed`; failures showed the complete research table, an actual hallucinated `bash` reaching the background branch, and the absence of writing completion events. The existing TraceWriteError identity expectation passed.
- GREEN: focused selection produced `10 passed`.
- Affected modules: `123 passed`.
- Trace-event infrastructure edge: `1 passed`.

Covered behaviors include custom-runtime intersection, default SourceRuntime tables, ordinary full-table compatibility, prompt/runtime name equality, synchronous forbidden-tool rejection without handler/background execution, successful single writing, one rewrite, provider exceptions for attempts 1 and 2, bounded safe failure payloads, and exact `TraceWriteError` propagation.

## Verification

- Final full suite including the trace-edge regression: `441 passed, 1 deselected`.
- Approved linked-worktree deselection only: `tests/test_source_audit.py::test_source_map_pins_baseline_and_classifies_every_top_level_source_block`.
- `python -m compileall -q simple_cc eval tests`: exit 0.
- `git diff --check`: clean (Git emitted only configured LF-to-CRLF warnings).

Clean-status and commit evidence is recorded in the task handoff after commit.

## Fix Round 1: close remaining isolation gaps

### Baseline and review findings

- Verified exact clean baseline `535e2ccb98568de83fdfd36fc95df77f9547b8e9` before the fix round.
- The independent review found four gaps: conditional prompt content still named unavailable tools and loaded skills/memory; the research agent loop still shared memory/cron/background/todo and model compaction; malformed/duplicate tool definitions were not fail-closed; and a failed-event `TraceWriteError` could replace the provider exception.
- Production scope expanded, as approved, only to `simple_cc/context.py` for explicit model-summary suppression and bounded local compaction.
- The accepted embedded-scheme scanner limitation remained untouched.

### Final implementation

- Research prompt rules are generated from the exact validated tool tuple. Cutoff instructions enumerate only present tools, and PDF instructions exist only when `pdf_fetch` exists. The research profile neither scans/lists skills nor injects memories; ordinary prompt defaults remain unchanged.
- `AgentExecutionPolicy.RESEARCH_ISOLATED` now owns the complete stage boundary. It requires an explicit research system prompt, forces memory off, does not drain cron/background queues, does not inject notifications or todo reminders, does not mutate shared todo state, and bypasses background-result collection at the end of tool rounds.
- Proactive and reactive research compaction call only local bounded transforms (`allow_model_summary=False`). They retain a complete recent tail where possible and return controlled `LocalContextLimitExceeded` without another provider call if the newest complete unit cannot fit. Ordinary compaction defaults remain model-enabled.
- `_research_tool_view()` validates dict shape, non-empty allowed string names, string descriptions, dict schemas, callable configured handlers, and duplicate consistency before exposing a tool. Malformed/unhashable entries are skipped; a conflicting same-name occurrence excludes that name. Research prompt construction no longer touches `state_builder()` or the unvalidated full registry.
- When a provider error and failed-finished trace write both fail, the identical provider exception remains primary, receives a bounded audit note when supported, and has the `TraceWriteError` as `__cause__`. Standalone trace failures and direct write/rewrite `TraceWriteError` objects still propagate unchanged.

### Fix Round 1 TDD and verification

- Initial focused RED: `13 failed, 3 passed`, with failures covering all four review findings. The three passes characterized already-correct direct trace identity and safe identical-duplicate/empty behavior.
- Additional explicit-prompt RED: `1 failed` before the fail-closed system-prompt requirement.
- Focused GREEN after all safeguards: `20 passed`.
- Affected modules: `171 passed`.
- Full suite: `457 passed, 1 deselected` using only the approved linked-worktree source-map deselection.

### Fix Round 1 file-by-file scope

- `simple_cc/agent.py`: validates the configured research-tool intersection, constructs the prompt from that validated view, and enforces the explicit isolated execution policy around memory, queues, todo state, tool dispatch, and compaction.
- `simple_cc/context.py`: adds the opt-out from model-generated summaries plus a bounded local-only compactor; existing callers retain model compaction by default.
- `simple_cc/prompts.py`: conditionally renders research rules from the exact tool tuple and disables skill/memory catalogs only for the research execution profile.
- `simple_cc/research_workflow.py`: preserves the provider exception as primary when writing its failed-attempt event also raises `TraceWriteError`.
- `tests/test_agent_loop_source.py`: covers malformed and conflicting registries, default-runtime isolation, unavailable-tool rejection, memory/queue/todo separation, and proactive/reactive/local-limit behavior.
- `tests/test_context_prompts.py`: covers conditional tool rules, empty intersections, and the absence of research skill/memory catalog work.
- `tests/test_research_workflow.py`: covers attempt 1/2 provider-plus-trace double failures and direct `TraceWriteError` identity.
- `.superpowers/sdd/2026-08-19-routed-research-workflow/hardening-e-report.md`: records the review findings, RED/GREEN evidence, implementation boundaries, and final verification.

## Fix Round 2: enforce the policy at its own boundary

### Baseline and root cause

- Verified exact clean baseline `8be4c6177c8a51f449b1da6f0a6929c0ba25a525` before edits.
- `agent_loop(RESEARCH_ISOLATED)` derived an availability-name set from its input but retained and exposed the original definitions and handlers. A direct caller using omitted or complete default tables therefore exposed all tools and, when permission was approved, could execute `bash`; malformed tables also crashed during the unchecked name comprehension.
- `research_execution_prompt(tool_names=None)` still selected the complete ordinary research text, and non-research names were rendered without validation or stable deduplication.
- After an ordinary provider exception, `_record_terminal_failure()` re-raised a terminal `TraceWriteError`, replacing both the primary provider object and any earlier failed-finished trace cause.
- Production scope remained `simple_cc/agent.py`, `simple_cc/prompts.py`, and `simple_cc/research_workflow.py`. `simple_cc/context.py` and the accepted embedded-scheme scanner limitation were not touched.

### Implementation and call-chain evidence

- `agent_loop()` now resolves passed/default tables and immediately replaces both with `_research_tool_view()` snapshots under `RESEARCH_ISOLATED`, before provider, permission, hook, background, or handler logic. The provider sees only validated configured `RESEARCH_TOOLS`; unavailable names have no retained handler and receive the synchronous stage rejection.
- `research_execution_prompt()` normalizes omitted/`None` tool names to the empty tuple and accepts only exact non-empty names from the shared `RESEARCH_TOOLS` source, preserving first-seen order while removing duplicates. It always passes this safe tuple into prompt assembly, so it cannot fall back to the complete tool catalog.
- `_record_terminal_failure()` now returns a terminal trace failure to `run()`. With no existing cause, `run()` re-raises the identical provider exception from that trace error. With an existing failed-finished trace cause, it preserves the first cause and adds one whitespace-normalized, bounded terminal note containing the trace type and message. Direct/standalone `TraceWriteError` behavior is unchanged.

The resulting research boundary is:

`SourceRuntime` or direct caller -> `agent_loop(RESEARCH_ISOLATED)` -> `_research_tool_view(passed/default tables)` -> filtered definitions + filtered handler snapshot -> provider/tool dispatch.

### File-by-file scope

- `simple_cc/agent.py`: enforces the validated research snapshot inside the isolated policy itself and safely rejects malformed top-level tables.
- `simple_cc/prompts.py`: adds fail-closed research tool-name normalization using `RESEARCH_TOOLS`.
- `simple_cc/research_workflow.py`: merges terminal audit failures without replacing the provider-primary exception or its first trace cause.
- `tests/test_agent_loop_source.py`: exercises direct omitted/default, explicit-full, bash-only, and malformed policy inputs, including a permission-approved forbidden handler.
- `tests/test_context_prompts.py`: exercises omitted, `None`, non-research, and duplicate prompt tool names.
- `tests/test_research_workflow.py`: exercises attempt 1/2 persistent failed-finished plus terminal trace failures and the terminal-only trace-failure cause path.
- `.superpowers/sdd/2026-08-19-routed-research-workflow/hardening-e-report.md`: records Fix Round 2 root cause, scope, evidence, and verification.

### TDD and verification

- Focused RED: `9 failed`, covering every newly reviewed path; the direct complete-table cases visibly reached the approved forbidden `bash` handler before the fix.
- Focused GREEN: `9 passed`.
- Directly affected test files: `134 passed`.
- Affected regression suite: `180 passed`.
- Full suite: `466 passed, 1 deselected`; the only deselection was the approved source-map baseline test, using a unique external basetemp.
- `python -m compileall -q simple_cc eval tests` and baseline-range `git diff --check` are required final gates; commit and clean-status evidence is included in the handoff.

## Fix Round 3: preserve acyclic failure chains

### Baseline and root cause

- Verified exact clean baseline `00c58c69134b33dbde27bf6da69197d49475c05e` before edits.
- `_call_writing_attempt()` performed failed-finished trace I/O inside the active provider `except`; `run()` did the same for terminal trace I/O. A terminal-only failure therefore produced a traversable provider/trace context cycle. Even where CPython later removed a direct back-reference during re-raise, both preconstructed and real recorder failures observed the provider as `sys.exception()` while audit I/O ran, violating the stage's exception boundary and allowing underlying `OSError` context to originate under the provider failure.
- `_research_tool_view()` and research prompt normalization used `isinstance(name, str)` before set membership. An unhashable `str` subclass passed that guard and raised `TypeError` at the membership check.
- Production scope remained `simple_cc/research_workflow.py`, `simple_cc/agent.py`, and `simple_cc/prompts.py`. `simple_cc/context.py` and the accepted scanner limitation were not touched.

### Implementation and exception-chain evidence

- `_call_writing_attempt()` now captures only the identical non-trace provider exception and its traceback inside the provider `except`, exits that scope, then attempts the failed-finished event. `_record_writing_failure()` likewise catches and returns `TraceWriteError`, exiting the trace `except` before the provider is re-raised with its original traceback.
- `run()` applies the same boundary: it saves the workflow exception and traceback, exits the provider `except`, captures any terminal trace failure through the existing return path, then re-raises outside both exception contexts. With no first cause the terminal trace becomes the cause; with a failed-finished cause it remains first and the terminal failure is represented only by the bounded typed note.
- Tests traverse both `__cause__` and `__context__` with gray/visited sets. Attempt 1 and 2 are covered for failed-finished-only, persistent failed-finished plus terminal failure, and terminal-only failure. Every case runs once with a preconstructed `TraceWriteError` and once through real `TraceRecorder.record()` translation of a forced `_append_line()` `OSError`; the final graphs are acyclic, trace/OSError subgraphs do not refer back to the provider, provider identity is unchanged, and the original traceback tail remains reachable.
- Both tool-name boundaries now require `type(name) is str` before truth, membership, or deduplication checks. Unhashable `str` subclasses are skipped without hashing.

### File-by-file scope

- `simple_cc/research_workflow.py`: separates provider capture, trace capture, audit recording, and final re-raise into non-overlapping exception contexts while preserving identity, traceback, first cause, and bounded terminal notes.
- `simple_cc/agent.py`: rejects `str` subclasses before research allowlist membership.
- `simple_cc/prompts.py`: rejects `str` subclasses before research allowlist membership and stable deduplication.
- `tests/test_research_workflow.py`: adds the exception-graph walker, active-exception probes, original-traceback checks, and preconstructed/real-recorder matrices for attempts 1 and 2.
- `tests/test_agent_loop_source.py`: covers the unhashable tool-name subclass at the runtime research view.
- `tests/test_context_prompts.py`: covers the same boundary at research prompt normalization.
- `.superpowers/sdd/2026-08-19-routed-research-workflow/hardening-e-report.md`: records Fix Round 3 root cause, scope, RED/GREEN evidence, and verification.

### TDD and verification

- Initial graph-only RED: `6 failed, 8 passed`; terminal-only cases exposed the cycle and both tool-name boundaries raised `TypeError`. CPython had already removed some direct failed-finished back-references during re-raise, which motivated an explicit active-exception boundary assertion.
- Strengthened focused RED: `14 failed`; all twelve attempt/mode trace cases observed the provider exception during audit I/O, plus the two unhashable-name failures.
- Focused GREEN: `14 passed`.
- Directly affected test files: `143 passed`.
- Affected regression suite: `189 passed`.
- Full suite: `475 passed, 1 deselected`; the sole deselection was the approved source-map baseline test, using a unique external basetemp.
- Final compile, baseline-range diff, commit, and clean-status evidence is included in the handoff.
