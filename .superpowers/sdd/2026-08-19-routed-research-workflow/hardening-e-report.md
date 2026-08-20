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
- Enables a research-only strict allowlist check before cutoff processing, permissions, hooks, background dispatch, or handler lookup. Hallucinated non-stage tools receive a synchronous `tool_not_available` result and cannot execute.
- Leaves the ordinary route on the complete runtime definitions/handlers and the default non-strict loop behavior.

Call chain after the change:

`SourceRuntime.run_turn(research)` -> `_research_tool_view(runtime tables)` -> `research_execution_prompt(tool_names)` + `agent_loop(tools, handlers, strict_tool_allowlist=True)` -> foreground-only `web_search` / `web_fetch` / `pdf_fetch` configured intersection -> research gate.

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
