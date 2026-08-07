# Task 4 Report: S12-S14 Durable Scheduling

## Status

Implemented and verified the Task 4 migration from
`learn-claude-code/s20_comprehensive/code.py` into the source-faithful
`simple_cc` module split.

Task 4 adds the S13 background-operation policy and completion queue, the S14
cron model/validation/persistence/scheduler/handlers, and their fixed tool
registrations. The already-approved S12 implementation in `tasks.py` required
no new edit: it already resolves task files dynamically through
`config.TASKS_DIR`, and no worktree fields or handlers were added.

## Scope

Changed Task 4 paths:

- `simple_cc/background.py`
  - Added `is_slow_operation`, `should_run_background`,
    `start_background_task`, and `collect_background_results` with the S20
    signatures and call sequence.
  - Retained `BackgroundManager` only as the behavior-preserving compatibility
    adapter used by the approved pre-migration runtime/tests.
  - Re-exported cron compatibility names from their new owning module.
- `simple_cc/cron.py`
  - Added source-faithful `CronJob`, field matching and validation, durable
    save/load, scheduling/cancellation, scheduler queue/loop, queue consumption,
    and the three cron handlers.
  - Durable operations read `config.DURABLE_PATH` at call time, so
    `configure_workspace` selects their storage root.
  - Retained `CronScheduler` only as the behavior-preserving compatibility
    adapter for the approved pre-migration application API.
- `simple_cc/config.py`
  - Changed both initial and configured durable paths to S20's
    `.scheduled_tasks.json` name.
- `simple_cc/tools.py`
  - Added the fixed `schedule_cron`, `list_crons`, and `cancel_cron` schemas.
  - Bound each schema to its single handler from `cron.py`.
- `tests/test_background_cron_source.py`
  - Added four observable behavior tests covering immediate background return,
    one-time completion delivery, literal five-field cron behavior, selected
    workspace persistence, durable reload, exhausted-job removal, queue drain,
    and fixed handler ownership.

Explicitly unchanged:

- `simple_cc/provider.py`
- worktree and MCP behavior/handlers
- user-owned `.env.example`, `.idea/`, and the untracked plan document
- the known dotenv/model baseline behavior

## TDD RED Evidence

The test was written before the Task 4 production move.

The first sandboxed command was:

```text
python -m pytest tests/test_background_cron_source.py -v
```

It could not reach the test behavior because pytest's Windows temp root was
unreadable inside the sandbox:

```text
PermissionError: [WinError 5] ...
C:\Users\Administrator\AppData\Local\Temp\pytest-of-Administrator
```

Per the task instructions, pytest alone was rerun elevated to bypass that temp
ACL. The expected RED result was:

```text
collected 4 items
4 failed in 0.58s
```

The failures were precisely the missing Task 4 production surface:

- `AttributeError`: `simple_cc.background` had no `is_slow_operation`.
- `ModuleNotFoundError`: `simple_cc.cron` did not exist (three cron tests).

This showed the new tests failed because the requested migration was absent,
not because of a typo or an unrelated failure.

## GREEN Evidence

Initial focused GREEN after the minimal move:

```text
python -m pytest tests/test_background_cron_source.py -v
4 passed in 0.50s
```

Compatibility and fixed-registry verification:

```text
python -m pytest tests/test_background_cron.py \
  tests/test_background_cron_source.py \
  tests/test_workspace_tasks_skills.py \
  tests/test_tools_permissions_hooks.py -v
20 passed in 0.57s
```

Fresh final focused verification after self-review cleanup:

```text
python -m pytest tests/test_background_cron_source.py -v
4 passed in 0.50s
```

All pytest runs that needed `tmp_path` were elevated only because of the known
Windows sandbox temp ACL described above.

## Full-Suite Evidence

Fresh final full-suite command:

```text
python -m pytest -q
```

Result:

```text
1 failed, 65 passed in 0.84s
```

The sole failure is the documented pre-existing baseline:

```text
tests/test_config_provider.py::test_settings_requires_model
Failed: DID NOT RAISE ValueError
```

As recorded in Tasks 2 and 3, `load_dotenv(override=False)` reloads the
repository `.env` model value after the test deletes the process variable.
Task 4 neither modified this behavior nor attempted the prohibited incidental
fix.

Additional checks:

- `python -m compileall -q simple_cc tests/test_background_cron_source.py`:
  pass.
- `git diff --check` over Task 4 tracked paths: pass.
- Provider diff: empty.

## Source-Faithfulness Review

- Background slow-operation keywords, Bash-only gate, counter-based IDs,
  task state, daemon worker, handler dispatch, `PostToolUse` hook, completion
  collection, 200-character summary, and XML notification shape follow the
  S20 flow.
- The immediate placeholder remains an agent-loop responsibility in S20;
  Task 4 verifies that `start_background_task` returns its ID while the handler
  is still running, which is the contract used to build that placeholder.
- `CronJob` retains the S20 fields: `id`, `cron`, `prompt`, `recurring`, and
  `durable`.
- Validation retains five-field bounds, steps, lists, and ranges. Matching
  retains standard DOM/DOW OR semantics and Sunday-as-zero conversion.
- Durable JSON retains S20's `asdict` list format and persists only durable
  jobs to the selected workspace's `.scheduled_tasks.json`.
- A firing one-shot is queued, removed from `scheduled_jobs`, and immediately
  removed from durable storage, retaining the S20 scheduler order.
- The three cron tool definitions are fixed entries with exactly one handler
  each. No late-bound handler tables, worktree tools, or MCP tools were added.
- `provider.py` remains unchanged.

## Test Quality / Mutation Review

The tests use literal cron expressions, literal dates, and literal expected
JSON/notification values rather than computing expectations with the code
under test.

The suite catches these realistic mutations:

- deleting or changing the Bash slow-operation gate;
- delaying `start_background_task` until handler completion;
- failing to record the task as running;
- omitting or duplicating completion delivery;
- using AND instead of OR for constrained day-of-month/day-of-week;
- accepting an out-of-range minute or a non-five-field expression;
- writing session-only jobs to durable storage;
- using an import-time/cwd-local durable path instead of the configured path;
- failing to reload a durable job;
- failing to remove or persist removal of an exhausted one-shot;
- leaving a cron schema without a handler or adding an unfixed extra handler.

The Bash handler double is limited to the slow operation itself. The test
exercises the real background state, thread, dispatch helper, hook chain,
collection logic, and notification formatting.

## Self-Review

- Reviewed every changed path against the Task 4 brief and the S20 source
  blocks at lines 1269-1528.
- Confirmed public S20 function signatures were retained.
- Confirmed `config.DURABLE_PATH` is dereferenced dynamically by persistence
  functions after workspace selection.
- Confirmed the legacy adapters do not own entries in `TOOL_HANDLERS`; the
  source cron handlers have single fixed ownership in `cron.py`/`tools.py`.
- Confirmed approved legacy tests still pass (20/20 related tests).
- Confirmed no Task 5+ team/protocol code and no Task 18/19 worktree/MCP code
  entered the change.
- Confirmed user-owned dirty paths were not edited, staged, or included.

## Concerns

- The full suite remains non-green only because of the known unrelated dotenv
  baseline (`65 passed, 1 failed`). The task explicitly requires preserving
  that baseline.
- The module retains the S20 import-time durable load and daemon scheduler
  start. Persistence functions themselves use the live configured path;
  callers that switch workspace after import must invoke `load_durable_jobs`
  for that newly selected workspace, as the focused persistence test does.
- Compatibility adapters remain temporarily necessary for the approved
  pre-migration runtime. They are isolated from fixed S20 handler ownership and
  can be retired only when later tasks replace that runtime surface.
