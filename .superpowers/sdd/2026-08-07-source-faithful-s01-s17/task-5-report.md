# Task 5 Report: S15-S17 Teams, Protocols, and Autonomous Claiming

## Status

Implemented and verified the Task 5 migration from
`learn-claude-code/s20_comprehensive/code.py` into the source-faithful
`simple_cc` module split.

The migration adds S20's JSONL `MessageBus`, correlated protocol state and
lead-inbox routing, autonomous inbox/task polling, teammate lifecycle and
plan/shutdown helpers, fixed lead team tools, and content-block teammate
provider calls. Teammates operate in the selected shared workspace. No
worktree context, worktree-wrapped file functions, worktree tools, MCP, or
team-side OpenAI response conversion was added.

## Scope

Changed Task 5 paths:

- `simple_cc/__main__.py`
  - Installs the runtime's actual configured provider into the source team
    lifecycle before any fixed source team handler can spawn a teammate.
- `simple_cc/teams.py`
  - Added `MessageBus`, `BUS`, `ProtocolState`, `pending_requests`, request-ID
    generation and response matching, `consume_lead_inbox`, unclaimed-task
    scanning, idle polling, provider configuration, teammate spawning, plan
    submission/review, shutdown request/acknowledgement, and lead message and
    inbox handlers.
  - Calls the injected `provider.py` content-block boundary through
    `provider.create(...)`; teammate histories retain `TextBlock`,
    `ToolUseBlock`, and S20 `tool_result` content blocks.
  - Directly binds shared-workspace `run_bash`, `run_read`, and `run_write`.
    No `wt_ctx`, alternate cwd, worktree wrapper, or OpenAI conversion branch
    exists.
  - Retained `Mailbox`, `ProtocolStore`, and `TeamManager` as compatibility
    adapters for the approved pre-migration application/tests.
- `simple_cc/tasks.py`
  - Made the pending/owner/dependency read-check-write transition atomic
    across autonomous teammate threads with a process-wide re-entrant lock.
- `simple_cc/prompts.py`
  - Exposed the six fixed lead team tools in the live S20 system prompt.
- `simple_cc/tools.py`
  - Added fixed schemas and one-to-one handlers for `spawn_teammate`,
    `send_message`, `check_inbox`, `request_shutdown`, `request_plan`, and
    `review_plan`.
- `tests/test_teams_source.py`
  - Added nine behavior tests covering one-time FIFO mailbox delivery,
    response-type and request-ID mismatch rejection, plan correlation/routing,
    shutdown acknowledgement/routing, a real two-thread atomic claim race,
    deterministic concurrent request-ID collision, scripted content-block
    provider execution in the selected workspace, real runtime provider
    bootstrap through the fixed spawn handler, and fixed tool
    definition/handler/prompt exposure.
- `tests/test_workspace_tasks_skills.py`
  - Updated the prior-stage negative `spawn_teammate` prompt assertion to the
    Task 5 positive expectation. Worktree and MCP negative assertions remain.

Explicitly unchanged and excluded from the commit:

- `simple_cc/provider.py`
- `.env.example`, `.idea/`, and the untracked plan document
- worktree and MCP behavior/handlers
- the known dotenv/model baseline behavior

## TDD RED Evidence

The Task 5 behavior tests were written before production changes.

The first sandboxed command was:

```text
python -m pytest tests/test_teams_source.py -v
```

It could not reach the intended behavior because pytest's Windows temp root
was unreadable inside the sandbox:

```text
PermissionError: [WinError 5]
C:\Users\Administrator\AppData\Local\Temp\pytest-of-Administrator
```

Per the task instructions, pytest alone was rerun elevated for that temp ACL.
The expected RED result was:

```text
collected 7 items
7 failed in 0.57s
```

Every failure reported the absent S15-S17 API: `MessageBus`, `ProtocolState`,
protocol routing, autonomous scan/poll/spawn functions, lead helpers, and the
content-block provider setter. This was a feature-missing failure, not a typo
or unrelated environmental failure.

## GREEN Evidence

An intermediate focused run produced six passes and one provider-thread
lifecycle failure. Systematic investigation found that the test had replaced
`teams.time.sleep`; because `time` is a process-wide module, this also removed
sleeping from the already-running cron scheduler and the test's own wait loop,
causing CPU-spin/starvation. The test harness was corrected to use a real
one-second poll without changing production semantics.

Single provider-boundary regression after that correction:

```text
python -m pytest \
  tests/test_teams_source.py::test_teammate_uses_content_block_provider_in_selected_shared_workspace \
  -v
1 passed in 1.53s
```

Fresh focused Task 5 verification:

```text
python -m pytest tests/test_teams_source.py -v
7 passed in 2.60s
```

All pytest runs needing `tmp_path` were elevated only for the Windows sandbox
temp ACL.

## Full-Suite Evidence

The first full run after focused GREEN found the known dotenv failure plus one
stale previous-stage assertion that `spawn_teammate` must not be prompted.
Task 5 intentionally exposes that tool, so the stale assertion was changed to
the positive Task 5 expectation. The new focused test had already driven and
verified the production behavior through RED/GREEN.

Fresh full-suite command:

```text
python -m pytest -q
```

Result:

```text
1 failed, 74 passed in 3.01s
```

The sole failure is the declared pre-existing baseline:

```text
tests/test_config_provider.py::test_settings_requires_model
Failed: DID NOT RAISE ValueError
```

As documented by earlier tasks, `load_dotenv(override=False)` reloads the
repository `.env` model after the test deletes the process variable. Task 5
does not modify or mask that behavior.

Additional checks:

- `python -m compileall -q simple_cc tests`: pass.
- `git diff --check`: pass.
- Fixed registry audit: 23 unique definitions, 23 handlers, equal name sets.
- Forbidden production artifact audit over Task 5 modules: no `wt_ctx`,
  worktree tool/wrapper name, MCP name, `OpenAI`, or `to_openai` occurrence.

## Source-Faithfulness Review

- `MessageBus.send(from_agent, to_agent, content, msg_type="message",
  metadata=None)` writes one JSON object per line with the S20 fields.
  `read_inbox(agent)` returns file order and consumes the mailbox once.
- `ProtocolState` preserves the S20 fields and request ID format. Protocol
  replies are routed by both request ID and the response type required by the
  pending request; an unknown or cross-protocol reply cannot approve it.
- `consume_lead_inbox(route_protocol=True)` drains `lead` and routes correlated
  response messages before returning their original FIFO list.
- `scan_unclaimed_tasks()` reads the live `config.TASKS_DIR`, filters pending
  ownerless tasks, and honors dependency completion through `can_start`.
- `idle_poll(agent_name, messages, name, role)` prioritizes inbox messages,
  acknowledges shutdown with the same request ID, then scans and attempts an
  atomic claim before timing out.
- `spawn_teammate_thread(name, role, prompt)` retains the S20 plan gate,
  identity refresh, bounded inner tool loop, inbox routing, idle transition,
  summary result, and duplicate-name guard.
- Teammate model calls use `provider.create(model, system, messages, tools,
  max_tokens)` and consume S20 content blocks. No direct OpenAI client or
  OpenAI-shaped teammate history is introduced.
- The teammate system text names the live selected `config.WORKDIR`; direct
  workspace file handlers receive no alternate cwd and no worktree state.
- Team lead schemas and handlers are fixed and bijective. Their names are also
  present in the rebuilt system prompt.

## Test Quality / Mutation Review

The tests exercise real filesystem mailboxes, live protocol dictionaries,
actual task files, real Python threads, the real shared-workspace file handler,
and the real teammate loop. Only the external provider is scripted, with a
complete S20 `ProviderResponse` containing `ToolUseBlock` and `TextBlock`.

The suite catches these realistic mutations:

- reversing mailbox lines, dropping a message, or failing to consume once;
- routing a plan response to a shutdown request or accepting an unknown ID;
- losing the plan request ID or replying to an uncorrelated teammate;
- acknowledging shutdown without the original request ID or failing to route
  the lead response into approved state;
- omitting dependency checks or allowing both racing owners to claim;
- calling `.complete(...)`, an OpenAI client, or another boundary instead of
  the content-block provider `.create(...)` path;
- sending OpenAI-shaped rather than S20 `tool_result` history;
- writing outside the configured shared workspace or introducing worktree
  language/state;
- defining a team tool without one handler, adding an unfixed handler, or
  failing to expose the tool in the live prompt.

## Self-Review

- Reviewed all changed modules against the Task 5 brief and S20 source blocks
  for MessageBus, protocol state, autonomous polling, teammate threads, lead
  helpers, tool schemas, and handlers.
- Confirmed compatibility tests remain green; the migration adds the S20 API
  without removing approved legacy adapter surfaces.
- Confirmed the atomic claim lock surrounds the complete task
  read/check/dependency-check/write transition, not just the final write.
- Confirmed teammate tool definitions contain no worktree/MCP tools and file
  handlers use the selected workspace directly.
- Confirmed no provider implementation or OpenAI conversion code changed.
- Confirmed user-owned dirty and untracked paths were not edited, staged, or
  included.

## Concerns

- The full suite remains non-green only because of the explicitly known,
  unrelated dotenv baseline (`74 passed, 1 failed`).
- The source-oriented teammate path requires its content-block provider to be
  installed through `set_team_provider()` before `spawn_teammate_thread` is
  called. `build_runtime` now performs that installation with its actual
  configured provider; alternate embedders must invoke the same explicit
  lifecycle hook. The approved legacy `TeamManager` path remains operational.
- Atomic claiming is process-wide, matching S20's thread-based teammates. It
  does not claim cross-process filesystem locking, which is outside this
  source architecture and task scope.

## Fix Round 1: Independent Review Findings

Independent review found two Important issues in the initial Task 5 change:

1. The fixed source `spawn_teammate` handler was exposed, but production
   runtime assembly never installed its configured provider; only tests called
   `set_team_provider`.
2. `new_request_id()` released `_protocol_lock` before callers inserted their
   `ProtocolState`, so concurrent request creators could select the same
   random ID and one state could overwrite the other.

### Provider Bootstrap RED/GREEN

The new integration regression builds the real application with a scripted
S20 content-block provider, invokes the fixed production
`TOOL_HANDLERS["spawn_teammate"]`, waits for the real teammate loop, and
asserts its shared-workspace write and two provider calls.

Expected RED:

```text
assert TOOL_HANDLERS["spawn_teammate"](...) ==
    "Teammate 'runtime-alice' spawned as developer"
actual: "Error: teammate provider is not configured"
```

Fix: `build_runtime` now calls `set_team_provider(provider)` immediately after
selecting the injected or default `SiliconFlowProvider`, before constructing
and exposing runtime handlers. This is the minimal lifecycle hook needed by
Task 5 and does not rewrite the later source agent-loop assembly task.

### Atomic Request Reservation RED/GREEN

The deterministic concurrency regression forces the first two random choices
to collide at `req_000017`, synchronizes both threads after the unreserved ID
selection, and invokes two real teammate plan submissions. It asserts literal
unique results and both sender states in `pending_requests`.

Expected RED:

```text
expected: Plan submitted (req_000017), Plan submitted (req_000018)
actual:   Plan submitted (req_000017), Plan submitted (req_000017)
```

Fix: `_create_protocol_request` now selects the ID, constructs the full
`ProtocolState`, and inserts/reserves it in `pending_requests` in one
`_protocol_lock` critical section. Public `new_request_id()` keeps its S20
shape, while all production request creators use the atomic helper.

Combined fix-round RED command:

```text
python -m pytest \
  tests/test_teams_source.py::test_concurrent_colliding_request_ids_preserve_both_protocol_states \
  tests/test_teams_source.py::test_runtime_bootstrap_installs_provider_before_source_spawn_handler \
  -v
2 failed in 0.59s
```

Combined GREEN:

```text
2 passed in 1.48s
```

### Fix-Round Verification

Focused Task 5 plus legacy teams, CLI composition, workspace/task/skill, and
permission/hook compatibility:

```text
python -m pytest tests/test_teams_source.py tests/test_teams.py \
  tests/test_cli.py tests/test_workspace_tasks_skills.py \
  tests/test_tools_permissions_hooks.py -v
31 passed in 3.74s
```

Fresh full suite:

```text
python -m pytest -q
1 failed, 76 passed in 3.95s
```

The sole failure remains the unchanged known dotenv baseline
`tests/test_config_provider.py::test_settings_requires_model`.

Independent re-review verdict after both fixes: **APPROVE**. The reviewer
confirmed that ID selection/retry/reservation is one critical section used by
both plan and shutdown creation, runtime bootstrap installs its real provider
before source team handler use, the integration tests exercise the intended
boundaries, and no Critical or Important issue remains.
