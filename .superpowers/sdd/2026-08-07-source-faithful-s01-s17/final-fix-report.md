# Final security and lifecycle fix report

Date: 2026-08-08 (Asia/Shanghai)
Starting HEAD: `af5dbd64cd7b98d3c74cac24361cb328901aded7`
Scope: final whole-branch fix wave for the source-faithful S01-S17 split.

## Outcome

All requested findings are fixed and covered by regression tests. The final
full suite passes 138/138 tests. The implementation remains within S01-S17:
no S18 worktree behavior, no S19 MCP behavior, and no OpenAI API use outside
the existing `provider.py` compatibility boundary were added.

## Fixes

### Mailbox and task path containment

- `MessageBus.send` validates both sender and recipient before creating the
  mailbox directory or constructing a path; `read_inbox` validates before
  existence checks or unlinking.
- Agent names use a strict ASCII identifier grammar:
  `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`. Absolute paths, drive-relative paths,
  UNC paths, separators, traversal, ADS/colon syntax, whitespace, and other
  identifier characters are rejected.
- The source task store accepts only its generated ID shape
  `task_<digits>_<four digits>` at `_task_path`, and all get/claim/complete,
  dependency, save, and public handler paths pass through that validation.
- The compatibility `TaskStore` is also protected using its generated shape
  `task_<digits>_<six lowercase hex characters>`.
- Loaded task IDs must match the requested file ID, and list operations ignore
  files whose names do not match the applicable generated-ID grammar.
- Tests use external sentinel files and cover POSIX absolute/traversal forms,
  Windows absolute/drive-relative/UNC forms, both separators, ADS/colon forms,
  and malformed identifiers. No rejected operation can create, read, write,
  or unlink outside the selected `.mailboxes` or `.tasks` directory.

### Teammate permission and hook boundary

- Runtime bootstrap creates one `PermissionPolicy` and supplies the same
  instance and lead/user approval callback to the lead and teammate paths.
- Source teammate tools now execute in lead order:
  `PreToolUse` -> permission decision/approval -> handler -> `PostToolUse`.
- A hook denial returns its S20-style `tool_result`; a policy/approval denial
  returns `Permission denied for tool '<name>'. Choose a safer approach.`
- Denied tools never invoke the handler or `PostToolUse`, matching the retained
  source lead loop. The built-in approval callback refuses background-thread
  stdin reads, so a teammate cannot read CLI stdin to approve itself.
- Integration coverage returns Bash, write, and read calls in one provider
  response, proves denied Bash/write calls have no file effects, and proves
  the allowed read follows the expected hook order.

### Shutdown ownership

- Source background jobs now track their thread, cancellation event, command,
  and lifecycle status. Initialization refuses to replace a live registry.
- Shutdown stops acceptance, sets every cancellation event, bounded-joins all
  tracked workers, and returns `BackgroundShutdownOutcome`. It reports exact
  live job IDs and never reports `stopped=True` while a tracked worker lives.
- The real Bash handler uses `Popen`, polls a cooperative cancellation event,
  terminates then kills on cancellation when needed, and retains the existing
  120-second timeout and output bound.
- Teammate shutdown stops acceptance and returns an explicit outcome with any
  live teammate names. The teammate loop re-checks shutdown immediately after
  `provider.create` and again before every tool dispatch.
- `SimpleCCApp.close(timeout=...)` owns a single bounded deadline, stops
  background acceptance first, signals teammates/cron/autorun, joins owned
  threads, and returns `ApplicationCloseOutcome(stopped, live_threads)`.
- Deterministic tests cover direct close, `q`, and `KeyboardInterrupt` while a
  teammate provider call is in flight. Releasing the late write response only
  after shutdown causes no write. A mutation check removing both shutdown
  guards produced the late file in both CLI cases, proving the tests detect the
  regression.

Subprocess limitation: terminating a shell cannot guarantee that every
platform-specific descendant process has already exited. The implementation
cooperatively terminates/kills the owned shell process and joins the tracked
Python worker. If an uncooperative handler or descendant keeps that worker
live beyond the bound, the close outcome lists the live background job and
does not claim shutdown completed.

### SOURCE_MAP row audit

- The audit now parses every retained mapping row independently.
- Every target module named by a row must exist.
- Every retained source symbol must exist in one of that same row's claimed
  modules, rather than merely somewhere in the package-wide union.
- Qualified target claims such as `Settings.workspace` and
  `SiliconFlowProvider.client/create` are checked against class fields,
  instance attributes, and methods.
- Explicit grouped constants may resolve across the modules listed by their
  row. Deletion/exclusion prose remains classified by the dedicated exclusion
  audit and is not misread as a retained target-module claim.
- A negative regression maps `create_task` to `teams.py` and proves the row
  validator rejects a symbol that exists only in another module.

## TDD evidence

- Source task/mailbox safety RED: 23 failed, 2 passed, 15 deselected; failures
  showed absent validation or attempted traversal/subdirectory I/O.
- Source task/mailbox safety GREEN: 25 passed, 15 deselected.
- Teammate permission boundary RED: 1 failed because policy/callback plumbing
  was absent; GREEN: 1 passed.
- Background lifecycle RED: 2 failed because lifecycle APIs were absent;
  GREEN: 2 passed, 6 deselected.
- Late-provider direct-close RED: 1 failed because close had no bounded
  outcome/stop signal; GREEN: 1 passed.
- CLI `q`/`KeyboardInterrupt` mutation RED: 2 failed with real late file
  creation when the new stop guards were removed; restored GREEN: 2 passed.
- Compatibility TaskStore RED: 8 failed; GREEN: 8 passed, 3 deselected.
- SOURCE_MAP focused audit: 2 passed, 4 deselected, including the negative-row
  validator regression.
- Related security/lifecycle/audit suite: 95 passed.
- Final full suite: 138 passed in 5.09 seconds.

The first sandboxed pytest attempt could not create/read pytest's Windows temp
directory because of the managed ACL. All recorded RED/GREEN and final runs
used the approved `E:\AgentLearnProject\simple_cc_test_tmp` base directory;
the feature RED failures listed above are from those valid runs.

## Final verification

- `python -m pytest --basetemp E:\AgentLearnProject\simple_cc_test_tmp -q`:
  138 passed.
- `python -m compileall -q simple_cc tests`: passed.
- `python -m simple_cc --help`: passed and printed the expected CLI options.
- Forbidden implementation scan: no S18/S19 terms in `simple_cc/*.py`.
- Direct OpenAI scan: no OpenAI import/client construction outside
  `simple_cc/provider.py`; the existing provider boundary contains the only
  two matches.
- `git diff --check`: passed.

## Preservation and concerns

- The pre-existing user changes in `.env.example`, `.idea/`, and
  `docs/superpowers/plans/2026-08-07-source-faithful-s01-s17.md` were not
  edited, staged, or committed by this fix wave.
- No external sentinel files were left by the RED security tests.
- The only residual semantic limitation is the subprocess-descendant caveat
  documented above; outcomes remain truthful whenever the tracked worker is
  still live.
