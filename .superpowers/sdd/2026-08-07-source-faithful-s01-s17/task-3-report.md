# Task 3 report: source-faithful context and recovery

## Implementation summary

Moved the S20 context-compaction and error-recovery blocks into
`simple_cc.context` and `simple_cc.recovery` while retaining their function
names, mutation order, retry state, and Anthropic-shaped `tool_use` /
`tool_result` handling. Workspace-derived transcript and persisted-output paths
are read through `simple_cc.config`, so `configure_workspace(...)` remains the
single path authority.

`summarize_history(...)` retains the S20 positional interface and prompt, but
calls the Task 1 provider adapter's `create(...)` contract. No OpenAI request or
response conversion was added outside `provider.py`.

The S20 `compact` tool schema and prompt advertisement were activated for this
stage. The fixed tool table includes a `run_compact(...)` dispatch placeholder
to preserve definition/handler bijection; the source-faithful agent-loop branch
in Task 6 will continue to own the actual history mutation, as in S20.

The earlier compatibility `ContextManager`, `MemoryStore`, and `SkillStore`
remain in `context.py` so the staged Tasks 1-2 runtime and its existing tests
continue to work until Task 6 replaces the old loop.

## Files changed

- `simple_cc/context.py`
- `simple_cc/recovery.py`
- `simple_cc/prompts.py`
- `simple_cc/tools.py`
- `tests/test_context_recovery.py`
- `tests/test_workspace_tasks_skills.py`
- `.superpowers/sdd/2026-08-07-source-faithful-s01-s17/task-3-report.md`

The Task 2 registry/prompt test was advanced from its previous "compact is not
yet present" expectation to Task 3's active compact schema and restored strict
definition/handler bijection after review.

## RED evidence

The required focused command was run after adding the Task 3 tests and before
production migration:

```text
python -m pytest tests/test_context_recovery.py -v
```

Result:

```text
collected 0 items / 1 error
ImportError: cannot import name 'recovery' from 'simple_cc'
Interrupted: 1 error during collection
```

This was the expected missing-module boundary: `simple_cc.recovery` and the new
S20 interfaces did not exist.

### Review-fix RED

Independent review found that `compact` had been advertised without a fixed
handler and that the prior registry invariant had been weakened. Tests were
strengthened before the fix:

```text
python -m pytest tests/test_context_recovery.py::test_compact_tool_has_a_fixed_dispatch_placeholder_for_agent_loop_handling tests/test_workspace_tasks_skills.py::test_fixed_foundation_registry_has_one_handler_per_definition -v
```

Result:

```text
2 failed in 0.56s
KeyError: 'compact'
Extra items in the definition-name set: 'compact'
```

After adding the minimal `run_compact(...)` placeholder and restoring exact
bijection, the same command returned:

```text
2 passed in 0.46s
```

## GREEN evidence

The first ordinary post-migration run reached all seven initial tests. Four
non-filesystem tests passed, while three `tmp_path` tests failed during fixture
setup because the sandbox could not scan pytest's Windows temporary directory:

```text
python -m pytest tests/test_context_recovery.py -v
```

```text
4 passed, 3 errors in 0.96s
PermissionError: [WinError 5]
C:\Users\Administrator\AppData\Local\Temp\pytest-of-Administrator
```

Per the brief's ACL fallback, the exact focused command was rerun elevated. The
initial migration returned `7 passed in 0.50s`. After the review fix, the final
fresh focused run returned:

```text
python -m pytest tests/test_context_recovery.py -v
```

```text
collected 8 items
tests/test_context_recovery.py ........                         [100%]
8 passed in 0.50s
```

The tests cover persisted large output and its exact S20 reference, tool-pair
safe snipping, configured transcript placement, provider-adapter summaries,
transient 429 recovery, one guarded reactive compaction, and the compact
schema/dispatch contract.

## Full-suite output

The final full suite was run elevated for the same pytest temporary-directory
ACL:

```text
python -m pytest -q
```

Result:

```text
...............F..............................................           [100%]
FAILED tests/test_config_provider.py::test_settings_requires_model
1 failed, 61 passed in 0.86s
```

This is the documented pre-existing baseline failure: unchanged
`load_dotenv(override=False)` reloads the repository `.env` value after the
test deletes only the process variable. Dotenv behavior was not changed.

Additional final verification:

```text
python -m compileall -q simple_cc
```

Exit 0.

```text
git diff --check
```

Exit 0 (line-ending warnings only). A scoped `rg` audit found no `worktree`,
`MCPClient`, `connect_mcp`, `mcp__`, or `connected_mcp` token in Task 3 runtime
or test files.

## Review and self-review

Independent review found no Critical issues. Its Important registry-bijection
issue was corrected with the RED/GREEN cycle above. The final implementation
was then checked against the Task 3 brief and S20 lines 1055-1256:

- All requested public interfaces are present: `estimate_size`,
  `tool_result_budget`, `snip_compact`, `micro_compact`, `write_transcript`,
  `summarize_history`, `compact_history`, `reactive_compact`, `RecoveryState`,
  `with_retry`, and `is_prompt_too_long_error`.
- Context history remains Anthropic-shaped; tool pairing checks inspect
  content blocks rather than OpenAI `tool_calls` messages.
- Output budgeting mutates the last user-side tool-result group in S20 order,
  persists only above `PERSIST_THRESHOLD`, and keeps the exact S20 reference
  wrapper and preview length.
- Snip and reactive boundaries pull an assistant `tool_use` back in whenever
  the retained tail begins with its user-side `tool_result`.
- Summary requests cross only the provider adapter's S20-facing `create(...)`
  boundary, with an empty system string, no tools, and 2000 max tokens.
- Retry classification, exponential jitter, 429/529 paths, fallback-model
  switch state, and max-retry terminal error retain the S20 flow.
- `compact` has one schema and one callable fixed-table entry. Its placeholder
  does not preempt Task 6's S20 special branch that mutates the live history.
- No S18 worktree or S19 MCP code was introduced.
- `.env.example`, `.idea/`, and the uncommitted plan document were not edited
  or staged by Task 3. `provider.py` and dotenv configuration were unchanged.

## Concerns

- The full suite retains the known unrelated dotenv/model failure above.
- The prompt-too-long test composes the migrated classifier, state guard, and
  reactive compactor, but the production agent-loop wiring belongs to Task 6.
  Task 6 should add a scripted-provider integration test proving a real
  prompt-too-long failure causes exactly one reactive-compaction retry.
- The compact fixed-table handler is intentionally only a dispatch placeholder;
  Task 6 must retain S20's `block.name == "compact"` special handling before
  generic handler dispatch.
