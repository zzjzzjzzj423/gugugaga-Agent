# Simple CC with SiliconFlow — Design

## Goal

Build an independent, compact Python coding-agent harness in
`E:\AgentLearnProject\simple_cc`. It consolidates every capability taught in
`learn-claude-code/s01_agent_loop` through
`learn-claude-code/s17_autonomous_agents`, while excluding S18 worktree
isolation and S19 MCP integration. The model provider is SiliconFlow through
its OpenAI-compatible Chat Completions and Function Calling API.

The implementation may reuse and adapt the teaching code, but it must not
import modules from `learn-claude-code` at runtime.

## Success Criteria

- `python -m simple_cc --workspace <path>` starts an interactive coding agent.
- Configuration comes from environment variables or a local `.env` file.
- The agent can call built-in coding tools and continue the model/tool loop.
- All S01-S17 capabilities listed below are present in the integrated runtime.
- File operations cannot escape the selected workspace.
- Sensitive commands require approval; child agents route approval through the
  lead instead of prompting independently.
- Persistent state is stored below `<workspace>/.simple_cc/`.
- Unit and integration tests run without a real SiliconFlow API key.

## Source Strategy

Use `s20_comprehensive/code.py` as the behavioral integration reference because
it restores early mechanisms intentionally omitted by later focused chapters.
Remove all S18 worktree and S19 MCP behavior, then split the remaining S01-S17
logic into focused modules. Use S15-S17 files as the reference for team
protocols and autonomous teammate behavior.

## Package Layout

```text
simple_cc/
├── simple_cc/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent.py
│   ├── background.py
│   ├── config.py
│   ├── context.py
│   ├── hooks.py
│   ├── models.py
│   ├── permissions.py
│   ├── planning.py
│   ├── prompts.py
│   ├── provider.py
│   ├── teams.py
│   └── tools.py
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

`agent.py` owns the only model/tool loop. Other modules expose services with
small interfaces and do not call the CLI directly.

## Capability Mapping

| Chapter | Capability | Owner |
|---|---|---|
| S01 | Agent loop | `agent.py` |
| S02 | Bash, read, write, edit, glob | `tools.py` |
| S03 | Permission policy and workspace boundary | `permissions.py` |
| S04 | UserPromptSubmit, PreToolUse, PostToolUse, Stop hooks | `hooks.py` |
| S05 | Session todo list | `planning.py` |
| S06 | One-shot subagent | `teams.py` |
| S07 | Discover and load skills on demand | `context.py` |
| S08 | Output trimming and history compaction | `context.py` |
| S09 | Persistent memory | `context.py` |
| S10 | Dynamic system-prompt assembly | `prompts.py` |
| S11 | Retry and context-overflow recovery | `provider.py`, `context.py` |
| S12 | Persistent task dependency graph | `planning.py` |
| S13 | Background command execution and notifications | `background.py` |
| S14 | Cron scheduler | `background.py` |
| S15 | Team members and file mailboxes | `teams.py` |
| S16 | Plan approval and graceful shutdown protocols | `teams.py` |
| S17 | Idle polling and autonomous task claiming | `teams.py` |

## Model Provider

`provider.py` wraps the OpenAI Python SDK:

```python
OpenAI(
    api_key=settings.siliconflow_api_key,
    base_url=settings.siliconflow_base_url,
)
```

Defaults:

- `SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1`
- `SILICONFLOW_MODEL` is required and must support Function Calling.
- `SILICONFLOW_API_KEY` is required for real runs but not for tests.

Tool schemas are converted from the teaching format into OpenAI function tools:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file in the workspace",
    "parameters": {"type": "object", "properties": {}}
  }
}
```

Assistant tool calls are appended as an assistant message containing
`tool_calls`; every result is appended as a separate `role: tool` message with
the matching `tool_call_id`. Provider-specific objects are normalized into
plain dataclasses before reaching the runtime.

The initial implementation uses non-streaming model calls. This keeps partial
tool-call assembly out of the small teaching harness and does not prevent a
future streaming UI adapter.

## Runtime Data Flow

For each user turn, the lead runtime:

1. Runs UserPromptSubmit hooks.
2. Drains cron, background-task, and lead-mailbox notifications.
3. Applies output budgeting and context compaction.
4. Assembles the system prompt from workspace, tool, skill, memory, task, and
   team state.
5. Calls SiliconFlow with messages and the current tool registry.
6. If no tool calls exist, runs Stop hooks and returns the assistant text.
7. For each tool call, runs PreToolUse hooks and permission checks.
8. Executes the tool in the foreground or starts an approved background job.
9. Runs PostToolUse hooks, appends tool results, and repeats from step 2.

The runtime uses actual returned tool calls, not only the API finish reason, to
decide whether another loop iteration is required.

## Tools and Planning

The lead tool pool contains file/shell tools, todo management, one-shot
subagents, skill loading, manual compaction, memory writes, persistent task
operations, background jobs, cron operations, team operations, plan review,
and graceful shutdown.

Todo items are session-local. Persistent tasks are JSON records with status,
owner, and `blocked_by` identifiers. A task can only be claimed when it is
pending, unowned, and all dependencies are complete.

## Context, Skills, and Memory

- Skills are markdown files discovered from `<workspace>/.agents/skills` and
  `<workspace>/.simple_cc/skills`; only metadata is listed in the prompt, and
  full content is loaded through a tool.
- Memory is stored as small markdown records plus an index. The prompt receives
  only the bounded index/relevant excerpts.
- Large tool results are persisted under `.simple_cc/outputs` and replaced in
  history by a path and short preview.
- Compaction proceeds from cheap to expensive: trim large results, crop old
  history, micro-compact old tool results, then summarize history with the
  provider. The pre-summary transcript is retained.

## Permissions and Hooks

- All filesystem paths are resolved and checked against the selected workspace.
- Read-only commands and workspace reads run without prompting.
- Destructive shell patterns, privilege escalation, broad deletion, and risky
  Git operations require lead approval.
- Background jobs pass through the same permission path before a thread starts.
- A denied operation becomes a normal tool result so the model can recover.
- Teammates send permission requests to the lead mailbox; they never read from
  stdin.
- Built-in hooks implement permission gating, audit logging, large-output
  warnings, user-prompt logging, and stop statistics.

## Background Work and Cron

Background shell commands run in daemon threads with a locked notification
queue. Their initial tool result contains a job identifier; completion arrives
later as an independent notification, never as a second result for the same
tool call.

The cron scheduler evaluates validated five-field expressions. Durable jobs are
stored in `.simple_cc/cron.json`; fired prompts enter a queue and wake the lead
runtime. One-shot jobs are removed after firing.

## Teams, Protocols, and Autonomy

- A teammate has a name, role, independent message history, and background
  thread.
- File-backed JSONL mailboxes are protected with in-process locks.
- Teammates share the workspace, task store, provider factory, and bounded tool
  registry.
- Team members cannot spawn nested teammates.
- Plan approval and shutdown messages carry request IDs and explicit state.
- After finishing a turn, a teammate enters IDLE, checks its mailbox first, then
  scans for claimable tasks. It returns to WORK after a message or successful
  claim and exits after the configured idle timeout or approved shutdown.

## Persistence and Concurrency

State lives under `<workspace>/.simple_cc`. JSON state writes use a temporary
file followed by atomic replacement. In-process locks protect task claims,
mailbox append/drain, cron state, and background notifications. This is a
single-process educational implementation; cross-process locking is explicitly
out of scope.

## Error Handling

- Retry rate limits and transient server failures with bounded exponential
  backoff.
- On context-length failure, perform one reactive compaction and retry.
- Invalid or unknown tool calls return structured error text to the model.
- Tool exceptions are contained and returned as tool results.
- Provider authentication and configuration errors fail fast with concise
  startup messages.
- Agent and teammate loops have configurable maximum rounds.

## CLI

The CLI accepts `--workspace` and optional `--model`. It supports regular
prompts plus `/help`, `/status`, `/tasks`, `/team`, `/memory`, and `/exit`.
Interactive permission prompts are owned exclusively by the lead CLI.

## Testing

Tests use a scripted fake provider and never require network access.

- Unit tests: path containment, permission classification, tool schemas,
  message conversion, task dependencies, atomic claims, mailbox protocols,
  cron parsing, compaction, and retry classification.
- Integration tests: model → tool call → tool result → final answer; denied
  command recovery; background completion injection; subagent return; teammate
  plan approval; autonomous task claim; graceful shutdown.
- CLI smoke test: run a scripted session against the fake provider.
- Optional live smoke test is documented but skipped unless a SiliconFlow key
  and model are explicitly supplied.

## Exclusions

- S18 Git worktree creation/removal and task-to-worktree binding.
- S19 MCP discovery, transport, and dynamic external tools.
- Web UI or HTTP server.
- Cross-process distributed coordination.
- Streaming partial model/tool-call deltas.

