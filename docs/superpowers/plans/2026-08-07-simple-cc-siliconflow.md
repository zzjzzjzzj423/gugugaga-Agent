# Simple CC with SiliconFlow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent, modular Python coding-agent harness that combines the complete S01-S17 feature set and calls SiliconFlow through its OpenAI-compatible Function Calling API.

**Architecture:** A single `AgentRuntime` owns the model/tool loop. Focused services own tools, permissions/hooks, planning, context, asynchronous work, and teams; all persistent state stays below the selected workspace's `.simple_cc` directory. `SiliconFlowProvider` is the only module coupled to the OpenAI SDK.

**Tech Stack:** Python 3.11+, `openai`, `python-dotenv`, standard-library threads/queues/dataclasses/pathlib, `pytest`.

## Global Constraints

- Create the implementation only under `E:\AgentLearnProject\simple_cc`.
- Do not import from `learn-claude-code` at runtime.
- Include every S01-S17 capability; exclude S18 worktrees, S19 MCP, HTTP/Web UI, and partial-token streaming.
- Restrict file mutations to the selected workspace.
- Tests must run without network access or an API key.
- Use `SILICONFLOW_API_KEY`, `SILICONFLOW_MODEL`, and default base URL `https://api.siliconflow.cn/v1`.
- Keep OpenAI-compatible message conversion inside `provider.py`.

---

## File Map

- `simple_cc/__main__.py`: argument parsing and interactive CLI.
- `simple_cc/config.py`: validated settings and state-directory creation.
- `simple_cc/models.py`: provider-neutral tool and response dataclasses.
- `simple_cc/provider.py`: SiliconFlow/OpenAI adapter, retry classification, fake-provider protocol.
- `simple_cc/tools.py`: workspace-safe built-in coding tools and registry.
- `simple_cc/permissions.py`: allow/ask/deny policy and lead approval interface.
- `simple_cc/hooks.py`: hook registration, dispatch, and built-in audit hooks.
- `simple_cc/planning.py`: session todos and persistent task dependency graph.
- `simple_cc/context.py`: skills, memory, output budgets, transcripts, compaction.
- `simple_cc/prompts.py`: dynamic system-prompt assembly.
- `simple_cc/background.py`: background jobs, completion queue, cron scheduler.
- `simple_cc/teams.py`: subagents, mailbox, protocols, teammate lifecycle and autonomy.
- `simple_cc/agent.py`: integrated S01-S17 runtime loop.
- `tests/`: unit and integration coverage with scripted providers.

### Task 1: Package, Configuration, and Provider Boundary

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `simple_cc/__init__.py`
- Create: `simple_cc/config.py`
- Create: `simple_cc/models.py`
- Create: `simple_cc/provider.py`
- Test: `tests/test_config_provider.py`

**Interfaces:**
- Produces: `Settings.from_env(workspace: Path, model_override: str | None) -> Settings`
- Produces: `ToolSpec`, `ToolCall`, `ModelResponse`
- Produces: `ChatProvider.complete(system: str, messages: list[dict], tools: list[ToolSpec], max_tokens: int) -> ModelResponse`
- Produces: `SiliconFlowProvider(settings: Settings)`

- [ ] **Step 1: Write failing configuration and provider-normalization tests**

```python
def test_settings_builds_state_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    settings = Settings.from_env(tmp_path)
    assert settings.base_url == "https://api.siliconflow.cn/v1"
    assert settings.state_dir == tmp_path / ".simple_cc"

def test_normalize_tool_call():
    raw = SimpleNamespace(id="call_1", function=SimpleNamespace(
        name="read_file", arguments='{"path":"README.md"}'))
    call = normalize_tool_call(raw)
    assert call == ToolCall("call_1", "read_file", {"path": "README.md"})
```

- [ ] **Step 2: Run `python -m pytest tests/test_config_provider.py -v` and verify import failures**

- [ ] **Step 3: Implement settings, dataclasses, OpenAI tool conversion, response normalization, and bounded retry**

```python
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

class ChatProvider(Protocol):
    def complete(self, system, messages, tools, max_tokens=8192) -> ModelResponse: ...
```

`SiliconFlowProvider.complete` must pass `model`, a leading system message,
OpenAI function tools, `max_tokens`, and `stream=False`; retry HTTP 429 and 5xx
responses with bounded exponential backoff and expose context-length failures as
`ContextLengthError`.

- [ ] **Step 4: Run the provider tests and verify they pass**
- [ ] **Step 5: Commit with `git commit -am "feat: add siliconflow provider boundary"` after adding new files**

### Task 2: Tool Registry, Workspace Safety, Permissions, and Hooks

**Files:**
- Create: `simple_cc/tools.py`
- Create: `simple_cc/permissions.py`
- Create: `simple_cc/hooks.py`
- Test: `tests/test_tools_permissions_hooks.py`

**Interfaces:**
- Consumes: `ToolSpec`, `ToolCall`, `Settings`
- Produces: `ToolRegistry.register(spec, handler)`, `specs()`, `execute(name, arguments)`
- Produces: `WorkspaceTools(workspace: Path).register_into(registry)`
- Produces: `PermissionPolicy.decide(call: ToolCall) -> PermissionDecision`
- Produces: `HookManager.trigger(event: HookEvent, **payload) -> list[Any]`

- [ ] **Step 1: Write failing tests for path escape, edit semantics, permission classification, and hook order**

```python
def test_read_rejects_parent_escape(tmp_path):
    tools = WorkspaceTools(tmp_path)
    assert tools.read_file("../secret.txt").startswith("Error: path escapes workspace")

@pytest.mark.parametrize("command", ["rm -rf build", "git reset --hard", "sudo apt update"])
def test_dangerous_commands_require_approval(command):
    decision = PermissionPolicy().decide(ToolCall("1", "bash", {"command": command}))
    assert decision is PermissionDecision.ASK
```

- [ ] **Step 2: Run the focused tests and confirm failure**
- [ ] **Step 3: Implement registry and `bash/read_file/write_file/edit_file/glob` handlers**
- [ ] **Step 4: Implement allow/ask/deny policy, lead approval callback, and built-in audit/large-output/stop hooks**
- [ ] **Step 5: Run `python -m pytest tests/test_tools_permissions_hooks.py -v`**
- [ ] **Step 6: Commit with `git commit -am "feat: add safe tools permissions and hooks"` after adding new files**

### Task 3: Todo and Persistent Task Graph

**Files:**
- Create: `simple_cc/planning.py`
- Test: `tests/test_planning.py`

**Interfaces:**
- Produces: `TodoStore.update(items: list[dict]) -> str`
- Produces: `TaskStore.create/list/get/claim/complete`
- Produces: `Task(id, subject, description, status, owner, blocked_by)`

- [ ] **Step 1: Write failing normalization and dependency tests**

```python
def test_claim_requires_completed_dependencies(task_store):
    schema = task_store.create("schema")
    api = task_store.create("api", blocked_by=[schema.id])
    assert "Blocked" in task_store.claim(api.id, "alice")
    task_store.claim(schema.id, "alice")
    task_store.complete(schema.id)
    assert "Claimed" in task_store.claim(api.id, "bob")
```

- [ ] **Step 2: Run `python -m pytest tests/test_planning.py -v` and confirm failure**
- [ ] **Step 3: Implement todo validation and atomic JSON task persistence using temp-file replacement**
- [ ] **Step 4: Guard create/claim/complete with a lock and report newly unblocked tasks**
- [ ] **Step 5: Run planning tests and commit `feat: add todo and persistent task graph`**

### Task 4: Skills, Memory, Prompt Assembly, and Compaction

**Files:**
- Create: `simple_cc/context.py`
- Create: `simple_cc/prompts.py`
- Test: `tests/test_context_prompts.py`

**Interfaces:**
- Consumes: `Settings`, `ToolSpec`, `TaskStore`
- Produces: `SkillStore.discover/list/load`
- Produces: `MemoryStore.remember/search/index_text`
- Produces: `ContextManager.prepare(messages, provider) -> list[dict]`
- Produces: `PromptAssembler.build(runtime_state: dict) -> str`

- [ ] **Step 1: Write failing tests for skill metadata-only discovery, bounded memory injection, large-output persistence, and tool-pair-preserving compaction**

```python
def test_large_tool_output_is_persisted(context_manager):
    messages = [{"role": "tool", "tool_call_id": "c1", "content": "x" * 250_000}]
    compacted = context_manager.apply_output_budget(messages)
    assert len(compacted[0]["content"]) < 2_000
    assert "outputs" in compacted[0]["content"]
```

- [ ] **Step 2: Run context tests and confirm failure**
- [ ] **Step 3: Implement skill frontmatter parsing and on-demand content loading**
- [ ] **Step 4: Implement memory records/index, transcript storage, output budgeting, crop, micro-compaction, and provider summary fallback**
- [ ] **Step 5: Implement named prompt sections for workspace, tools, skills, memory, tasks, teams, safety, and operating rules**
- [ ] **Step 6: Run context tests and commit `feat: add context memory skills and prompts`**

### Task 5: Background Jobs and Cron Scheduler

**Files:**
- Create: `simple_cc/background.py`
- Test: `tests/test_background_cron.py`

**Interfaces:**
- Produces: `BackgroundManager.start(call, fn) -> str`, `drain() -> list[str]`, `has_pending() -> bool`
- Produces: `CronScheduler.schedule/list/cancel/drain/start/stop`

- [ ] **Step 1: Write failing tests for one-result semantics, completion notification, cron validation, and one-shot removal**

```python
def test_background_completion_is_independent_notification(manager):
    job_id = manager.start(ToolCall("c1", "bash", {}), lambda: "done")
    wait_until(lambda: manager.has_pending())
    assert job_id in manager.drain()[0]
    assert manager.drain() == []
```

- [ ] **Step 2: Run background tests and verify failure**
- [ ] **Step 3: Implement locked job registry and notification queue**
- [ ] **Step 4: Implement five-field cron parser, durable atomic store, scheduler thread, and fired-prompt queue**
- [ ] **Step 5: Run tests and commit `feat: add background jobs and cron scheduler`**

### Task 6: Integrated Agent Loop and One-Shot Subagents

**Files:**
- Create: `simple_cc/agent.py`
- Create: `tests/fakes.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: all services from Tasks 1-5
- Produces: `AgentRuntime.run_turn(query: str) -> str`
- Produces: `AgentRuntime.run_messages(messages: list[dict], max_rounds: int) -> str`
- Produces: `SubagentRunner.run(prompt: str, agent_type: str) -> str`

- [ ] **Step 1: Create `ScriptedProvider` and a failing tool-loop integration test**

```python
def test_tool_call_result_returns_to_model(runtime, provider):
    provider.queue(ModelResponse("", [ToolCall("c1", "write_file", {
        "path": "hello.txt", "content": "hi"})], "tool_calls"))
    provider.queue(ModelResponse("Created hello.txt", [], "stop"))
    assert runtime.run_turn("create hello.txt") == "Created hello.txt"
    assert provider.requests[-1]["messages"][-1]["role"] == "tool"
```

- [ ] **Step 2: Run loop tests and verify failure**
- [ ] **Step 3: Implement notification drain, context prepare, prompt build, provider call, tool permission/hooks, foreground/background dispatch, and stop handling**
- [ ] **Step 4: On `ContextLengthError`, compact once and retry; enforce maximum rounds**
- [ ] **Step 5: Add a one-shot subagent with isolated history, bounded tools, and summary-only return**
- [ ] **Step 6: Run loop tests and commit `feat: integrate agent loop and subagents`**

### Task 7: Teams, Protocols, and Autonomous Claiming

**Files:**
- Create: `simple_cc/teams.py`
- Test: `tests/test_teams.py`

**Interfaces:**
- Consumes: `ChatProvider`, `TaskStore`, `ToolRegistry`, `AgentRuntime`
- Produces: `Mailbox.send/drain/peek`
- Produces: `ProtocolStore.request/resolve/get`
- Produces: `TeamManager.spawn/send/check_inbox/request_shutdown/request_plan/review_plan/status/stop_all`

- [ ] **Step 1: Write failing mailbox and protocol-correlation tests**

```python
def test_plan_response_must_match_request_type(protocols):
    request = protocols.request("plan_approval", "alice", "lead", "plan")
    with pytest.raises(ProtocolError):
        protocols.resolve(request.id, "shutdown_response", True)
```

- [ ] **Step 2: Write a failing autonomous-claim integration test using `ScriptedProvider`**
- [ ] **Step 3: Implement locked JSONL mailbox, request IDs, plan approval, and graceful shutdown**
- [ ] **Step 4: Implement teammate WORK/IDLE/SHUTDOWN lifecycle, mailbox-first polling, and atomic task claims**
- [ ] **Step 5: Prevent nested teammate spawning and route teammate approval requests to the lead**
- [ ] **Step 6: Run team tests and commit `feat: add autonomous agent teams`**

### Task 8: CLI Composition and Documentation

**Files:**
- Create: `simple_cc/__main__.py`
- Create: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every service and runtime factory
- Produces: `build_runtime(settings, approval_callback) -> AgentRuntime`
- Produces: `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write failing CLI tests for missing configuration, `--workspace`, `/status`, and `/exit`**
- [ ] **Step 2: Implement CLI parser, runtime composition, lead approval prompt, commands `/help`, `/status`, `/tasks`, `/team`, `/memory`, `/exit`, and shutdown cleanup**
- [ ] **Step 3: Document installation, `.env`, SiliconFlow Function Calling model requirement, usage, state layout, safety, and optional live smoke test**
- [ ] **Step 4: Run `python -m pytest tests/test_cli.py -v` and commit `feat: add simple cc cli and docs`**

### Task 9: Feature-Parity and Final Verification

**Files:**
- Create: `tests/test_feature_parity.py`
- Modify: any implementation file only when a parity test exposes a defect

**Interfaces:**
- Consumes: public runtime and service APIs from Tasks 1-8
- Produces: verified S01-S17 package

- [ ] **Step 1: Add a parameterized test asserting every S01-S17 capability has a registered runtime owner or tool**

```python
EXPECTED = {
    "bash", "read_file", "write_file", "edit_file", "glob",
    "todo_write", "subagent", "load_skill", "compact", "remember",
    "create_task", "list_tasks", "get_task", "claim_task", "complete_task",
    "background_run", "schedule_cron", "list_crons", "cancel_cron",
    "spawn_teammate", "send_message", "check_inbox",
    "request_shutdown", "request_plan", "review_plan",
}
assert EXPECTED <= {spec.name for spec in runtime.registry.specs()}
```

- [ ] **Step 2: Run `python -m pytest -q` and fix only observed failures**
- [ ] **Step 3: Run `python -m compileall simple_cc`**
- [ ] **Step 4: Run `python -m simple_cc --help`**
- [ ] **Step 5: Verify `rg -n "worktree|connect_mcp|mcp__" simple_cc tests` finds no implementation references**
- [ ] **Step 6: Inspect `git diff --check` and `git status --short`**
- [ ] **Step 7: Commit final verification changes with `test: verify s01 through s17 feature parity`**

