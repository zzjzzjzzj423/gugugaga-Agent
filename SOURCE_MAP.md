# S20 source map (S01-S17)

Source baseline: `E:\AgentLearnProject\learn-claude-code\s20_comprehensive\code.py`.
Baseline SHA-256: `9EACF2F2C6F6DBE3B31117008A1A0BE44F52EE29585E5AFA0F4126D8D964D213`.

This map is the audit index for the source-faithful split. Names below are
retained unless they appear in an exclusion table. A target can use a small
adapter name where SiliconFlow's OpenAI-compatible API requires one; the agent
side remains S20's Anthropic-style content-block protocol.

| S20 source item(s) | Target module / symbol | Status |
| --- | --- | --- |
| `WORKDIR` | `config.py` / `Settings.workspace` | retained configuration |
| `client` | `provider.py` / `SiliconFlowProvider.client` | retained behind the sole provider boundary |
| `MODEL`, `PRIMARY_MODEL`, `FALLBACK_MODEL` | `config.py` / model settings | retained configuration |
| `SKILLS_DIR`, `TRANSCRIPT_DIR`, `TOOL_RESULTS_DIR`, `TASKS_DIR`, `MAILBOX_DIR`, `MEMORY_DIR`, `MEMORY_INDEX`, `DURABLE_PATH` | `config.py` / configured workspace paths | retained configuration |
| `DEFAULT_MAX_TOKENS`, `ESCALATED_MAX_TOKENS`, `MAX_RETRIES`, `MAX_CONSECUTIVE_529`, `MAX_RECOVERY_RETRIES`, `BASE_DELAY_MS`, `CONTEXT_LIMIT`, `KEEP_RECENT_TOOL_RESULTS`, `PERSIST_THRESHOLD`, `CONTINUATION_PROMPT` | `config.py`, `context.py`, `recovery.py` | retained recovery/context constants |
| `PROMPT`, `CLI_ACTIVE`, `terminal_print`, `print_turn_assistants`, `cron_autorun_loop` | `__main__.py` | retained CLI behavior |
| `CURRENT_TODOS`, `Task`, `_task_path`, `create_task`, `save_task`, `load_task`, `list_tasks`, `get_task_json`, `can_start`, `claim_task`, `complete_task`, `_normalize_todos`, `run_todo_write`, `run_create_task`, `run_list_tasks`, `run_get_task`, `run_claim_task`, `run_complete_task` | `tasks.py` | retained S01/S02 task and todo flow |
| `SKILL_REGISTRY`, `_parse_frontmatter`, `scan_skills`, `list_skills`, `load_skill` | `skills.py` | retained skill discovery |
| `PROMPT_SECTIONS`, `assemble_system_prompt`, `SUB_SYSTEM` | `prompts.py` | retained prompt construction |
| `safe_path`, `run_bash`, `run_read`, `run_write`, `run_edit`, `run_glob`, `call_tool_handler` | `workspace.py` and `tools.py` | retained file/tool dispatch |
| `HOOKS`, `register_hook`, `trigger_hooks`, `DENY_LIST`, `DESTRUCTIVE`, `permission_hook`, `log_hook`, `large_output_hook`, `user_prompt_hook`, `stop_hook` | `hooks.py` | retained hooks and permission policy |
| `SUB_TOOLS`, `SUB_HANDLERS`, `extract_text`, `has_tool_use`, `spawn_subagent` | `subagents.py` | retained one-shot subagent loop |
| `estimate_size`, `block_type`, `message_has_tool_use`, `is_tool_result_message`, `collect_tool_results`, `persist_large_output`, `tool_result_budget`, `snip_compact`, `micro_compact`, `write_transcript`, `summarize_history`, `compact_history`, `reactive_compact`, `prepare_context`, `build_user_content`, `inject_background_notifications`, `update_context` | `context.py` | retained context and compaction flow |
| `RecoveryState`, `retry_delay`, `with_retry`, `is_prompt_too_long_error` | `recovery.py` | retained recovery flow |
| `_bg_counter`, `background_tasks`, `background_results`, `background_lock`, `is_slow_operation`, `should_run_background`, `start_background_task`, `collect_background_results` | `background.py` | retained background-task flow |
| `CronJob`, `scheduled_jobs`, `cron_queue`, `cron_lock`, `_last_fired`, `_cron_field_matches`, `cron_matches`, `_validate_cron_field`, `validate_cron`, `save_durable_jobs`, `load_durable_jobs`, `schedule_job`, `cancel_job`, `cron_scheduler_loop`, `consume_cron_queue`, `run_schedule_cron`, `run_list_crons`, `run_cancel_cron` | `cron.py` | retained durable scheduling |
| `MessageBus`, `BUS`, `active_teammates`, `ProtocolState`, `pending_requests`, `new_request_id`, `match_response`, `consume_lead_inbox`, `IDLE_POLL_INTERVAL`, `IDLE_TIMEOUT`, `scan_unclaimed_tasks`, `idle_poll`, `spawn_teammate_thread`, `_teammate_submit_plan`, `run_request_shutdown`, `run_request_plan`, `run_review_plan`, `run_spawn_teammate`, `run_send_message`, `run_check_inbox` | `teams.py` | retained S15-S17 team/protocol flow |
| `BUILTIN_TOOLS`, `BUILTIN_HANDLERS` | `tools.py` / fixed definitions and handlers | retained fixed S01-S17 registry |
| `rounds_since_todo`, `agent_lock`, `call_llm`, `agent_loop` | `agent.py` | retained S20 agent-loop sequence |
| S20 `client.messages.create(...)` calls | `provider.py` / `SiliconFlowProvider.create(...)` | converted to the only OpenAI-compatible boundary |

## Provider contract

`SiliconFlowProvider.create(messages, system, tools, max_tokens, model=None)`
accepts the S20 message and tool shapes. It converts `tool_use` blocks into
Chat Completions `tool_calls`, turns `tool_result` blocks into `tool` messages,
and returns `ProviderResponse(content=[TextBlock | ToolUseBlock],
stop_reason=...)`. `length` is mapped to `max_tokens`; a function-call finish
maps to `tool_use`.

## Explicit exclusions

### S18 Worktree Isolation

| Deleted source item(s) | Reason |
| --- | --- |
| `WORKTREES_DIR`, `VALID_WT_NAME`, `validate_worktree_name`, `run_git`, `log_event`, `create_worktree`, `bind_task_to_worktree`, `_count_worktree_changes`, `remove_worktree`, `keep_worktree` | S18 is outside this split. |
| `Task.worktree` | S18-only task state is removed rather than stubbed. |
| `run_create_worktree`, `run_remove_worktree`, `run_keep_worktree` | S18-only tool handlers are removed. |
| teammate `wt_ctx`, `_wt_cwd`, and workspace wrappers | S18-only teammate isolation is removed. |

### S19 MCP Plugin

| Deleted source item(s) | Reason |
| --- | --- |
| `MCPClient`, `mcp_clients`, `_DISALLOWED_CHARS`, `normalize_mcp_name`, `_mock_server_docs`, `_mock_server_deploy`, `MOCK_SERVERS`, `connect_mcp`, `run_connect_mcp` | S19 MCP functionality is outside this split. |
| `assemble_tool_pool` dynamic MCP merge | The retained registry is fixed S01-S17 tools only. |
| `connect_mcp` tool/handler, `mcp__*` permission behavior, `connected_mcp` context/prompt content | S19-only dynamic discovery is removed. |
