from __future__ import annotations

import ast
import hashlib
import importlib
import io
import json
import re
import tokenize
from pathlib import Path
from types import SimpleNamespace

from simple_cc import agent, config
from simple_cc.config import Settings
from simple_cc.provider import SiliconFlowProvider
from simple_cc.tools import BUILTIN_HANDLERS, BUILTIN_TOOLS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
S20_SOURCE = (
    PROJECT_ROOT.parent / "learn-claude-code" / "s20_comprehensive" / "code.py"
)
SOURCE_MAP = PROJECT_ROOT / "SOURCE_MAP.md"
S20_SHA256 = "9EACF2F2C6F6DBE3B31117008A1A0BE44F52EE29585E5AFA0F4126D8D964D213"

TARGET_MODULES = (
    "simple_cc",
    "simple_cc.__main__",
    "simple_cc.agent",
    "simple_cc.background",
    "simple_cc.config",
    "simple_cc.context",
    "simple_cc.cron",
    "simple_cc.hooks",
    "simple_cc.models",
    "simple_cc.permissions",
    "simple_cc.planning",
    "simple_cc.prompts",
    "simple_cc.provider",
    "simple_cc.recovery",
    "simple_cc.skills",
    "simple_cc.subagents",
    "simple_cc.tasks",
    "simple_cc.teams",
    "simple_cc.tools",
    "simple_cc.workspace",
)

EXCLUDED_SOURCE_SYMBOLS = {
    "WORKTREES_DIR",
    "VALID_WT_NAME",
    "validate_worktree_name",
    "run_git",
    "log_event",
    "create_worktree",
    "bind_task_to_worktree",
    "_count_worktree_changes",
    "remove_worktree",
    "keep_worktree",
    "run_create_worktree",
    "run_remove_worktree",
    "run_keep_worktree",
    "MCPClient",
    "mcp_clients",
    "_DISALLOWED_CHARS",
    "normalize_mcp_name",
    "_mock_server_docs",
    "_mock_server_deploy",
    "MOCK_SERVERS",
    "connect_mcp",
    "assemble_tool_pool",
    "run_connect_mcp",
}

RETAINED_TOOL_NAMES = (
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "todo_write",
    "task",
    "load_skill",
    "compact",
    "create_task",
    "list_tasks",
    "get_task",
    "claim_task",
    "complete_task",
    "schedule_cron",
    "list_crons",
    "cancel_cron",
    "spawn_teammate",
    "send_message",
    "check_inbox",
    "request_shutdown",
    "request_plan",
    "review_plan",
)

FORBIDDEN_IMPLEMENTATION_TERMS = re.compile(
    r"worktree|mcpclient|connect_mcp|mcp__|connected_mcp", re.IGNORECASE
)
FORBIDDEN_IMPLEMENTATION_IDENTIFIERS = EXCLUDED_SOURCE_SYMBOLS | {
    "worktree",
    "wt_ctx",
    "_wt_cwd",
}


def _top_level_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        names.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )
    return names


def _first_column_symbols(markdown: str) -> set[str]:
    symbols: set[str] = set()
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        first_column = line.split("|", 2)[1]
        symbols.update(re.findall(r"`([^`]+)`", first_column))
    return symbols


def _implementation_findings(path: Path, source: str) -> list[str]:
    findings = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.NAME:
            continue
        if (
            token.string in FORBIDDEN_IMPLEMENTATION_IDENTIFIERS
            or FORBIDDEN_IMPLEMENTATION_TERMS.search(token.string)
        ):
            findings.append(
                f"{path.name}:{token.start[0]}: identifier {token.string}"
            )

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            excluded_symbol = next(
                (
                    symbol
                    for symbol in FORBIDDEN_IMPLEMENTATION_IDENTIFIERS
                    if symbol.casefold() in node.value.casefold()
                ),
                None,
            )
            matched_term = FORBIDDEN_IMPLEMENTATION_TERMS.search(node.value)
            if excluded_symbol or matched_term:
                findings.append(
                    f"{path.name}:{node.lineno}: string term "
                    f"{excluded_symbol or matched_term.group(0)}"
                )
    return findings


def _settings(workspace: Path) -> Settings:
    state = workspace / ".simple_cc"
    return Settings(
        workspace=workspace,
        state_dir=state,
        tasks_dir=state / "tasks",
        memory_dir=state / "memory",
        mailboxes_dir=state / "mailboxes",
        transcripts_dir=state / "transcripts",
        outputs_dir=state / "outputs",
        skills_dir=state / "skills",
        api_key="audit-key",
        model="audit-model",
    )


def _completion(
    *, finish_reason: str, content: str = "", calls: tuple[tuple, ...] = ()
):
    tool_calls = [
        SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )
        for call_id, name, arguments in calls
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ]
    )


class ScriptedOpenAITransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **request):
        self.requests.append(request)
        assert self.responses, "unexpected provider request"
        return self.responses.pop(0)


def test_all_target_modules_import_and_match_the_audited_module_set():
    package_modules = {
        path.stem
        for path in (PROJECT_ROOT / "simple_cc").glob("*.py")
        if path.name != "__init__.py"
    }
    assert package_modules == {
        name.removeprefix("simple_cc.")
        for name in TARGET_MODULES
        if name != "simple_cc"
    }
    assert [importlib.import_module(name).__name__ for name in TARGET_MODULES] == list(
        TARGET_MODULES
    )


def test_source_map_pins_baseline_and_classifies_every_top_level_source_block():
    baseline_bytes = S20_SOURCE.read_bytes()
    assert hashlib.sha256(baseline_bytes).hexdigest().upper() == S20_SHA256

    source_map = SOURCE_MAP.read_text(encoding="utf-8")
    declared_hash = re.search(
        r"Baseline SHA-256:\s*`([0-9A-Fa-f]{64})`", source_map
    )
    assert declared_hash is not None, (
        "SOURCE_MAP.md must pin the audited baseline SHA-256"
    )
    assert declared_hash.group(1).upper() == S20_SHA256

    source_names = _top_level_names(baseline_bytes.decode("utf-8"))
    target_names = set().union(
        *(
            _top_level_names(path.read_text(encoding="utf-8"))
            for path in (PROJECT_ROOT / "simple_cc").glob("*.py")
        )
    )
    retained_section, exclusions = source_map.split("## Explicit exclusions", 1)
    retained_table = retained_section.split("## Provider contract", 1)[0]
    documented_retained = _first_column_symbols(retained_table) & source_names
    documented_excluded = _first_column_symbols(exclusions) & source_names

    assert documented_retained == source_names - EXCLUDED_SOURCE_SYMBOLS
    assert documented_excluded == EXCLUDED_SOURCE_SYMBOLS
    assert documented_retained.isdisjoint(documented_excluded)
    assert documented_retained <= target_names


def test_fixed_s01_s17_tool_definitions_and_handlers_are_a_bijection():
    definition_names = [definition["name"] for definition in BUILTIN_TOOLS]

    assert definition_names == list(RETAINED_TOOL_NAMES)
    assert len(definition_names) == len(set(definition_names))
    assert set(BUILTIN_HANDLERS) == set(definition_names)
    assert all(callable(BUILTIN_HANDLERS[name]) for name in definition_names)


def test_s18_s19_symbols_are_absent_from_implementation_but_documented_as_deleted():
    findings = []
    for path in sorted((PROJECT_ROOT / "simple_cc").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        findings.extend(_implementation_findings(path, source))

    assert findings == []

    deletion_audit = SOURCE_MAP.read_text(encoding="utf-8").split(
        "## Explicit exclusions", 1
    )[1]
    assert _first_column_symbols(deletion_audit) & EXCLUDED_SOURCE_SYMBOLS == (
        EXCLUDED_SOURCE_SYMBOLS
    )


def test_provider_runs_two_sequential_tool_turns_through_scripted_transport(
    tmp_path, monkeypatch
):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    transport = ScriptedOpenAITransport(
        [
            _completion(
                finish_reason="tool_calls",
                calls=(("call_read", "read_file", {"path": "sample.txt"}),),
            ),
            _completion(
                finish_reason="tool_calls",
                calls=(("call_glob", "glob", {"pattern": "*.txt"}),),
            ),
            _completion(finish_reason="stop", content="Both checks completed."),
        ]
    )
    provider = SiliconFlowProvider(_settings(tmp_path), client=transport)
    original_workspace = config.WORKDIR
    config.configure_workspace(tmp_path)
    monkeypatch.setattr(agent, "client", provider)
    monkeypatch.setattr(agent, "rounds_since_todo", 0)
    messages = [{"role": "user", "content": "Inspect the sample twice."}]

    try:
        agent.agent_loop(messages, {})
    finally:
        config.configure_workspace(original_workspace)

    assert len(transport.requests) == 3
    assert [
        message["role"] for message in transport.requests[1]["messages"]
    ] == ["system", "user", "assistant", "tool"]
    assert transport.requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_read",
        "content": "alpha",
    }
    assert [
        message["role"] for message in transport.requests[2]["messages"]
    ] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert transport.requests[2]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_glob",
        "content": "sample.txt",
    }
    assert transport.requests[2]["tools"]
    assert not transport.responses
    assert messages[-1]["content"][0].text == "Both checks completed."
