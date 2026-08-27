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

import pytest

from gugugaga import agent, config
from gugugaga.config import Settings
from gugugaga.provider import SiliconFlowProvider
from gugugaga.tools import BUILTIN_HANDLERS, BUILTIN_TOOLS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
S20_SOURCE = (
    PROJECT_ROOT.parent / "learn-claude-code" / "s20_comprehensive" / "code.py"
)
SOURCE_MAP = PROJECT_ROOT / "SOURCE_MAP.md"
S20_SHA256 = "9EACF2F2C6F6DBE3B31117008A1A0BE44F52EE29585E5AFA0F4126D8D964D213"

TARGET_MODULES = (
    "gugugaga",
    "gugugaga.__main__",
    "gugugaga.agent",
    "gugugaga.background",
    "gugugaga.config",
    "gugugaga.context",
    "gugugaga.context_modes",
    "gugugaga.cron",
    "gugugaga.hooks",
    "gugugaga.models",
    "gugugaga.observability",
    "gugugaga.permissions",
    "gugugaga.planning",
    "gugugaga.prompts",
    "gugugaga.provider",
    "gugugaga.recovery",
    "gugugaga.skills",
    "gugugaga.subagents",
    "gugugaga.tasks",
    "gugugaga.teams",
    "gugugaga.tools",
    "gugugaga.web",
    "gugugaga.web_config",
    "gugugaga.web_search",
    "gugugaga.workspace",
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
    "web_search",
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


def _module_symbol_claims(source: str) -> set[str]:
    claims = _top_level_names(source)
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        claims.add(node.name)
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                claims.add(f"{node.name}.{child.name}")
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = (
                    child.targets if isinstance(child, ast.Assign) else [child.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        claims.add(f"{node.name}.{target.id}")
                    elif (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        claims.add(f"{node.name}.{target.attr}")
    return claims


def _retained_mapping_rows(source_map: str) -> list[tuple[str, str]]:
    retained = source_map.split("## Provider contract", 1)[0]
    rows = []
    for line in retained.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) != 3 or columns[0] in {
            "S20 source item(s)",
            "---",
        }:
            continue
        rows.append((columns[0], columns[1]))
    return rows


def _validate_retained_mapping_rows(
    source_map: str, source_names: set[str], project_root: Path
) -> None:
    module_claims: dict[str, set[str]] = {}
    for source_cell, target_cell in _retained_mapping_rows(source_map):
        module_names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*\.py)", target_cell)
        assert module_names, f"mapping row has no target module: {source_cell}"
        row_claims: set[str] = set()
        for module_name in module_names:
            module_path = project_root / "gugugaga" / module_name
            assert module_path.is_file(), (
                f"mapping row target module does not exist: {module_name}"
            )
            claims = module_claims.setdefault(
                module_name,
                _module_symbol_claims(module_path.read_text(encoding="utf-8")),
            )
            row_claims.update(claims)

        explicit_symbol_claims = [
            claim.removesuffix("(...)")
            for claim in re.findall(r"`([^`]+)`", target_cell)
            if ".py" not in claim
        ]
        for claim in explicit_symbol_claims:
            assert claim in row_claims, (
                f"mapping row target symbol does not exist in {module_names}: "
                f"{claim}"
            )

        for source_name in set(re.findall(r"`([^`]+)`", source_cell)) & source_names:
            qualified_match = any(
                claim.rsplit(".", 1)[-1] == source_name
                for claim in explicit_symbol_claims
            )
            assert source_name in row_claims or qualified_match, (
                f"mapping row maps {source_name} to {module_names}, but that "
                "symbol is not present in the claimed row target"
            )


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
    state = workspace / ".gugugaga"
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
        for path in (PROJECT_ROOT / "gugugaga").glob("*.py")
        if path.name != "__init__.py"
    }
    assert package_modules == {
        name.removeprefix("gugugaga.")
        for name in TARGET_MODULES
        if name != "gugugaga"
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
            for path in (PROJECT_ROOT / "gugugaga").glob("*.py")
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
    _validate_retained_mapping_rows(source_map, source_names, PROJECT_ROOT)


def test_source_map_row_audit_rejects_symbol_found_only_in_another_module():
    wrong_row = (
        "| S20 source item(s) | Target module / symbol | Status |\n"
        "| --- | --- | --- |\n"
        "| `create_task` | `teams.py` | retained |\n"
        "\n## Provider contract\n"
    )

    with pytest.raises(AssertionError, match="create_task.*claimed row target"):
        _validate_retained_mapping_rows(wrong_row, {"create_task"}, PROJECT_ROOT)


def test_fixed_s01_s17_tool_definitions_and_handlers_are_a_bijection():
    definition_names = [definition["name"] for definition in BUILTIN_TOOLS]

    assert definition_names == list(RETAINED_TOOL_NAMES)
    assert len(definition_names) == len(set(definition_names))
    assert set(BUILTIN_HANDLERS) == set(definition_names)
    assert all(callable(BUILTIN_HANDLERS[name]) for name in definition_names)


def test_s18_s19_symbols_are_absent_from_implementation_but_documented_as_deleted():
    findings = []
    for path in sorted((PROJECT_ROOT / "gugugaga").glob("*.py")):
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
