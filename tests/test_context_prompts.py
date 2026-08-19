from simple_cc.context import ContextManager, MemoryStore, SkillStore
from simple_cc.prompts import (
    PromptAssembler,
    ordinary_system_prompt,
    research_execution_prompt,
    subagent_system_prompt,
)
from simple_cc.research_models import ResearchPlan, ResearchRank


def test_skill_store_discovers_metadata_then_loads_body(tmp_path):
    root = tmp_path / "skills"
    skill = root / "review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: review\ndescription: Review code\n---\nFull instructions")
    store = SkillStore([root])
    assert "Review code" in store.list_text()
    assert "Full instructions" not in store.list_text()
    assert "Full instructions" in store.load("review")


def test_memory_store_persists_and_searches(tmp_path):
    memory = MemoryStore(tmp_path / "memory")
    memory.remember("python-style", "Use pathlib for paths")
    assert "pathlib" in memory.search("paths")


def test_large_tool_output_is_persisted(tmp_path):
    manager = ContextManager(tmp_path / "outputs", tmp_path / "transcripts", max_output_chars=100)
    messages = [{"role": "tool", "tool_call_id": "c1", "content": "x" * 500}]
    compacted = manager.apply_output_budget(messages)
    assert len(compacted[0]["content"]) < 250
    assert "outputs" in compacted[0]["content"]


def test_compaction_keeps_assistant_tool_group_with_all_results(tmp_path):
    manager = ContextManager(
        tmp_path / "outputs", tmp_path / "transcripts", max_messages=3
    )
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
        {"role": "assistant", "content": "done"},
    ]
    compacted = manager.compact(messages, "summary")
    kept = compacted[1:]
    assert kept[0]["role"] == "user"
    assert kept[1]["role"] == "assistant"
    assert [m.get("tool_call_id") for m in kept if m["role"] == "tool"] == ["c1", "c2"]


def test_compacted_history_recompacts_after_new_tail_exceeds_budget(tmp_path):
    manager = ContextManager(
        tmp_path / "outputs", tmp_path / "transcripts", max_messages=3
    )
    first = manager.compact(
        [{"role": "user", "content": str(i)} for i in range(5)], "first summary"
    )
    assert not manager.needs_compaction(first)
    first.extend({"role": "user", "content": f"new-{i}"} for i in range(4))
    assert manager.needs_compaction(first)


def test_prompt_assembler_includes_named_runtime_sections():
    prompt = PromptAssembler().build({
        "identity": "alice (reviewer)",
        "workspace": "C:/repo", "tools": "bash, read_file", "skills": "review",
        "memory": "Use pathlib", "tasks": "task_1", "team": "alice: idle",
    })
    for heading in ("Workspace", "Tools", "Skills", "Memory", "Tasks", "Team", "Safety"):
        assert f"## {heading}" in prompt
    assert "alice (reviewer)" in prompt
    assert "workspace agent" in prompt
    assert "financial research agent" not in prompt


def test_subagent_prompt_has_general_workspace_identity():
    prompt = subagent_system_prompt()
    assert "workspace subagent" in prompt
    assert "financial research subagent" not in prompt


def test_ordinary_prompt_is_not_financial_research_only():
    prompt = ordinary_system_prompt({"workspace": "C:/repo", "tools": "read_file"})

    assert "workspace agent" in prompt
    assert "financial research agent" not in prompt
    assert "read at least two sources" not in prompt


def test_research_execution_prompt_contains_plan_and_gaps():
    plan = ResearchPlan(
        ResearchRank.STANDARD,
        ("first-party facts", "independent impact analysis"),
        "requires corroboration",
    )

    prompt = research_execution_prompt(
        {"workspace": "C:/repo", "tools": "web_search, web_fetch"},
        question="How did the event affect the company?",
        cutoff="2025-05-01",
        plan=plan,
        gaps=("second direction lacks evidence",),
        remaining_rounds=8,
    )

    assert "standard" in prompt
    assert "second direction lacks evidence" in prompt
    assert "2025-05-01" in prompt
    assert "8" in prompt
