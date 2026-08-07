from simple_cc.context import ContextManager, MemoryStore, SkillStore
from simple_cc.prompts import PromptAssembler


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


def test_prompt_assembler_includes_named_runtime_sections():
    prompt = PromptAssembler().build({
        "workspace": "C:/repo", "tools": "bash, read_file", "skills": "review",
        "memory": "Use pathlib", "tasks": "task_1", "team": "alice: idle",
    })
    for heading in ("Workspace", "Tools", "Skills", "Memory", "Tasks", "Team", "Safety"):
        assert f"## {heading}" in prompt

