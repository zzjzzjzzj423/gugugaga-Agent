from __future__ import annotations

import pytest

import simple_cc.config as config
import simple_cc.skills as skills
import simple_cc.tasks as tasks
from simple_cc.prompts import assemble_system_prompt
from simple_cc.tools import TOOL_DEFINITIONS, TOOL_HANDLERS, call_tool_handler
from simple_cc.workspace import run_edit, run_read


def test_file_tools_reject_workspace_escape_and_edit_exactly_once(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("alpha alpha", encoding="utf-8")

    assert run_read("../secret.txt", cwd=tmp_path).startswith(
        "Error: Path escapes workspace:"
    )
    assert run_edit("notes.txt", "alpha", "beta", cwd=tmp_path) == "Edited notes.txt"
    assert target.read_text(encoding="utf-8") == "beta alpha"

    assert run_edit("notes.txt", "missing", "unused", cwd=tmp_path) == (
        "Error: text not found in notes.txt"
    )
    assert target.read_text(encoding="utf-8") == "beta alpha"


def test_todo_normalization_accepts_serialized_lists_and_rejects_bad_status():
    normalized, error = tasks._normalize_todos(
        '[{"content": "inspect", "status": "in_progress"}]'
    )

    assert error is None
    assert normalized == [{"content": "inspect", "status": "in_progress"}]
    assert tasks._normalize_todos(
        [{"content": "inspect", "status": "started"}]
    ) == (None, "Error: todos[0] has invalid status 'started'")


def test_task_dependency_claim_gate_is_durable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TASKS_DIR", tmp_path / ".tasks")
    config.TASKS_DIR.mkdir()

    dependency = tasks.create_task("dependency")
    dependent = tasks.create_task("dependent", blockedBy=[dependency.id])

    assert tasks.claim_task(dependent.id) == (
        f"Cannot start — blocked by: ['{dependency.id}']"
    )
    assert tasks.load_task(dependent.id).status == "pending"

    assert tasks.claim_task(dependency.id).startswith("Claimed")
    assert tasks.complete_task(dependency.id).startswith("Completed")
    assert tasks.claim_task(dependent.id).startswith("Claimed")
    assert tasks.load_task(dependent.id).status == "in_progress"


@pytest.mark.parametrize(
    "malicious_id",
    [
        "../outside",
        r"..\outside",
        "/tmp/outside",
        r"C:\outside\task",
        r"C:outside",
        r"\\server\share\task",
        "task_123_0001/child",
        r"task_123_0001\child",
        "task_123_0001:stream",
        "task-not-generated",
    ],
)
def test_task_ids_cannot_escape_selected_tasks_directory(
    tmp_path, monkeypatch, malicious_id
):
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(config, "TASKS_DIR", tasks_dir)

    with pytest.raises(ValueError, match="invalid task id"):
        tasks._task_path(malicious_id)
    with pytest.raises(ValueError, match="invalid task id"):
        tasks.get_task_json(malicious_id)
    with pytest.raises(ValueError, match="invalid task id"):
        tasks.claim_task(malicious_id)
    with pytest.raises(ValueError, match="invalid task id"):
        tasks.complete_task(malicious_id)
    with pytest.raises(ValueError, match="invalid task id"):
        tasks.save_task(
            tasks.Task(
                malicious_id,
                "escape",
                "",
                "pending",
                None,
                [],
            )
        )

    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert list(tasks_dir.iterdir()) == []


def test_public_task_handlers_reject_non_generated_ids_without_io(
    tmp_path, monkeypatch
):
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(config, "TASKS_DIR", tasks_dir)

    for handler in (
        tasks.run_get_task,
        tasks.run_claim_task,
        tasks.run_complete_task,
    ):
        assert handler("../outside") == "Error: invalid task id: ../outside"

    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert list(tasks_dir.iterdir()) == []


def test_generated_task_id_is_the_only_accepted_task_path_shape(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "TASKS_DIR", tmp_path / ".tasks")
    task = tasks.create_task("safe")

    assert task.id.startswith("task_")
    assert tasks._task_path(task.id).parent == config.TASKS_DIR
    assert tasks.load_task(task.id) == task


def test_skill_metadata_is_read_only_from_frontmatter(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    reviewed = root / "review" / "SKILL.md"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_text(
        "---\nname: code-review\ndescription: Review changes\n---\nFull instructions",
        encoding="utf-8",
    )
    plain = root / "plain" / "SKILL.md"
    plain.parent.mkdir(parents=True)
    plain.write_text(
        "# Plain skill\nname: not-frontmatter\ndescription: not metadata",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "SKILLS_DIR", root)

    skills.scan_skills()

    assert list(skills.SKILL_REGISTRY) == ["plain", "code-review"]
    assert skills.SKILL_REGISTRY["plain"]["description"] == "Plain skill"
    assert "not-frontmatter" not in skills.SKILL_REGISTRY
    assert skills.load_skill("code-review").endswith("Full instructions")


def test_system_prompt_rebuilds_from_live_workspace_skills_and_memory(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    manifest = root / "review" / "SKILL.md"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "---\nname: review\ndescription: Review changes\n---\nInstructions",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(config, "SKILLS_DIR", root)
    monkeypatch.setattr(
        skills,
        "SKILL_REGISTRY",
        {
            "stale": {
                "name": "stale",
                "description": "Old workspace skill",
                "content": "old",
            }
        },
    )

    first = assemble_system_prompt({})
    second = assemble_system_prompt({"memories": "Prefer pathlib."})

    assert f"Working directory: {tmp_path}" in first
    assert "- review: Review changes" in first
    assert "Relevant memories:\nPrefer pathlib." not in first
    assert "Relevant memories:\nPrefer pathlib." in second
    assert "stale" not in second
    assert "compact" in second
    assert "schedule_cron" not in second
    assert "spawn_teammate" in second
    assert "create_worktree" not in second
    assert "connect_mcp" not in second


def test_fixed_foundation_registry_has_one_handler_per_definition():
    names = [definition["name"] for definition in TOOL_DEFINITIONS]

    assert len(names) == len(set(names))
    assert set(names) == set(TOOL_HANDLERS)
    assert {
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
    } <= set(names)
    assert not {"create_worktree", "remove_worktree", "keep_worktree", "connect_mcp"} & set(
        names
    )
    assert call_tool_handler(None, {}, "missing") == "Unknown: missing"
