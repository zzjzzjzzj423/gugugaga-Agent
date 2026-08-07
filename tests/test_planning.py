import pytest

from simple_cc.planning import TaskStore, TodoStore
from simple_cc.planning import Task


def test_todo_store_normalizes_single_in_progress_item():
    todos = TodoStore()
    todos.update([
        {"content": "one", "status": "in_progress"},
        {"content": "two", "status": "in_progress"},
    ])
    assert [item["status"] for item in todos.items] == ["in_progress", "pending"]


def test_claim_requires_completed_dependencies(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    schema = store.create("schema")
    api = store.create("api", blocked_by=[schema.id])
    assert "Blocked" in store.claim(api.id, "alice")
    assert "Claimed" in store.claim(schema.id, "alice")
    assert "Completed" in store.complete(schema.id)
    assert "Claimed" in store.claim(api.id, "bob")


def test_claim_rejects_existing_owner(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task = store.create("owned")
    store.claim(task.id, "alice")
    assert "cannot claim" in store.claim(task.id, "bob")


@pytest.mark.parametrize(
    "malicious_id",
    [
        "../outside",
        r"..\outside",
        "/tmp/outside",
        r"C:\outside\task",
        r"C:outside",
        r"\\server\share\task",
        "task_123_abcdef/child",
        "task-not-generated",
    ],
)
def test_compatibility_task_store_rejects_ids_before_path_io(
    tmp_path, malicious_id
):
    store = TaskStore(tmp_path / "tasks")
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid task id"):
        store._path(malicious_id)
    with pytest.raises(ValueError, match="invalid task id"):
        store.get(malicious_id)
    with pytest.raises(ValueError, match="invalid task id"):
        store.claim(malicious_id, "alice")
    with pytest.raises(ValueError, match="invalid task id"):
        store.complete(malicious_id)
    with pytest.raises(ValueError, match="invalid task id"):
        store._save(Task(malicious_id, "escape"))

    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert list(store.directory.iterdir()) == []
