from simple_cc.planning import TaskStore, TodoStore


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

