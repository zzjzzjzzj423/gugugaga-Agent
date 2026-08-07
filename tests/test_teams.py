import json

import pytest

from simple_cc.planning import TaskStore
from simple_cc.models import ToolCall
from simple_cc.teams import Mailbox, ProtocolError, ProtocolStore, TeamManager


class DummyRuntime:
    def __init__(self, name):
        self.name = name
        self.queries = []

    def run_turn(self, query):
        self.queries.append(query)
        return f"{self.name} completed: {query}"


def test_mailbox_send_drain_and_peek(tmp_path):
    box = Mailbox(tmp_path / "mailboxes")
    box.send("lead", "alice", "hello")
    assert box.peek("alice")
    assert box.drain("alice")[0]["content"] == "hello"
    assert not box.peek("alice")


def test_plan_response_must_match_request_type():
    protocols = ProtocolStore()
    request = protocols.request("plan_approval", "alice", "lead", "plan")
    with pytest.raises(ProtocolError):
        protocols.resolve(request.id, "shutdown_response", True)


def test_claim_next_task_uses_dependency_rules(tmp_path):
    tasks = TaskStore(tmp_path / "tasks")
    first = tasks.create("schema")
    tasks.create("api", blocked_by=[first.id])
    manager = TeamManager(
        Mailbox(tmp_path / "mailboxes"), tasks, ProtocolStore(),
        runtime_factory=lambda name, role: DummyRuntime(name),
        poll_seconds=0.01, idle_timeout=0.05,
    )
    claimed = manager.claim_next("alice")
    assert claimed.id == first.id
    assert tasks.get(first.id).owner == "alice"


def test_graceful_shutdown_request_is_correlated(tmp_path):
    protocols = ProtocolStore()
    manager = TeamManager(
        Mailbox(tmp_path / "mailboxes"), TaskStore(tmp_path / "tasks"), protocols,
        runtime_factory=lambda name, role: DummyRuntime(name),
    )
    request_id = manager.request_shutdown("alice")
    message = manager.mailbox.drain("alice")[0]
    assert message["type"] == "shutdown_request"
    assert message["metadata"]["request_id"] == request_id


def test_teammate_permission_request_routes_to_lead(tmp_path):
    protocols = ProtocolStore()
    manager = TeamManager(
        Mailbox(tmp_path / "mailboxes"), TaskStore(tmp_path / "tasks"), protocols,
        runtime_factory=lambda name, role: DummyRuntime(name),
    )
    request_id = manager.request_permission(
        "alice", ToolCall("call_1", "bash", {"command": "git reset --hard"})
    )
    message = manager.mailbox.drain("lead")[0]
    assert message["type"] == "permission_request"
    assert message["metadata"]["request_id"] == request_id
    assert protocols.get(request_id).type == "permission"
