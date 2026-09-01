from __future__ import annotations

import threading
import time
from collections import deque

import pytest

from gugugaga import config, tasks
import gugugaga.teams as teams
from gugugaga.context_modes import (
    SessionContextConfig,
    SessionContextCoordinator,
    validate_tool_protocol,
)
from gugugaga.permissions import PermissionPolicy
from gugugaga.provider import ProviderResponse, TextBlock, ToolUseBlock


SOURCE_TEAM_API = {
    "MessageBus",
    "ProtocolState",
    "active_teammates",
    "consume_lead_inbox",
    "idle_poll",
    "match_response",
    "pending_requests",
    "run_request_plan",
    "run_request_shutdown",
    "run_review_plan",
    "scan_unclaimed_tasks",
    "set_team_provider",
    "spawn_teammate_thread",
}


def require_source_team_api() -> None:
    missing = sorted(name for name in SOURCE_TEAM_API if not hasattr(teams, name))
    assert not missing, f"missing S15-S17 team API: {', '.join(missing)}"


@pytest.fixture(autouse=True)
def isolated_team_state(tmp_path, monkeypatch):
    original_workspace = config.WORKDIR
    config.configure_workspace(tmp_path)
    if hasattr(teams, "active_teammates"):
        teams.active_teammates.clear()
        teams._teammate_states.clear()
        teams._teammate_stop_events.clear()
    if hasattr(teams, "pending_requests"):
        teams.pending_requests.clear()
    if hasattr(teams, "MessageBus"):
        monkeypatch.setattr(teams, "BUS", teams.MessageBus())
    teams._lead_inbox_event.clear()
    if hasattr(teams, "set_team_provider"):
        teams.set_team_provider(None)
    teams._lead_inbox_event.clear()
    yield
    if hasattr(teams, "active_teammates"):
        teams.active_teammates.clear()
        teams._teammate_states.clear()
        teams._teammate_stop_events.clear()
    if hasattr(teams, "pending_requests"):
        teams.pending_requests.clear()
    if hasattr(teams, "set_team_provider"):
        teams.set_team_provider(None)
    config.configure_workspace(original_workspace)


def test_message_bus_delivers_each_mailbox_in_fifo_order_once():
    require_source_team_api()
    bus = teams.MessageBus()

    bus.send("lead", "alice", "first")
    bus.send("bob", "alice", "second", "status", {"sequence": 2})
    bus.send("lead", "alice", "third")

    delivered = bus.read_inbox("alice")
    assert [(item["from"], item["content"]) for item in delivered] == [
        ("lead", "first"),
        ("bob", "second"),
        ("lead", "third"),
    ]
    assert delivered[1]["type"] == "status"
    assert delivered[1]["metadata"] == {"sequence": 2}
    assert bus.read_inbox("alice") == []


@pytest.mark.parametrize(
    "malicious_name",
    [
        "../outside",
        r"..\outside",
        "/tmp/outside",
        r"C:\outside\mailbox",
        r"C:outside",
        r"\\server\share\mailbox",
        "alice/bob",
        r"alice\bob",
        "alice:stream",
        ".",
        "..",
        "alice name",
    ],
)
def test_message_bus_rejects_unsafe_mailbox_names_without_external_io(
    tmp_path, malicious_name
):
    bus = teams.MessageBus()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"sentinel": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid agent name"):
        bus.send("lead", malicious_name, "escape")
    with pytest.raises(ValueError, match="invalid agent name"):
        bus.send(malicious_name, "lead", "escape")
    with pytest.raises(ValueError, match="invalid agent name"):
        bus.read_inbox(malicious_name)

    assert outside.read_text(encoding="utf-8") == '{"sentinel": true}\n'
    assert not config.MAILBOX_DIR.exists()


def test_message_bus_accepts_only_simple_identifier_mailbox_names():
    bus = teams.MessageBus()

    bus.send("lead", "alice-1_test", "safe")

    assert bus.read_inbox("alice-1_test")[0]["content"] == "safe"


def test_protocol_response_rejects_wrong_type_and_request_id():
    require_source_team_api()
    request = teams.ProtocolState(
        request_id="req_plan",
        type="plan_approval",
        sender="alice",
        target="lead",
        status="pending",
        payload="Inspect, edit, verify.",
    )
    teams.pending_requests[request.request_id] = request

    teams.match_response("shutdown_response", "req_plan", True)
    teams.match_response("plan_approval_response", "req_other", True)

    assert teams.pending_requests["req_plan"].status == "pending"


def test_protocol_response_requires_the_expected_sender_and_target():
    request = teams.ProtocolState(
        request_id="req_shutdown",
        type="shutdown",
        sender="lead",
        target="alice",
        status="pending",
        payload="",
    )
    teams.pending_requests[request.request_id] = request

    assert not teams.match_response(
        "shutdown_response",
        request.request_id,
        True,
        sender="mallory",
        target="lead",
    )
    assert request.status == "pending"
    assert teams.match_response(
        "shutdown_response",
        request.request_id,
        True,
        sender="alice",
        target="lead",
    )
    assert request.status == "approved"


def test_plan_review_rejects_non_plan_and_terminal_requests():
    shutdown = teams._create_protocol_request(
        "shutdown", "lead", "alice", ""
    )
    assert teams.run_review_plan(shutdown.request_id, True) == (
        f"Request {shutdown.request_id} is not a plan approval request"
    )
    assert shutdown.status == "pending"

    plan = teams._create_protocol_request(
        "plan_approval", "alice", "lead", "plan"
    )
    plan.status = "expired"
    assert teams.run_review_plan(plan.request_id, True) == (
        f"Request {plan.request_id} is already expired"
    )
    assert plan.status == "expired"


def test_concurrent_colliding_request_ids_preserve_both_protocol_states(
    monkeypatch,
):
    require_source_team_api()
    original_new_request_id = teams.new_request_id
    generated = deque([17, 17, 18])
    generated_lock = threading.Lock()
    release_duplicate_ids = threading.Barrier(2)

    def deterministic_randint(_low: int, _high: int) -> int:
        with generated_lock:
            return generated.popleft()

    def synchronize_after_unreserved_id() -> str:
        request_id = original_new_request_id()
        release_duplicate_ids.wait(timeout=2)
        return request_id

    monkeypatch.setattr(teams.random, "randint", deterministic_randint)
    monkeypatch.setattr(teams, "new_request_id", synchronize_after_unreserved_id)
    results: list[str] = []

    workers = [
        threading.Thread(
            target=lambda name=name: results.append(
                teams._teammate_submit_plan(name, f"{name} plan")
            )
        )
        for name in ("alice", "bob")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert not any(worker.is_alive() for worker in workers)
    assert sorted(results) == [
        "Plan submitted (req_000017)",
        "Plan submitted (req_000018)",
    ]
    assert set(teams.pending_requests) == {"req_000017", "req_000018"}
    assert {state.sender for state in teams.pending_requests.values()} == {
        "alice",
        "bob",
    }


def test_plan_submission_and_approval_are_correlated_and_routed():
    require_source_team_api()

    result = teams._teammate_submit_plan("alice", "Inspect, edit, verify.")
    request_id = result.removeprefix("Plan submitted (").removesuffix(")")
    lead_message = teams.consume_lead_inbox()

    assert lead_message[0]["type"] == "plan_approval_request"
    assert lead_message[0]["metadata"]["request_id"] == request_id
    assert teams.pending_requests[request_id].status == "pending"

    assert teams.run_review_plan(request_id, True) == "Plan approved"
    reply = teams.BUS.read_inbox("alice")
    assert len(reply) == 1
    assert reply[0]["id"].startswith("msg_")
    assert {
        key: reply[0][key]
        for key in ("from", "to", "content", "type", "metadata")
    } == {
        "from": "lead",
        "to": "alice",
        "content": "Approved",
        "type": "plan_approval_response",
        "metadata": {"request_id": request_id, "approve": True},
    }
    assert teams.pending_requests[request_id].status == "approved"


def test_shutdown_request_is_acknowledged_and_routed_to_its_request(monkeypatch):
    require_source_team_api()
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1)

    teams.active_teammates["alice"] = True
    assert teams.run_request_shutdown("alice") == "Shutdown request sent to alice"
    request_id = next(iter(teams.pending_requests))

    assert teams.idle_poll("alice", [], "alice", "developer") == "shutdown"
    acknowledgement = teams.consume_lead_inbox(route_protocol=True)

    assert acknowledgement[0]["type"] == "shutdown_response"
    assert acknowledgement[0]["metadata"] == {
        "request_id": request_id,
        "approve": True,
    }
    assert teams.pending_requests[request_id].status == "approved"


def test_reserved_lead_name_cannot_be_spawned_or_targeted():
    class UnusedProvider:
        pass

    teams.set_team_provider(UnusedProvider())

    assert teams.spawn_teammate_thread("lead", "developer", "steal inbox") == (
        "Error: reserved teammate name: lead"
    )
    assert teams.run_send_message("lead", "orphan") == (
        "Error: reserved teammate name: lead"
    )
    assert "lead" not in teams.active_teammates


def test_check_inbox_returns_complete_content_before_acknowledging():
    content = "begin-" + ("x" * 500) + "-end"
    teams.BUS.send("alice", "lead", content, "result")

    rendered = teams.run_check_inbox()

    assert content in rendered
    assert rendered.endswith("-end")
    assert teams.run_check_inbox() == "(inbox empty)"


def test_lead_delivery_only_acks_after_explicit_success():
    teams.BUS.send("alice", "lead", "completed", "result")

    batch = teams.claim_lead_inbox()

    assert batch.messages[0]["content"] == "completed"
    assert batch.path is not None and batch.path.exists()
    assert not (config.MAILBOX_DIR / "lead.jsonl").exists()

    teams.nack_lead_inbox(batch, "retry")
    assert (config.MAILBOX_DIR / "lead.jsonl").exists()

    retried = teams.claim_lead_inbox()
    teams.ack_lead_inbox(retried)
    assert not retried.path.exists()
    assert not (config.MAILBOX_DIR / "lead.jsonl").exists()


def test_team_results_errors_and_plan_requests_emit_unread_events(monkeypatch):
    observed = []
    monkeypatch.setattr(
        teams, "notify", lambda event_type, payload: observed.append((event_type, payload))
    )

    teams.BUS.send("alice", "lead", "done", "result", {"task_id": "task_1"})
    teams.BUS.send("alice", "lead", "failed", "error")
    teams.BUS.send("alice", "lead", "review", "plan_approval_request")
    teams.BUS.send("alice", "lead", "ordinary", "message")

    unread = [item for item in observed if item[0] == "team_inbox_unread"]
    assert [item[1]["message_type"] for item in unread] == [
        "result",
        "error",
        "plan_approval_request",
    ]
    assert unread[0][1]["task_id"] == "task_1"
    assert teams._lead_inbox_event.is_set()


def test_unacknowledged_mailbox_batch_is_recovered_after_restart():
    first_bus = teams.MessageBus()
    first_bus.send("lead", "alice", "recover me")
    claimed = first_bus.claim_inbox("alice")
    assert claimed.messages[0]["content"] == "recover me"

    restarted_bus = teams.MessageBus()
    recovered = restarted_bus.claim_inbox("alice")
    assert recovered.messages[0]["id"] == claimed.messages[0]["id"]
    restarted_bus.ack_inbox(recovered)

    assert restarted_bus.read_inbox("alice") == []


def test_protocol_requests_reload_from_durable_state():
    request = teams._create_protocol_request(
        "plan_approval", "alice", "lead", "durable plan"
    )
    assert (config.MAILBOX_DIR / "protocol-requests.json").exists()

    teams.pending_requests.clear()
    teams._protocol_workspace = None
    teams._ensure_protocol_state_loaded()

    assert teams.pending_requests[request.request_id].payload == "durable plan"


def test_inactive_teammate_does_not_receive_orphan_control_messages():
    assert teams.run_request_shutdown("missing") == (
        "Error: teammate 'missing' is not active"
    )
    assert teams.run_request_plan("missing", "work") == (
        "Error: teammate 'missing' is not active"
    )
    assert teams.run_send_message("missing", "work") == (
        "Error: teammate 'missing' is not active"
    )
    assert not (config.MAILBOX_DIR / "missing.jsonl").exists()


def test_auto_claim_injects_full_task_description(monkeypatch):
    task = tasks.create_task(
        "api",
        description="Implement the complete API contract and its edge cases.",
    )
    messages = []
    work_state = {}
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.1)
    teams.update_team_settings(True)

    assert teams.idle_poll(
        "alice", messages, "alice", "developer", work_state=work_state
    ) == "work"

    assert task.id in messages[0]["content"]
    assert task.description in messages[0]["content"]
    assert work_state["task_id"] == task.id
    assert tasks.load_task(task.id).owner == "alice"


def test_workspace_auto_claim_defaults_off_and_persists(monkeypatch):
    task = tasks.create_task("manual by default")
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.005)

    assert teams.get_team_settings()["auto_claim_enabled"] is False
    assert teams.idle_poll("alice", [], "alice", "developer") == "timeout"
    assert tasks.load_task(task.id).status == "pending"

    teams.update_team_settings(True)
    assert teams.get_team_settings()["auto_claim_enabled"] is True


def test_default_idle_wait_has_no_lifetime_timeout(monkeypatch):
    stop_event = threading.Event()
    result = []
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)

    worker = threading.Thread(
        target=lambda: result.append(
            teams.idle_poll(
                "always-online",
                [],
                "always-online",
                "developer",
                stop_event=stop_event,
            )
        )
    )
    worker.start()
    time.sleep(0.05)

    assert teams.IDLE_TIMEOUT is None
    assert worker.is_alive()
    stop_event.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result == ["shutdown"]


def test_teammate_profile_persists_and_stopped_agent_can_restart(monkeypatch):
    class UnusedProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            raise AssertionError("an unassigned teammate must remain idle")

    teams.set_team_provider(UnusedProvider())
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)

    assert teams.run_spawn_teammate(
        "persistent-alice", "frontend developer", "Wait for user assignment."
    ).startswith("Teammate")
    profile_path = config.WORKDIR / ".gugugaga" / "team-agents.json"
    profile_data = profile_path.read_text(encoding="utf-8")
    assert '"persistent-alice"' in profile_data
    assert '"frontend developer"' in profile_data

    assert teams.stop_teammate("persistent-alice") == (
        "Stop requested for persistent-alice"
    )
    deadline = time.monotonic() + 1
    while "persistent-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "persistent-alice" not in teams.active_teammates

    teams._teammate_states.clear()
    persisted = {
        item["name"]: item for item in teams.list_teammate_states()
    }["persistent-alice"]
    assert persisted["status"] == "stopped"
    assert persisted["online"] is False
    assert persisted["role"] == "frontend developer"

    assert teams.restart_teammate("persistent-alice").startswith("Teammate")
    assert teams.active_teammates["persistent-alice"] is True
    assert teams.stop_teammate("persistent-alice").startswith("Stop requested")
    deadline = time.monotonic() + 1
    while "persistent-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)


def test_stopping_one_teammate_does_not_stop_others(monkeypatch):
    class UnusedProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            raise AssertionError("an unassigned teammate must remain idle")

    teams.set_team_provider(UnusedProvider())
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    assert teams.run_spawn_teammate("alice", "developer", "Wait.").startswith(
        "Teammate"
    )
    assert teams.run_spawn_teammate("bob", "reviewer", "Wait.").startswith(
        "Teammate"
    )

    assert teams.stop_teammate("alice") == "Stop requested for alice"
    deadline = time.monotonic() + 1
    while "alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "alice" not in teams.active_teammates
    assert teams.active_teammates["bob"] is True

    assert teams.stop_teammate("bob") == "Stop requested for bob"
    deadline = time.monotonic() + 1
    while "bob" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "bob" not in teams.active_teammates


def test_manual_assignment_is_claimed_even_when_auto_claim_is_off(monkeypatch):
    task = tasks.create_task("manual dispatch", description="full requirements")
    teams.active_teammates["alice"] = True
    teams._teammate_states["alice"] = {
        "name": "alice",
        "role": "developer",
        "status": "idle",
        "online": True,
        "current_task_id": None,
        "started_at": time.time(),
        "last_active_at": time.time(),
    }
    teams.assign_task_to_teammate(task.id, "alice")
    messages = []
    work_state = {}
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.1)

    assert teams.idle_poll(
        "alice", messages, "alice", "developer", work_state=work_state
    ) == "work"
    claimed = tasks.load_task(task.id)
    assert claimed.owner == "alice"
    assert claimed.assignee == "alice"
    assert task.description in messages[0]["content"]


def test_tool_spawned_teammate_waits_for_manual_assignment(monkeypatch):
    class SummaryProvider:
        def __init__(self):
            self.calls = 0

        def create(self, messages, system, tools, max_tokens, model=None):
            self.calls += 1
            return ProviderResponse(
                content=[TextBlock(text="Assigned task received.")],
                stop_reason="end_turn",
            )

    provider = SummaryProvider()
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1.0)
    task = tasks.create_task("manual only", description="Do not infer dispatch.")

    assert teams.run_spawn_teammate(
        "manual-alice", "developer", "You are the developer."
    ).startswith("Teammate")
    time.sleep(0.05)

    assert provider.calls == 0
    assert tasks.load_task(task.id).status == "pending"
    assert tasks.load_task(task.id).assignee is None

    teams.assign_task_to_teammate(task.id, "manual-alice")
    deadline = time.monotonic() + 1
    while provider.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    claimed = tasks.load_task(task.id)
    assert provider.calls == 1
    assert claimed.status == "in_progress"
    assert claimed.owner == "manual-alice"
    assert claimed.assignee == "manual-alice"
    assert teams.run_request_shutdown("manual-alice").startswith(
        "Shutdown request"
    )


def test_automatic_lead_inbox_turn_cannot_spawn_teammate():
    from gugugaga.observability import event_scope

    with event_scope(source="team_inbox", agent_type="main"):
        result = teams.run_spawn_teammate(
            "chain-agent", "developer", "Start more work"
        )

    assert result.startswith("Error:")
    assert "automatic Lead inbox Turn" in result
    assert "chain-agent" not in teams.active_teammates

    with event_scope(source="team_inbox", agent_type="main"):
        assert teams.run_stop_teammate("chain-agent").startswith("Error:")
        assert teams.run_restart_teammate("chain-agent").startswith("Error:")


def test_model_claim_is_denied_without_assignment_when_auto_claim_is_off(
    monkeypatch,
):
    class ClaimingProvider:
        def __init__(self, task_id):
            self.task_id = task_id
            self.requests = []

        def create(self, messages, system, tools, max_tokens, model=None):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                return ProviderResponse(
                    content=[
                        ToolUseBlock(
                            id="toolu_claim_unassigned",
                            name="claim_task",
                            input={"task_id": self.task_id},
                        )
                    ],
                    stop_reason="tool_use",
                )
            return ProviderResponse(
                content=[TextBlock(text="Did not claim unassigned work.")],
                stop_reason="end_turn",
            )

    task = tasks.create_task("must be assigned")
    provider = ClaimingProvider(task.id)
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)

    assert teams.spawn_teammate_thread(
        "guard-alice", "developer", "Try to claim the task"
    ).startswith("Teammate")
    deadline = time.monotonic() + 1
    while "guard-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)

    assert tasks.load_task(task.id).status == "pending"
    assert tasks.load_task(task.id).owner is None
    assert "manual assignment required" in str(
        provider.requests[1][-1]["content"]
    )


def test_task_completion_enforces_claim_owner():
    task = tasks.create_task("owned")
    assert tasks.claim_task(task.id, "alice").startswith("Claimed")

    assert tasks.complete_task(task.id, owner="bob") == (
        f"Task {task.id} is owned by alice, not bob"
    )
    assert tasks.load_task(task.id).status == "in_progress"
    assert tasks.complete_task(task.id, owner="alice").startswith("Completed")


def test_teammate_reports_each_completed_burst_without_idle_delay(monkeypatch):
    class SummaryProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            return ProviderResponse(
                content=[TextBlock(text="Immediate complete result.")],
                stop_reason="end_turn",
            )

    teams.set_team_provider(SummaryProvider())
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1.0)
    started = time.monotonic()
    assert teams.spawn_teammate_thread(
        "fast-alice", "developer", "Summarize"
    ).startswith("Teammate")

    deadline = time.monotonic() + 0.5
    result = []
    while not result and time.monotonic() < deadline:
        result = teams.consume_lead_inbox()
        if not result:
            time.sleep(0.01)

    assert result[0]["content"] == "Immediate complete result."
    assert time.monotonic() - started < 0.5
    assert "fast-alice" in teams.active_teammates
    assert teams.run_request_shutdown("fast-alice").startswith("Shutdown request")
    deadline = time.monotonic() + 1
    while "fast-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "fast-alice" not in teams.active_teammates


def test_two_agents_atomically_claim_only_one_unblocked_task():
    require_source_team_api()
    dependency = tasks.create_task("schema")
    assert tasks.claim_task(dependency.id, "lead").startswith("Claimed")
    assert tasks.complete_task(dependency.id, owner="lead").startswith("Completed")
    candidate = tasks.create_task("api", blockedBy=[dependency.id])
    assert [item["id"] for item in teams.scan_unclaimed_tasks()] == [candidate.id]

    barrier = threading.Barrier(3)
    results: dict[str, str] = {}

    def claim(owner: str) -> None:
        barrier.wait()
        results[owner] = tasks.claim_task(candidate.id, owner)

    workers = [
        threading.Thread(target=claim, args=(owner,))
        for owner in ("alice", "bob")
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2)

    claimed = tasks.load_task(candidate.id)
    assert not any(worker.is_alive() for worker in workers)
    assert sum(value.startswith("Claimed") for value in results.values()) == 1
    assert claimed.status == "in_progress"
    assert claimed.owner in {"alice", "bob"}
    assert teams.scan_unclaimed_tasks() == []


class ScriptedContentBlockProvider:
    def __init__(self):
        self.responses = [
            ProviderResponse(
                content=[
                    ToolUseBlock(
                        id="toolu_write",
                        name="write_file",
                        input={"path": "teammate.txt", "content": "shared"},
                    )
                ],
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content=[TextBlock(text="Shared-workspace write complete.")],
                stop_reason="end_turn",
            ),
        ]
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def create(self, messages, system, tools, max_tokens, model=None):
        with self._lock:
            self.requests.append(
                {
                    "messages": list(messages),
                    "system": system,
                    "tools": list(tools),
                    "max_tokens": max_tokens,
                    "model": model,
                }
            )
            assert self.responses, "unexpected provider call"
            return self.responses.pop(0)


def test_teammate_uses_content_block_provider_in_selected_shared_workspace(
    monkeypatch,
):
    require_source_team_api()
    provider = ScriptedContentBlockProvider()
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1)

    assert teams.spawn_teammate_thread("alice", "developer", "Create the file") == (
        "Teammate 'alice' spawned as developer"
    )
    deadline = time.monotonic() + 3
    while "alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "alice" not in teams.active_teammates
    assert (config.WORKDIR / "teammate.txt").read_text() == "shared"
    assert len(provider.requests) == 2
    assert str(config.WORKDIR) in provider.requests[0]["system"]
    assert "worktree" not in provider.requests[0]["system"].lower()
    assert any(
        block.get("type") == "tool_result"
        and block.get("tool_use_id") == "toolu_write"
        for block in provider.requests[1]["messages"][-1]["content"]
    )
    assert teams.consume_lead_inbox()[-1]["content"] == (
        "Shared-workspace write complete."
    )


def test_teammate_dispatch_uses_lead_permission_and_hook_boundary(
    monkeypatch,
):
    require_source_team_api()
    (config.WORKDIR / "input.txt").write_text("source", encoding="utf-8")
    provider = ScriptedContentBlockProvider()
    provider.responses = [
        ProviderResponse(
            content=[
                ToolUseBlock(
                    id="toolu_bash",
                    name="bash",
                    input={"command": "echo denied> bash-marker.txt"},
                ),
                ToolUseBlock(
                    id="toolu_write",
                    name="write_file",
                    input={"path": "write-marker.txt", "content": "denied"},
                ),
                ToolUseBlock(
                    id="toolu_read",
                    name="read_file",
                    input={"path": "input.txt"},
                ),
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            content=[TextBlock(text="Used the permitted read only.")],
            stop_reason="end_turn",
        ),
    ]
    events: list[str] = []

    def hooks(event, block, *args):
        events.append(f"{event}:{block.name}")
        if event == "PreToolUse" and block.name == "write_file":
            return "Permission denied by teammate test hook"
        return None

    def approve(call):
        events.append(f"approval:{call.name}")
        return False

    monkeypatch.setattr(teams, "trigger_hooks", hooks, raising=False)
    teams.set_team_provider(provider, PermissionPolicy(), approve)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1)

    assert teams.spawn_teammate_thread(
        "secure-alice", "developer", "Use the requested tools"
    ).startswith("Teammate")
    deadline = time.monotonic() + 3
    while "secure-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "secure-alice" not in teams.active_teammates
    assert not (config.WORKDIR / "bash-marker.txt").exists()
    assert not (config.WORKDIR / "write-marker.txt").exists()
    assert provider.requests[1]["messages"][-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_bash",
            "content": (
                "Permission denied for tool 'bash'. Choose a safer approach."
            ),
        },
        {
            "type": "tool_result",
            "tool_use_id": "toolu_write",
            "content": "Permission denied by teammate test hook",
        },
        {
            "type": "tool_result",
            "tool_use_id": "toolu_read",
            "content": "source",
        },
    ]
    assert events == [
        "PreToolUse:bash",
        "approval:bash",
        "PreToolUse:write_file",
        "PreToolUse:read_file",
        "PostToolUse:read_file",
    ]


class LateWriteProvider:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests: list[dict] = []

    def create(self, messages, system, tools, max_tokens, model=None):
        self.requests.append({"messages": list(messages), "system": system})
        self.entered.set()
        assert self.release.wait(timeout=3)
        return ProviderResponse(
            content=[
                ToolUseBlock(
                    id="toolu_late_write",
                    name="write_file",
                    input={"path": "late-write.txt", "content": "too late"},
                )
            ],
            stop_reason="tool_use",
        )


def test_app_close_discards_late_teammate_provider_tool_response(monkeypatch):
    require_source_team_api()
    from gugugaga.__main__ import build_runtime
    from gugugaga.config import Settings

    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    provider = LateWriteProvider()
    app = build_runtime(Settings.from_env(config.WORKDIR), provider=provider)
    assert teams.spawn_teammate_thread(
        "late-alice", "developer", "Wait before writing"
    ).startswith("Teammate")
    assert provider.entered.wait(timeout=1)
    close_results = []
    closer = threading.Thread(
        target=lambda: close_results.append(app.close(timeout=2))
    )
    closer.start()
    assert teams._teammate_stop_event.wait(timeout=1)

    provider.release.set()
    closer.join(timeout=3)

    assert not closer.is_alive()
    assert close_results[0].stopped
    assert close_results[0].live_threads == ()
    assert not (config.WORKDIR / "late-write.txt").exists()
    assert len(provider.requests) == 1


def test_runtime_bootstrap_installs_provider_before_source_spawn_handler(
    monkeypatch,
):
    require_source_team_api()
    from gugugaga.__main__ import build_runtime
    from gugugaga.config import Settings
    from gugugaga.tools import TOOL_HANDLERS

    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1)
    settings = Settings.from_env(config.WORKDIR)
    provider = ScriptedContentBlockProvider()
    app = build_runtime(settings, provider=provider)
    try:
        assert app.runtime.provider is provider
        task = tasks.create_task("runtime manual dispatch")
        assert TOOL_HANDLERS["spawn_teammate"](
            name="runtime-alice",
            role="developer",
            prompt="Create the file",
        ) == "Teammate 'runtime-alice' spawned as developer"
        time.sleep(0.05)
        assert provider.requests == []
        teams.assign_task_to_teammate(task.id, "runtime-alice")

        deadline = time.monotonic() + 3
        while (
            "runtime-alice" in teams.active_teammates
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        assert "runtime-alice" not in teams.active_teammates
        assert (config.WORKDIR / "teammate.txt").read_text() == "shared"
        assert len(provider.requests) == 2
    finally:
        app.close()


def test_plan_wait_preserves_ordinary_messages_until_approval(monkeypatch):
    provider = ScriptedContentBlockProvider()
    provider.responses = [
        ProviderResponse(
            content=[
                ToolUseBlock(
                    id="toolu_plan",
                    name="submit_plan",
                    input={"plan": "Inspect, edit, verify."},
                )
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            content=[TextBlock(text="Approved plan and urgent message handled.")],
            stop_reason="end_turn",
        ),
    ]
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)
    monkeypatch.setattr(teams, "PLAN_APPROVAL_TIMEOUT", 1.0)

    assert teams.spawn_teammate_thread(
        "plan-alice", "developer", "Submit a plan"
    ).startswith("Teammate")
    deadline = time.monotonic() + 2
    while not teams.BUS.read_inbox("lead") and time.monotonic() < deadline:
        time.sleep(0.01)
    request_id = next(iter(teams.pending_requests))
    teams.run_send_message("plan-alice", "Urgent requirement from Lead")
    assert teams.run_review_plan(request_id, True) == "Plan approved"

    deadline = time.monotonic() + 2
    while "plan-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "plan-alice" not in teams.active_teammates
    second_request = provider.requests[1]["messages"]
    assert any(
        "Urgent requirement from Lead" in str(message.get("content", ""))
        for message in second_request
    )


def test_plan_wait_expires_and_resumes_the_teammate(monkeypatch):
    provider = ScriptedContentBlockProvider()
    provider.responses = [
        ProviderResponse(
            content=[
                ToolUseBlock(
                    id="toolu_plan_timeout",
                    name="submit_plan",
                    input={"plan": "Wait forever."},
                )
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            content=[TextBlock(text="Stopped safely after approval timeout.")],
            stop_reason="end_turn",
        ),
    ]
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)
    monkeypatch.setattr(teams, "PLAN_APPROVAL_TIMEOUT", 0.03)

    assert teams.spawn_teammate_thread(
        "timeout-alice", "developer", "Submit a plan"
    ).startswith("Teammate")
    deadline = time.monotonic() + 2
    while "timeout-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "timeout-alice" not in teams.active_teammates
    request = next(iter(teams.pending_requests.values()))
    assert request.status == "expired"
    assert any(
        "Plan approval timed out" in str(message.get("content", ""))
        for message in provider.requests[1]["messages"]
    )


def test_teammate_provider_failure_is_reported_as_error(monkeypatch):
    class FailingProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            raise ValueError("provider unavailable")

    teams.set_team_provider(FailingProvider())
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)

    assert teams.spawn_teammate_thread(
        "failure-alice", "developer", "Do the work"
    ).startswith("Teammate")
    deadline = time.monotonic() + 2
    while "failure-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)

    messages = teams.consume_lead_inbox()
    assert [message["type"] for message in messages] == ["error"]
    assert "provider unavailable" in messages[0]["content"]


def test_teammate_context_keeps_tool_pairs_beyond_twenty_messages(
    tmp_path, monkeypatch
):
    class LongToolProvider:
        def __init__(self):
            self.calls = 0
            self.message_counts = []

        def create(self, messages, system, tools, max_tokens, model=None):
            validate_tool_protocol(messages)
            self.message_counts.append(len(messages))
            self.calls += 1
            if self.calls <= 11:
                return ProviderResponse(
                    content=[
                        ToolUseBlock(
                            id=f"toolu_read_{self.calls}",
                            name="read_file",
                            input={"path": "missing.txt"},
                        )
                    ],
                    stop_reason="tool_use",
                )
            return ProviderResponse(
                content=[TextBlock(text="Long task completed safely.")],
                stop_reason="end_turn",
            )

    parent = SessionContextCoordinator(
        SessionContextConfig.parse("cc"),
        workspace=tmp_path,
        transcripts_dir=tmp_path / ".gugugaga" / "transcripts",
        memory_dir=tmp_path / ".gugugaga" / "memory",
        tool_results_dir=tmp_path / ".gugugaga" / "tool-results",
    )
    provider = LongToolProvider()
    teams.set_team_provider(
        provider,
        context_parent_resolver=lambda: parent,
        max_rounds_per_burst=20,
    )
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)

    assert teams.spawn_teammate_thread(
        "context-alice", "developer", "Read repeatedly"
    ).startswith("Teammate")
    deadline = time.monotonic() + 2
    while "context-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "context-alice" not in teams.active_teammates
    assert max(provider.message_counts) > 20
    assert teams.consume_lead_inbox()[-1]["type"] == "result"


def test_team_tool_definitions_and_handlers_are_bijective_and_prompted():
    require_source_team_api()
    from gugugaga.prompts import assemble_system_prompt
    from gugugaga.tools import TOOL_DEFINITIONS, TOOL_HANDLERS

    team_tools = {
        "spawn_teammate",
        "stop_teammate",
        "restart_teammate",
        "send_message",
        "check_inbox",
        "request_shutdown",
        "request_plan",
        "review_plan",
    }
    names = {definition["name"] for definition in TOOL_DEFINITIONS}

    assert names == set(TOOL_HANDLERS)
    assert team_tools <= names
    prompt = assemble_system_prompt({})
    assert all(name in prompt for name in team_tools)
    assert "create_worktree" not in prompt
    assert "connect_mcp" not in prompt
