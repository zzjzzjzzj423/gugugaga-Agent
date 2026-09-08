from __future__ import annotations

import json
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
        teams._teammate_profile_restart_pending.clear()
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
        teams._teammate_profile_restart_pending.clear()
    if hasattr(teams, "pending_requests"):
        teams.pending_requests.clear()
    if hasattr(teams, "set_team_provider"):
        teams.set_team_provider(None)
    config.configure_workspace(original_workspace)


def match_task(task, *members):
    current = tasks.load_task(task.id)
    return tasks.set_task_candidates(
        task.id, list(members), "Matched task requirements to member capabilities",
        expected_revision=current.matching_revision,
    )


def register_idle_teammate(name):
    teams._persist_teammate_profile(name, "developer", "Implement and test APIs")
    teams.active_teammates[name] = True
    teams._teammate_states[name] = {
        "name": name, "status": "idle", "online": True,
        "current_task_id": None, "dispatch_available": True,
    }


def test_runtime_claim_enforces_candidates_busy_state_and_manual_override():
    for name in ("alice", "bob", "outsider"):
        register_idle_teammate(name)
    teams.update_team_settings(True)
    task = match_task(tasks.create_task("API work"), "alice", "bob")
    assert "not a candidate" in teams.claim_task_for_teammate(task.id, "outsider")
    teams._teammate_states["alice"]["dispatch_available"] = False
    teams._teammate_states["bob"]["dispatch_available"] = False
    assert "not idle" in teams.claim_task_for_teammate(task.id, "alice")
    assert "not idle" in teams.claim_task_for_teammate(task.id, "bob")
    assert tasks.load_task(task.id).owner is None

    teams.assign_task_to_teammate(task.id, "outsider")
    teams._teammate_states["alice"]["dispatch_available"] = True
    assert "assigned to outsider" in teams.claim_task_for_teammate(task.id, "alice")
    assert teams.claim_task_for_teammate(task.id, "outsider").startswith("Claimed")
    assert tasks.load_task(task.id).owner == "outsider"


def test_manual_override_clears_previous_runtime_reservation():
    for name in ("alice", "bob"):
        register_idle_teammate(name)
    task = tasks.create_task("manual override")
    teams.assign_task_to_teammate(task.id, "alice")
    teams.assign_task_to_teammate(task.id, "bob")
    assert teams._teammate_states["alice"]["current_task_id"] is None
    assert "assigned to bob" in teams.claim_task_for_teammate(task.id, "alice")
    assert teams.claim_task_for_teammate(task.id, "bob").startswith("Claimed")


def test_runtime_claim_candidates_compete_atomically():
    for name in ("alice", "bob"):
        register_idle_teammate(name)
    teams.update_team_settings(True)
    task = match_task(tasks.create_task("one owner"), "alice", "bob")
    barrier = threading.Barrier(3)
    results = []

    def claim(name):
        barrier.wait()
        results.append(teams.claim_task_for_teammate(task.id, name))

    workers = [threading.Thread(target=claim, args=(name,)) for name in ("alice", "bob")]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert sum(result.startswith("Claimed") for result in results) == 1
    claimed = tasks.load_task(task.id)
    assert claimed.owner in {"alice", "bob"}
    assert claimed.assignee is None


def test_model_claim_can_follow_completion_but_cannot_take_a_second_active_task(monkeypatch):
    teams._persist_teammate_profile("alice", "developer", "Implement API tasks")
    first = match_task(tasks.create_task("first"), "alice")
    second = match_task(tasks.create_task("second"), "alice")
    outputs = []

    class ClaimProvider:
        calls = 0

        def create(self, messages, system, tools, max_tokens, model=None):
            self.calls += 1
            if self.calls > 1:
                outputs.extend(messages[-1]["content"])
            if self.calls == 1:
                return ProviderResponse(content=[
                    ToolUseBlock(id="claim_first", name="claim_task", input={"task_id": first.id}),
                    ToolUseBlock(id="claim_busy", name="claim_task", input={"task_id": second.id}),
                    ToolUseBlock(id="complete_first", name="complete_task", input={"task_id": first.id}),
                    ToolUseBlock(id="claim_second", name="claim_task", input={"task_id": second.id}),
                    ToolUseBlock(id="complete_second", name="complete_task", input={"task_id": second.id}),
                ], stop_reason="tool_use")
            return ProviderResponse(content=[TextBlock(text="Both completed sequentially")], stop_reason="end_turn")

    teams.set_team_provider(ClaimProvider())
    teams.update_team_settings(True)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.005)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.03)
    teams.spawn_teammate_thread("alice", "developer", "Implement API tasks", persist_profile=False)
    deadline = time.monotonic() + 1
    while "alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.005)
    assert tasks.load_task(first.id).status == "completed"
    assert tasks.load_task(second.id).status == "completed"
    by_id = {item["tool_use_id"]: item["content"] for item in outputs}
    assert by_id["claim_first"].startswith("Claimed")
    assert not by_id["claim_busy"].startswith("Claimed")
    assert by_id["claim_second"].startswith("Claimed")


def test_matching_context_is_read_only_and_runtime_marks_profile_reload_unavailable():
    register_idle_teammate("alice")
    task = match_task(tasks.create_task("API"), "alice")
    before = task.matching_revision
    for _ in range(3):
        assert teams.task_candidates_context()["tasks"][0]["matching_revision"] == before
        assert teams.scan_unclaimed_tasks("alice")[0]["id"] == task.id
    assert tasks.pending_matching_requests() == []
    teams._teammate_profile_restart_pending.add("alice")
    assert teams.list_teammate_states()[0]["dispatch_available"] is False


def test_matching_inbox_coalesces_changes_and_preserves_newer_revision_on_ack():
    teams._persist_teammate_profile("alice", "developer", "Implement APIs", ["read_file", "bash"])
    first = tasks.create_task("API")
    second = tasks.create_task("Schema")
    tasks.update_task(first.id, description="Updated API contract")
    batch = teams.claim_lead_inbox()
    messages = [item for item in batch.messages if item["type"] == "assignment_match_requested"]
    assert len(messages) == 1
    assert len(messages[0]["metadata"]["tasks"]) == 2
    assert messages[0]["metadata"]["members"][0]["prompt"] == "Implement APIs"
    assert "bash" in messages[0]["metadata"]["members"][0]["allowed_tools"]
    tasks.update_task(first.id, description="A newer contract arrived during matching")
    match_task(second, "alice")
    assert teams.ack_lead_inbox(batch) is True
    remaining = tasks.pending_matching_requests()
    assert [item["id"] for item in remaining] == [first.id]
    assert remaining[0]["matching_revision"] > messages[0]["metadata"]["matching_revisions"][first.id]

    retried = teams.claim_lead_inbox()
    teams.nack_lead_inbox(retried, "provider failed")
    assert tasks.pending_matching_requests()[0]["id"] == first.id
    assert tasks.load_task(second.id).matching_notified_revision > 0


def test_unresolved_matching_is_not_acknowledged_but_regular_mail_is_consumed():
    task = tasks.create_task("Needs candidates")
    teams.BUS.send("alice", "lead", "Completed other work", "result")
    batch = teams.claim_lead_inbox()
    assert len(batch.messages) == 2
    assert teams.ack_lead_inbox(batch) is False
    assert [item["id"] for item in tasks.pending_matching_requests()] == [task.id]
    assert tasks.load_task(task.id).matching_notified_revision == 0
    retry = teams.claim_lead_inbox()
    assert [item["type"] for item in retry.messages] == ["assignment_match_requested"]

    match_task(task)
    assert teams.ack_lead_inbox(retry) is True
    assert tasks.pending_matching_requests() == []


def test_matching_ack_accepts_tasks_taken_over_or_deleted():
    manual = tasks.create_task("Manual takeover")
    removed = tasks.create_task("Removed task")
    batch = teams.claim_lead_inbox()
    tasks.assign_task(manual.id, "alice")
    tasks.delete_task(removed.id)
    assert teams.ack_lead_inbox(batch) is True
    assert tasks.pending_matching_requests() == []


def test_check_inbox_matching_notice_includes_requirements_and_member_capabilities():
    teams._persist_teammate_profile("alice", "developer", "Special API instructions")
    task = tasks.create_task("API", description="Precise acceptance criteria")
    rendered = teams.run_check_inbox()
    assert task.id in rendered
    assert "Precise acceptance criteria" in rendered
    assert "Special API instructions" in rendered
    assert "matching_revisions" in rendered
    assert [item["id"] for item in tasks.pending_matching_requests()] == [task.id]


def test_profile_changes_invalidate_only_unclaimed_nonmanual_candidates():
    teams._persist_teammate_profile("alice", "developer", "Implement APIs")
    pending = match_task(tasks.create_task("pending"), "alice")
    manual = tasks.create_task("manual")
    tasks.assign_task(manual.id, "bob")
    active = match_task(tasks.create_task("active"), "alice")
    assert tasks.claim_task(active.id, "alice").startswith("Claimed")
    teams.update_teammate_profile("alice", role="designer", prompt="Draw icons", allowed_tools=["read_file"])
    updated = tasks.load_task(pending.id)
    assert updated.matching_status != "matched"
    assert updated.matching_revision > pending.matching_revision
    assert tasks.load_task(manual.id).assignee == "bob"
    assert tasks.load_task(active.id).owner == "alice"
    assert tasks.load_task(active.id).matching_revision == active.matching_revision
    assert "stale" in teams.run_set_task_candidates(
        pending.id, ["alice"], "Old API capability", pending.matching_revision,
    ).lower()
    assert "unknown candidate" in teams.run_set_task_candidates(
        pending.id, ["missing"], "Unknown member", updated.matching_revision,
    )


def test_profile_reload_preserves_pending_manual_dispatch_and_then_claims(monkeypatch):
    received = threading.Event()

    class SummaryProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            received.set()
            return ProviderResponse(content=[TextBlock(text="Assigned task received")], stop_reason="end_turn")

    task = tasks.create_task("Manually reserved")
    teams.set_team_provider(SummaryProvider())
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.005)
    teams.run_spawn_teammate("alice", "developer", "Original instructions")
    try:
        deadline = time.monotonic() + 1
        while teams._teammate_states["alice"]["status"] != "idle" and time.monotonic() < deadline:
            time.sleep(0.005)
        with teams._teammate_lock:
            teams.assign_task_to_teammate(task.id, "alice")
            updated = teams.update_teammate_profile("alice", prompt="Updated instructions")
            assert updated["apply_state"] == "restarting"
            assert tasks.load_task(task.id).status == "pending"
            assert tasks.load_task(task.id).assignee == "alice"
        assert received.wait(timeout=1)
        assert tasks.load_task(task.id).owner == "alice"
        assert teams._teammate_states["alice"]["configuration_updated_at"] == updated["updated_at"]
        assert teams.get_team_settings()["auto_claim_enabled"] is False
    finally:
        teams.stop_all_teammates(timeout=1)


@pytest.mark.parametrize("auto_claim", [False, True])
def test_new_member_triggers_matching_and_only_autoclaims_when_enabled(monkeypatch, auto_claim):
    class SummaryProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            return ProviderResponse(content=[TextBlock(text="Task received")], stop_reason="end_turn")

    task = match_task(tasks.create_task("Implement API"))
    teams.set_team_provider(SummaryProvider())
    teams.update_team_settings(auto_claim)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.005)
    teams.run_spawn_teammate("alice", "developer", "Implement APIs")
    try:
        batch = teams.claim_lead_inbox()
        request = next(item for item in batch.messages if item["type"] == "assignment_match_requested")
        revision = request["metadata"]["matching_revisions"][task.id]
        assert revision > task.matching_revision
        result = teams.run_set_task_candidates(task.id, ["alice"], "Developer can implement APIs", revision)
        assert not result.startswith("Error:")
        teams.ack_lead_inbox(batch)
        if auto_claim:
            deadline = time.monotonic() + 1
            while tasks.load_task(task.id).owner is None and time.monotonic() < deadline:
                time.sleep(0.005)
            assert tasks.load_task(task.id).owner == "alice"
        else:
            time.sleep(0.03)
            assert tasks.load_task(task.id).status == "pending"
            assert tasks.load_task(task.id).owner is None
    finally:
        teams.stop_all_teammates(timeout=1)


def test_mismatch_releases_task_and_blocks_remaining_tools_in_batch(monkeypatch):
    task = tasks.create_task("Requires unavailable expertise")

    class MismatchProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            return ProviderResponse(
                content=[
                    ToolUseBlock(id="mismatch", name="report_task_mismatch", input={
                        "task_id": task.id, "reason": "Required tool is unavailable",
                        "work_summary": "Reviewed the API contract",
                    }),
                    ToolUseBlock(id="write_after_release", name="write_file", input={
                        "path": "must-not-write.txt", "content": "stale task work",
                    }),
                ], stop_reason="tool_use",
            )

    teams.set_team_provider(MismatchProvider())
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.005)
    teams.run_spawn_teammate("alice", "developer", "Implement API")
    try:
        deadline = time.monotonic() + 1
        while teams._teammate_states["alice"]["status"] != "idle" and time.monotonic() < deadline:
            time.sleep(0.005)
        teams.assign_task_to_teammate(task.id, "alice")
        while not tasks.load_task(task.id).mismatch_reports and time.monotonic() < deadline:
            time.sleep(0.005)
        returned = tasks.load_task(task.id)
        assert returned.owner is None and returned.assignee is None
        assert returned.matching_status == "rematch_required"
        assert returned.mismatch_reports[0]["work_summary"] == "Reviewed the API contract"
        assert not (config.WORKDIR / "must-not-write.txt").exists()
        assert not teams.claim_task_for_teammate(task.id, "alice").startswith("Claimed")
    finally:
        teams.stop_all_teammates(timeout=1)


def test_legacy_agent_cannot_resume_mutations_from_a_message_after_mismatch(monkeypatch):
    task = tasks.create_task("Unsupported task")
    teams._persist_teammate_profile("alice", "developer", "Implement APIs")
    tasks.assign_task(task.id, "alice")
    outputs = []

    class LegacyMismatchProvider:
        calls = 0

        def create(self, messages, system, tools, max_tokens, model=None):
            self.calls += 1
            if self.calls == 1:
                return ProviderResponse(content=[
                    ToolUseBlock(id="claim", name="claim_task", input={"task_id": task.id}),
                    ToolUseBlock(id="mismatch", name="report_task_mismatch", input={
                        "task_id": task.id, "reason": "Tool unavailable",
                    }),
                ], stop_reason="tool_use")
            if self.calls == 2:
                return ProviderResponse(content=[ToolUseBlock(
                    id="stale_write", name="write_file", input={
                        "path": "stale-after-message.txt", "content": "unassigned",
                    },
                )], stop_reason="tool_use")
            outputs.extend(messages[-1]["content"])
            return ProviderResponse(content=[TextBlock(text="Waiting for reassignment")], stop_reason="end_turn")

    teams.set_team_provider(LegacyMismatchProvider())
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.005)
    teams.spawn_teammate_thread("alice", "developer", "Implement APIs", persist_profile=False)
    try:
        deadline = time.monotonic() + 1
        while not tasks.load_task(task.id).mismatch_reports and time.monotonic() < deadline:
            time.sleep(0.005)
        teams.BUS.send("lead", "alice", "An ordinary message after relinquishing")
        while not outputs and time.monotonic() < deadline:
            time.sleep(0.005)
        assert any("no active assigned task" in item["content"] for item in outputs)
        assert not (config.WORKDIR / "stale-after-message.txt").exists()
        assert tasks.load_task(task.id).owner is None
    finally:
        teams.stop_all_teammates(timeout=1)


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


def test_message_bus_records_content_free_routes_for_team_graph():
    bus = teams.MessageBus()

    message_id = bus.send(
        "lead",
        "alice",
        "private implementation detail",
        "task_assignment",
        {"task_id": "task_7"},
    )

    item = teams.list_team_communications()[-1]
    assert item == {
        "id": message_id,
        "from": "lead",
        "to": "alice",
        "type": "task_assignment",
        "ts": item["ts"],
        "task_id": "task_7",
        "interaction_id": None,
    }
    assert "content" not in item


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


@pytest.mark.parametrize("alias", ["lead", "Lead", "LEAD", "leader", "Leader", "main"])
def test_message_bus_normalizes_lead_aliases(alias):
    bus = teams.MessageBus()

    bus.send("alice", alias, "status", "result")
    batch = bus.claim_inbox("lead")

    assert batch.agent == "lead"
    assert batch.messages[0]["to"] == "lead"
    assert batch.messages[0]["content"] == "status"
    bus.ack_inbox(batch)


def test_lead_claim_recovers_legacy_leader_mailbox():
    bus = teams.MessageBus()
    config.MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
    legacy = config.MAILBOX_DIR / "Leader.jsonl"
    legacy.write_text(
        json.dumps(
            {
                "id": "msg_legacy_leader",
                "from": "alice",
                "to": "Leader",
                "content": "recover result",
                "type": "result",
                "ts": time.time(),
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert teams.signal_pending_lead_inbox() is True
    batch = bus.claim_inbox("lead")

    assert batch.messages[0]["to"] == "lead"
    assert batch.messages[0]["content"] == "recover result"
    assert not legacy.exists()
    bus.ack_inbox(batch)


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


@pytest.mark.parametrize("alias", ["lead", "Leader", "main"])
def test_reserved_lead_name_cannot_be_spawned_or_targeted(alias):
    class UnusedProvider:
        pass

    teams.set_team_provider(UnusedProvider())

    assert teams.spawn_teammate_thread(alias, "developer", "steal inbox") == (
        f"Error: reserved teammate name: {alias}"
    )
    assert teams.run_send_message(alias, "orphan") == (
        "Error: you are the Lead Agent; send_message cannot be used "
        "to send a message to yourself"
    )
    assert alias not in teams.active_teammates


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
    routed = [item for item in observed if item[0] == "team_message"]
    assert len(routed) == 4
    assert routed[0][1]["from_agent"] == "alice"
    assert routed[0][1]["to_agent"] == "lead"


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
    register_idle_teammate("alice")
    match_task(task, "alice")
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


def test_ordinary_message_wakes_assignment_only_teammate(monkeypatch):
    messages = []
    work_state = {}
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.1)
    teams.BUS.send("lead", "alice", "Introduce yourself")

    assert teams.idle_poll(
        "alice",
        messages,
        "alice",
        "developer",
        work_state=work_state,
        require_task=True,
    ) == "work"

    assert "Introduce yourself" in messages[0]["content"]
    assert work_state.get("task_id") is None


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


def test_teammate_profile_configuration_persists_locks_core_tools_and_resets():
    teams._persist_teammate_profile("alice", "developer", "Initial prompt")

    updated = teams.update_teammate_profile(
        "alice",
        role="reviewer",
        prompt="Review frontend changes",
        allowed_tools=["glob"],
    )

    assert updated["role"] == "reviewer"
    assert updated["initial_role"] == "developer"
    assert updated["prompt"] == "Review frontend changes"
    assert updated["initial_prompt"] == "Initial prompt"
    assert updated["allowed_tools"] == [*teams.TEAM_CORE_TOOLS, "glob"]
    assert updated["apply_state"] == "next_start"
    payload = json.loads(
        (config.WORKDIR / ".gugugaga" / "team-agents.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["version"] == 2

    reset = teams.update_teammate_profile("alice", reset=True)
    assert reset["role"] == "developer"
    assert reset["prompt"] == "Initial prompt"
    assert reset["allowed_tools"] == list(teams.TEAM_DEFAULT_ALLOWED_TOOLS)


def test_teammate_runtime_receives_only_configured_tools(monkeypatch):
    class ToolCaptureProvider:
        def __init__(self):
            self.tools = None

        def create(self, messages, system, tools, max_tokens, model=None):
            self.tools = tools
            return ProviderResponse(
                content=[TextBlock(text="Configured tools received.")],
                stop_reason="end_turn",
            )

    provider = ToolCaptureProvider()
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)

    assert teams.spawn_teammate_thread(
        "configured-alice",
        "developer",
        "Inspect the workspace",
        allowed_tools=["glob"],
    ).startswith("Teammate")
    deadline = time.monotonic() + 1
    while provider.tools is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert {tool["name"] for tool in provider.tools} == {
        *teams.TEAM_CORE_TOOLS,
        "glob",
    }
    assert "bash" not in {tool["name"] for tool in provider.tools}


def test_idle_teammate_configuration_update_restarts_with_saved_profile(monkeypatch):
    class UnusedProvider:
        def create(self, messages, system, tools, max_tokens, model=None):
            raise AssertionError("an idle teammate must not call the provider")

    teams.set_team_provider(UnusedProvider())
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    assert teams.run_spawn_teammate(
        "reload-alice", "developer", "Initial prompt"
    ).startswith("Teammate")
    deadline = time.monotonic() + 1
    while (
        teams._teammate_states.get("reload-alice", {}).get("status") != "idle"
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)

    updated = teams.update_teammate_profile(
        "reload-alice",
        role="reviewer",
        prompt="Updated prompt",
        allowed_tools=["glob"],
    )
    assert updated["apply_state"] == "restarting"

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = teams._teammate_states.get("reload-alice", {})
        if (
            state.get("status") == "idle"
            and state.get("configuration_updated_at") == updated["updated_at"]
        ):
            break
        time.sleep(0.01)
    state = teams._teammate_states["reload-alice"]
    assert state["role"] == "reviewer"
    assert state["active_allowed_tools"] == [*teams.TEAM_CORE_TOOLS, "glob"]
    assert state["configuration_updated_at"] == updated["updated_at"]
    assert teams.stop_teammate("reload-alice").startswith("Stop requested")
    deadline = time.monotonic() + 1
    while "reload-alice" in teams.active_teammates and time.monotonic() < deadline:
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


def test_tool_spawned_teammate_replies_to_ordinary_message_without_task(monkeypatch):
    class ConversationalProvider:
        def __init__(self):
            self.calls = 0
            self.requests = []

        def create(self, messages, system, tools, max_tokens, model=None):
            self.calls += 1
            self.requests.append(messages)
            return ProviderResponse(
                content=[TextBlock(text="Hello, I am Alice, the frontend developer.")],
                stop_reason="end_turn",
            )

    provider = ConversationalProvider()
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1.0)

    assert teams.run_spawn_teammate(
        "chat-alice", "frontend developer", "Wait for assigned work."
    ).startswith("Teammate")
    time.sleep(0.05)
    assert provider.calls == 0

    assert teams.run_send_message("chat-alice", "Please introduce yourself") == (
        "Sent to chat-alice"
    )
    deadline = time.monotonic() + 2
    replies = []
    while not replies and time.monotonic() < deadline:
        replies = teams.BUS.read_inbox("lead")
        if not replies:
            time.sleep(0.01)

    assert provider.calls == 1
    assert any(
        "Please introduce yourself" in str(message.get("content", ""))
        for message in provider.requests[0]
    )
    assert replies[-1]["type"] == "result"
    assert "Hello, I am Alice" in replies[-1]["content"]
    assert tasks.list_tasks() == []

    assert teams.run_request_shutdown("chat-alice").startswith("Shutdown request")
    deadline = time.monotonic() + 1
    while "chat-alice" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)


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
    teams._persist_teammate_profile("guard-alice", "developer", "Try to claim the task")
    match_task(task, "guard-alice")
    provider = ClaimingProvider(task.id)
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)

    assert teams.spawn_teammate_thread(
        "guard-alice", "developer", "Try to claim the task", persist_profile=False,
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
    match_task(task, "alice")
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
    match_task(dependency, "lead")
    assert tasks.claim_task(dependency.id, "lead").startswith("Claimed")
    assert tasks.complete_task(dependency.id, owner="lead").startswith("Completed")
    candidate = tasks.create_task("api", blockedBy=[dependency.id])
    match_task(candidate, "alice", "bob")
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


def test_teammate_sending_to_leader_alias_reaches_canonical_lead_inbox(monkeypatch):
    provider = ScriptedContentBlockProvider()
    provider.responses = [
        ProviderResponse(
            content=[
                ToolUseBlock(
                    id="toolu_message_leader",
                    name="send_message",
                    input={"to": "Leader", "content": "Bob completed the task."},
                )
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            content=[TextBlock(text="Message delivered to Lead.")],
            stop_reason="end_turn",
        ),
    ]
    teams.set_team_provider(provider)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 0.02)

    assert teams.spawn_teammate_thread("Bob", "developer", "Report to Leader").startswith(
        "Teammate"
    )
    deadline = time.monotonic() + 2
    while "Bob" in teams.active_teammates and time.monotonic() < deadline:
        time.sleep(0.01)

    messages = teams.consume_lead_inbox()
    direct = next(message for message in messages if message["type"] == "message")
    assert direct["from"] == "Bob"
    assert direct["to"] == "lead"
    assert direct["content"] == "Bob completed the task."
    send_definition = next(
        tool for tool in provider.requests[0]["tools"] if tool["name"] == "send_message"
    )
    assert "set to='lead'" in send_definition["description"]


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
