from __future__ import annotations

import threading
import time
from collections import deque

import pytest

from simple_cc import config, tasks
import simple_cc.teams as teams
from simple_cc.permissions import PermissionPolicy
from simple_cc.provider import ProviderResponse, TextBlock, ToolUseBlock


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
    if hasattr(teams, "pending_requests"):
        teams.pending_requests.clear()
    if hasattr(teams, "MessageBus"):
        monkeypatch.setattr(teams, "BUS", teams.MessageBus())
    if hasattr(teams, "set_team_provider"):
        teams.set_team_provider(None)
    yield
    if hasattr(teams, "active_teammates"):
        teams.active_teammates.clear()
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
    assert reply == [
        {
            "from": "lead",
            "to": "alice",
            "content": "Approved",
            "type": "plan_approval_response",
            "ts": reply[0]["ts"],
            "metadata": {"request_id": request_id, "approve": True},
        }
    ]
    assert teams.pending_requests[request_id].status == "approved"


def test_shutdown_request_is_acknowledged_and_routed_to_its_request(monkeypatch):
    require_source_team_api()
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1)

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


def test_two_agents_atomically_claim_only_one_unblocked_task():
    require_source_team_api()
    dependency = tasks.create_task("schema")
    assert tasks.claim_task(dependency.id, "lead").startswith("Claimed")
    assert tasks.complete_task(dependency.id).startswith("Completed")
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
    from simple_cc.__main__ import build_runtime
    from simple_cc.config import Settings

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
    from simple_cc.__main__ import build_runtime
    from simple_cc.config import Settings
    from simple_cc.tools import TOOL_HANDLERS

    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 1)
    settings = Settings.from_env(config.WORKDIR)
    provider = ScriptedContentBlockProvider()
    app = build_runtime(settings, provider=provider)
    try:
        assert app.runtime.provider is provider
        assert TOOL_HANDLERS["spawn_teammate"](
            name="runtime-alice",
            role="developer",
            prompt="Create the file",
        ) == "Teammate 'runtime-alice' spawned as developer"

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


def test_team_tool_definitions_and_handlers_are_bijective_and_prompted():
    require_source_team_api()
    from simple_cc.prompts import assemble_system_prompt
    from simple_cc.tools import TOOL_DEFINITIONS, TOOL_HANDLERS

    team_tools = {
        "spawn_teammate",
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
