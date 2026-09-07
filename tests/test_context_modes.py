from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gugugaga.context_modes import (
    CompressionReason,
    ContextModeError,
    HERMES_HEADINGS,
    ModelAwareTokenEstimator,
    PI_HEADINGS,
    RequestContext,
    SessionContextConfig,
    SessionContextCoordinator,
    TokenCounterRegistry,
    create_child_context_coordinator,
    ensure_message_ids,
    legal_cut,
    validate_tool_protocol,
)


class LengthCounter:
    id = "length-test"
    version = "v1"

    @staticmethod
    def _content(message):
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content)
        return sum(len(str(block.get("content", block.get("text", "")))) for block in content)

    def count_messages(self, messages):
        return sum(self._content(message) + 1 for message in messages)

    def count_request(self, system, tools, messages):
        return len(system) + len(str(tools)) + self.count_messages(messages)


def structured(headings):
    return "\n".join(f"{heading}\nkept" for heading in headings)


def summary_callback(system, prompt, max_tokens):
    assert max_tokens == 2_000
    if "Pi Markdown" in system:
        return structured(PI_HEADINGS)
    if "Hermes Markdown" in system:
        return structured(HERMES_HEADINGS)
    return "Goal, constraints, completed work, and next step."


def test_child_agent_inherits_context_policy_with_isolated_state(tmp_path):
    parent = SessionContextCoordinator(
        SessionContextConfig.parse("pi"),
        summary_callback=summary_callback,
        workspace=tmp_path,
        transcripts_dir=tmp_path / ".gugugaga" / "transcripts",
        memory_dir=tmp_path / ".gugugaga" / "memory",
        tool_results_dir=tmp_path / ".gugugaga" / "tool-results",
    )

    child = create_child_context_coordinator(
        parent,
        agent_type="subagent",
        agent_id="child-1",
    )

    assert child is not parent
    assert child.config == parent.config
    assert child.registry is parent.registry
    assert child.state is not parent.state
    assert child.transcripts_dir == parent.transcripts_dir / "subagents"
    assert child.tool_results_dir == (
        parent.tool_results_dir / "subagents" / "child-1"
    )
    child.set_mode("pi")
    assert parent.status()["lifecycle"] == "configuring"
    child.close()
    assert child.status()["lifecycle"] == "closed"


def coordinator(tmp_path, mode="cc", **overrides):
    values = {
        "context_window_tokens": 1_000,
        "token_counter_id": "length-test",
        "pi_reserve_tokens": 200,
        "pi_keep_recent_tokens": 200,
    }
    values.update(overrides)
    return SessionContextCoordinator(
        SessionContextConfig.parse(mode, **values),
        counter_registry=TokenCounterRegistry([LengthCounter()]),
        summary_callback=summary_callback,
        workspace=tmp_path,
        transcripts_dir=tmp_path / ".gugugaga" / "transcripts",
    )


def request():
    return RequestContext(system="system", tools=[])


def test_mode_defaults_validation_and_locking(tmp_path):
    default = SessionContextConfig.parse()
    assert default.mode.value == "cc"
    assert default.source == "default"
    with pytest.raises(ContextModeError) as invalid:
        SessionContextConfig.parse("CC")
    assert invalid.value.code == "INVALID_CONTEXT_MODE"

    session = coordinator(tmp_path, "hermes")
    messages = [{"role": "user", "content": "start"}]
    session.observe_history(messages)
    session.set_mode("hermes")
    with pytest.raises(ContextModeError) as locked:
        session.set_mode("pi")
    assert locked.value.code == "CONTEXT_MODE_LOCKED"
    assert session.status()["mode"] == "hermes"


def test_structured_summary_prompt_names_every_required_heading(tmp_path):
    systems = []

    def recording_summary(system, prompt, max_tokens):
        systems.append(system)
        return structured(HERMES_HEADINGS)

    session = coordinator(tmp_path, "hermes")
    session.summary_callback = recording_summary
    session._summary("Return structured Markdown.", "history", HERMES_HEADINGS)

    assert len(systems) == 1
    assert all(heading in systems[0] for heading in HERMES_HEADINGS)


def test_unknown_counter_fails_before_session_use(tmp_path):
    config = SessionContextConfig.parse(
        "pi", context_window_tokens=1_000, token_counter_id="missing",
        pi_reserve_tokens=100, pi_keep_recent_tokens=100,
    )
    with pytest.raises(ContextModeError) as unavailable:
        SessionContextCoordinator(config, workspace=tmp_path)
    assert unavailable.value.code == "TOKEN_ACCOUNTING_UNAVAILABLE"


def test_model_aware_counter_selects_family_profile_and_counts_tokens():
    qwen = ModelAwareTokenEstimator("Qwen/Qwen3-32B")
    llama = ModelAwareTokenEstimator("meta-llama/Llama-3.3-70B")
    fallback = ModelAwareTokenEstimator("vendor/unknown-model")

    assert qwen.profile_id == "qwen"
    assert llama.profile_id == "llama"
    assert fallback.profile_id == "fallback"
    assert qwen.count_messages([{"role": "user", "content": "hello world"}]) > 0
    assert llama.count_messages([{"role": "user", "content": "你好世界" * 20}]) > qwen.count_messages(
        [{"role": "user", "content": "你好世界" * 20}]
    )


def test_default_coordinator_reports_model_aware_counter(tmp_path, monkeypatch):
    monkeypatch.setattr("gugugaga.context_modes.config.MODEL", "deepseek-ai/DeepSeek-V3")
    session = SessionContextCoordinator(
        SessionContextConfig.parse(),
        workspace=tmp_path,
        transcripts_dir=tmp_path / ".gugugaga" / "transcripts",
    )

    status = session.status()
    assert status["token_counter_id"] == "gugugaga_model_estimator"
    assert status["token_counter_model"] == "deepseek-ai/DeepSeek-V3"
    assert status["token_counter_profile"] == "deepseek"


def test_tool_protocol_and_cut_keep_parallel_group_together():
    messages = [
        {"role": "user", "content": "run both"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "a", "name": "read_file", "input": {"path": "a"}},
                {"type": "tool_use", "id": "b", "name": "read_file", "input": {"path": "b"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "A"},
                {"type": "tool_result", "tool_use_id": "b", "content": "B"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    validate_tool_protocol(messages)
    assert not legal_cut(messages, 2)
    assert legal_cut(messages, 1)
    broken = copy.deepcopy(messages)
    broken[2]["content"].pop()
    with pytest.raises(ContextModeError) as invalid:
        validate_tool_protocol(broken)
    assert invalid.value.code == "INVALID_TOOL_PROTOCOL"


def test_cc_manual_compaction_keeps_raw_ledger_and_projects_summary(tmp_path):
    session = coordinator(tmp_path)
    messages = [
        {"role": "user", "content": "old goal " + "x" * 500},
        {"role": "assistant", "content": [{"type": "text", "text": "old answer " + "y" * 500}]},
        {"role": "user", "content": "continue " + "z" * 500},
    ]
    original = copy.deepcopy(messages)
    result = session.manual_compact(messages, request())

    assert result.status == "success"
    assert [{k: v for k, v in m.items() if k != "message_id"} for m in messages] == original
    projected = session.project(messages)
    assert len(projected) == 1
    assert projected[0]["content"].startswith("[Compacted]")
    assert "message_id" not in projected[0]


def test_cc_automatic_summary_uses_registered_token_counter(tmp_path):
    session = coordinator(tmp_path, context_window_tokens=1_000)
    messages = [
        {"role": "user", "content": "a" * 450},
        {"role": "assistant", "content": "b" * 450},
    ]

    projected = session.prepare_request(messages, request())

    assert len(projected) == 1
    assert projected[0]["content"].startswith("[Compacted]")
    assert session.status()["successful_compactions"] == 1


def test_hermes_first_then_later_compaction_shapes(tmp_path):
    session = coordinator(tmp_path, "hermes")
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 110}
        for index in range(10)
    ]
    first_projection = session.prepare_request(messages, request())
    assert first_projection[:3] == [
        {k: v for k, v in message.items() if k != "message_id"}
        for message in messages[:3]
    ]
    assert sum("[Hermes summary]" in str(m["content"]) for m in first_projection) == 1
    assert session.status()["successful_compactions"] == 1

    messages.extend(
        {"role": "user" if index % 2 == 0 else "assistant", "content": "y" * 130}
        for index in range(8)
    )
    second_projection = session.prepare_request(messages, request())
    assert "[Hermes summary]" in second_projection[0]["content"]
    assert sum("[Hermes summary]" in str(m["content"]) for m in second_projection) == 1
    assert session.status()["successful_compactions"] == 2


def test_pi_appends_entry_preserves_raw_and_uses_next_projection(tmp_path):
    session = coordinator(tmp_path, "pi")
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "z" * 110}
        for index in range(10)
    ]
    original_contents = [message["content"] for message in messages]
    projected = session.prepare_request(messages, request())

    assert [message["content"] for message in messages] == original_contents
    assert len(session.state.pi_entries) == 1
    entry = session.state.pi_entries[0]
    assert entry.sequence == 1
    assert entry.tokens_after_estimate < entry.tokens_before
    assert projected[0]["content"].startswith("[Pi summary]")
    assert session.project(messages) == projected


def test_pi_split_turn_generates_turn_prefix(tmp_path):
    session = coordinator(
        tmp_path,
        "pi",
        context_window_tokens=2_500,
        pi_reserve_tokens=100,
        pi_keep_recent_tokens=300,
    )
    messages = [
        {"role": "user", "content": "request " + "r" * 500},
        {"role": "assistant", "content": "step " + "a" * 500},
        {"role": "assistant", "content": "step " + "b" * 500},
        {"role": "assistant", "content": "recent " + "c" * 300},
    ]
    session.prepare_request(
        messages, request(), reason=CompressionReason.MANUAL, force=True
    )
    entry = session.state.pi_entries[0]
    assert entry.is_split_turn
    assert entry.turn_prefix_summary is not None


def test_pi_force_compacts_when_history_is_below_keep_recent_budget(tmp_path):
    session = coordinator(
        tmp_path,
        "pi",
        context_window_tokens=5_000,
        pi_reserve_tokens=500,
        pi_keep_recent_tokens=4_000,
    )
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 200}
        for index in range(8)
    ]
    assert session.counter.count_messages(messages) < 4_000

    result = session.manual_compact(messages, request())

    assert result.status == "success"
    assert session.state.pi_entries


def test_summary_failure_does_not_partially_commit(tmp_path):
    session = coordinator(tmp_path, "hermes")
    session.summary_callback = lambda *args: "missing required headings"
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 110}
        for index in range(10)
    ]
    with pytest.raises(ContextModeError) as failure:
        session.prepare_request(messages, request())
    assert failure.value.code == "SUMMARY_FAILED"
    assert session.state.successful_compactions == 0
    assert session.state.projection is None
    assert session.state.pi_entries == []
    assert session.state.last_result.code == "SUMMARY_FAILED"


def test_status_and_events_do_not_expose_message_or_summary_text(tmp_path):
    secret = "sk-test-secret-value"
    session = coordinator(tmp_path, "hermes")
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": secret + "x" * 110}
        for index in range(10)
    ]
    session.prepare_request(messages, request())

    diagnostics = json.dumps(
        {
            "status": session.status(),
            "events": [event.__dict__ for event in session.state.events],
        },
        default=str,
    )
    assert secret not in diagnostics
    assert "## Active Task" not in diagnostics


def test_transcripts_are_isolated_by_workspace(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = coordinator(first_root)
    second = coordinator(second_root)
    first_messages = [
        {"role": "user", "content": "a" * 1_000},
        {"role": "assistant", "content": "b" * 1_000},
    ]
    second_messages = [
        {"role": "user", "content": "c" * 1_000},
        {"role": "assistant", "content": "d" * 1_000},
    ]

    assert first.manual_compact(first_messages, request()).status == "success"
    assert second.manual_compact(second_messages, request()).status == "success"
    assert list((first_root / ".gugugaga" / "transcripts").glob("*.jsonl"))
    assert list((second_root / ".gugugaga" / "transcripts").glob("*.jsonl"))
    assert not list(first_root.rglob("*second*"))


def test_large_tool_outputs_are_isolated_by_session_workspace(tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    sessions = [coordinator(root) for root in roots]
    for index, session in enumerate(sessions):
        messages = [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"call-{index}", "name": "read_file", "input": {"path": "large.txt"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"call-{index}", "content": "x" * 210_000}
                ],
            },
        ]
        session.prepare_request(messages, request())

    first_output = roots[0] / ".task_outputs" / "tool-results" / "call-0.txt"
    second_output = roots[1] / ".task_outputs" / "tool-results" / "call-1.txt"
    assert first_output.is_file()
    assert second_output.is_file()
    assert not (roots[0] / ".task_outputs" / "tool-results" / "call-1.txt").exists()
    assert not (roots[1] / ".task_outputs" / "tool-results" / "call-0.txt").exists()


def test_pi_accumulates_only_successful_workspace_file_events(tmp_path):
    session = coordinator(
        tmp_path,
        "pi",
        context_window_tokens=5_000,
        pi_reserve_tokens=500,
        pi_keep_recent_tokens=500,
    )
    messages = [
        {"role": "user", "content": "old request " + "x" * 1_000},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "read", "name": "read_file", "input": {"path": "README.md"}},
                {"type": "tool_use", "id": "write", "name": "write_file", "input": {"path": "src/new.py"}},
                {"type": "tool_use", "id": "escape", "name": "edit_file", "input": {"path": "../escape.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "read", "content": "contents"},
                {"type": "tool_result", "tool_use_id": "write", "content": "Wrote file"},
                {"type": "tool_result", "tool_use_id": "escape", "content": "Permission denied"},
            ],
        },
        {"role": "assistant", "content": "completed " + "y" * 1_000},
        {"role": "user", "content": "recent " + "z" * 600},
    ]
    session.prepare_request(messages, request(), force=True)
    entry = session.state.pi_entries[0]

    assert entry.files_read == ("README.md", "src/new.py")
    assert entry.files_modified == ("src/new.py",)


def test_pi_second_entry_links_previous_summary(tmp_path):
    prompts = []

    def recording_summary(system, prompt, max_tokens):
        prompts.append(prompt)
        return structured(PI_HEADINGS)

    session = coordinator(
        tmp_path,
        "pi",
        context_window_tokens=3_000,
        pi_reserve_tokens=300,
        pi_keep_recent_tokens=300,
    )
    session.summary_callback = recording_summary
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "a" * 500}
        for index in range(6)
    ]
    session.prepare_request(messages, request(), force=True)
    first = session.state.pi_entries[0]
    messages.extend(
        {"role": "user" if index % 2 == 0 else "assistant", "content": "b" * 500}
        for index in range(6)
    )
    session.prepare_request(messages, request(), force=True)
    second = session.state.pi_entries[1]

    assert second.sequence == 2
    assert second.previous_summary_id == first.id
    assert any(first.summary in prompt for prompt in prompts[1:])


def test_large_summary_input_is_bounded_and_hierarchically_merged(tmp_path):
    requests = []

    def bounded_summary(system, prompt, max_tokens):
        requests.append((system, prompt, max_tokens))
        assert len(prompt.encode("utf-8")) <= 65_536
        return structured(HERMES_HEADINGS)

    session = coordinator(
        tmp_path,
        "hermes",
        context_window_tokens=250_000,
        pi_reserve_tokens=10_000,
        pi_keep_recent_tokens=10_000,
    )
    session.summary_callback = bounded_summary
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 20_000}
        for index in range(10)
    ]
    session.prepare_request(messages, request(), force=True)

    assert len(requests) >= 3
    assert "final merged summary" in requests[-1][0]
    assert session.status()["successful_compactions"] == 1


def test_hermes_512k_boundary_and_trigger_equality(tmp_path):
    exact_boundary = coordinator(
        tmp_path / "exact",
        "hermes",
        context_window_tokens=512_000,
        pi_reserve_tokens=10_000,
        pi_keep_recent_tokens=10_000,
    )
    below_boundary = coordinator(
        tmp_path / "below",
        "hermes",
        context_window_tokens=511_999,
        pi_reserve_tokens=10_000,
        pi_keep_recent_tokens=10_000,
    )
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 200}
        for index in range(10)
    ]
    exact_boundary.prepare_request(copy.deepcopy(messages), request(), force=True)
    below_boundary.prepare_request(copy.deepcopy(messages), request(), force=True)
    assert exact_boundary.state.events[-1].details["effective_threshold_ratio"] == 0.50
    assert below_boundary.state.events[-1].details["effective_threshold_ratio"] == 0.75

    at_trigger = coordinator(tmp_path / "at", "hermes")
    at_messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 73}
        for index in range(9)
    ] + [{"role": "assistant", "content": "x" * 75}]
    assert at_trigger.counter.count_request("system", [], at_messages) == 750
    at_trigger.prepare_request(at_messages, request())
    assert at_trigger.status()["successful_compactions"] == 1

    below_trigger = coordinator(tmp_path / "under", "hermes")
    under_messages = copy.deepcopy(at_messages)
    under_messages[-1]["content"] = "x" * 74
    assert below_trigger.counter.count_request("system", [], under_messages) == 749
    projected = below_trigger.prepare_request(under_messages, request())
    assert below_trigger.status()["successful_compactions"] == 0
    assert len(projected) == len(under_messages)


def test_concurrent_sessions_keep_mode_state_and_entries_isolated(tmp_path):
    sessions = {
        "cc": coordinator(tmp_path / "cc", "cc"),
        "hermes": coordinator(tmp_path / "hermes", "hermes"),
        "pi": coordinator(
            tmp_path / "pi",
            "pi",
            context_window_tokens=1_500,
            pi_reserve_tokens=200,
            pi_keep_recent_tokens=300,
        ),
    }
    histories = {
        mode: [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": mode + "-" + "x" * 500,
            }
            for index in range(10)
        ]
        for mode in sessions
    }

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(
                lambda mode: sessions[mode].manual_compact(
                    histories[mode], request()
                ),
                sessions,
            )
        )

    results_by_mode = dict(zip(sessions, results))
    assert {
        mode: (result.status, result.code)
        for mode, result in results_by_mode.items()
    } == {
        "cc": ("success", "SUCCESS"),
        "hermes": ("success", "SUCCESS"),
        "pi": ("success", "SUCCESS"),
    }
    assert {mode: session.status()["mode"] for mode, session in sessions.items()} == {
        "cc": "cc",
        "hermes": "hermes",
        "pi": "pi",
    }
    assert len(sessions["pi"].state.pi_entries) == 1
    assert sessions["cc"].state.pi_entries == []
    assert sessions["hermes"].state.pi_entries == []
    assert len({session.session_id for session in sessions.values()}) == 3


def test_routing_p95_is_below_50ms_for_1000_messages(tmp_path):
    session = SessionContextCoordinator(
        SessionContextConfig.parse(
            "pi",
            context_window_tokens=2_000_000,
            pi_reserve_tokens=100_000,
            pi_keep_recent_tokens=20_000,
        ),
        workspace=tmp_path,
        transcripts_dir=tmp_path / ".gugugaga" / "transcripts",
    )
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 100}
        for index in range(1_000)
    ]
    samples = []
    for _ in range(20):
        started = time.perf_counter()
        session.prepare_request(messages, request())
        samples.append((time.perf_counter() - started) * 1_000)
    p95 = sorted(samples)[18]
    assert p95 < 50, f"routing p95 was {p95:.2f} ms"
