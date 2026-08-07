from __future__ import annotations

import importlib
import json
import time
from datetime import datetime
from threading import Event

import pytest

import simple_cc.background as background
from simple_cc import config
from simple_cc.provider import ToolUseBlock


def _cron_module():
    return importlib.import_module("simple_cc.cron")


@pytest.fixture
def isolated_scheduling_state(tmp_path):
    original_workspace = config.WORKDIR
    config.configure_workspace(tmp_path)
    if hasattr(background, "background_lock"):
        with background.background_lock:
            background.background_tasks.clear()
            background.background_results.clear()
            background._bg_counter = 0
    if hasattr(background, "initialize_background_tasks"):
        background.initialize_background_tasks()

    try:
        cron = _cron_module()
    except ModuleNotFoundError:
        cron = None
    if cron is not None:
        with cron.cron_lock:
            cron.scheduled_jobs.clear()
            cron.cron_queue.clear()
            cron._last_fired.clear()

    yield cron

    if hasattr(background, "shutdown_background_tasks"):
        background.shutdown_background_tasks(timeout=2)
    if hasattr(background, "background_lock"):
        with background.background_lock:
            background.background_tasks.clear()
            background.background_results.clear()
            background._bg_counter = 0
    if cron is not None:
        with cron.cron_lock:
            cron.scheduled_jobs.clear()
            cron.cron_queue.clear()
            cron._last_fired.clear()
    config.configure_workspace(original_workspace)


def test_slow_bash_returns_immediately_then_delivers_one_completion(
    isolated_scheduling_state,
):
    entered = Event()
    release = Event()

    def bash_handler(command: str, run_in_background: bool = False) -> str:
        entered.set()
        release.wait(timeout=2)
        return f"finished: {command}"

    block = ToolUseBlock(
        id="toolu_1",
        name="bash",
        input={"command": "python -m pytest -q"},
    )

    assert background.is_slow_operation(block.name, block.input)
    assert background.should_run_background(block.name, block.input)

    placeholder = background.dispatch_background_task(
        block, {"bash": bash_handler}
    )

    assert placeholder == (
        "[Background task bg_0001 started] "
        "Result will arrive as a task_notification."
    )
    background_id = "bg_0001"
    assert entered.wait(timeout=1)
    assert background.background_tasks[background_id]["status"] == "running"
    assert background.collect_background_results() == []

    release.set()
    deadline = time.monotonic() + 2
    while background.background_tasks[background_id]["status"] != "completed":
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert background.collect_background_results() == [
        "<task_notification>\n"
        "  <task_id>bg_0001</task_id>\n"
        "  <status>completed</status>\n"
        "  <command>python -m pytest -q</command>\n"
        "  <summary>finished: python -m pytest -q</summary>\n"
        "</task_notification>"
    ]
    assert background.collect_background_results() == []


def test_background_dispatch_leaves_fast_requests_for_foreground_execution(
    isolated_scheduling_state,
):
    block = ToolUseBlock(
        id="toolu_fast",
        name="bash",
        input={"command": "pwd"},
    )

    assert (
        background.dispatch_background_task(block, {"bash": lambda: ""})
        is None
    )
    assert background.background_tasks == {}


def test_background_shutdown_cancels_cooperative_job_and_rejects_new_work(
    isolated_scheduling_state, monkeypatch
):
    entered = Event()
    marker = config.WORKDIR / "post-close.txt"
    post_hooks: list[str] = []

    def cooperative_handler(
        command: str,
        run_in_background: bool = False,
        cancel_event: Event | None = None,
    ) -> str:
        del command, run_in_background
        entered.set()
        assert cancel_event is not None
        cancel_event.wait(timeout=2)
        if not cancel_event.is_set():
            marker.write_text("late side effect", encoding="utf-8")
        return "cancelled"

    monkeypatch.setattr(
        background,
        "trigger_hooks",
        lambda event, block, output: post_hooks.append(event),
    )
    background.initialize_background_tasks()
    block = ToolUseBlock(
        id="toolu_cancel",
        name="bash",
        input={"command": "python long-running.py"},
    )
    background_id = background.start_background_task(
        block, {"bash": cooperative_handler}
    )
    assert entered.wait(timeout=1)

    outcome = background.shutdown_background_tasks(timeout=1)

    assert outcome.stopped
    assert outcome.live_job_ids == ()
    assert background.background_tasks[background_id]["status"] == "cancelled"
    assert not marker.exists()
    assert post_hooks == []
    assert background.collect_background_results() == []
    with pytest.raises(RuntimeError, match="not accepting new jobs"):
        background.start_background_task(block, {"bash": cooperative_handler})


def test_background_shutdown_reports_uncooperative_live_thread(
    isolated_scheduling_state,
):
    entered = Event()
    release = Event()

    def uncooperative_handler(command: str) -> str:
        del command
        entered.set()
        release.wait(timeout=2)
        return "eventually finished"

    background.initialize_background_tasks()
    block = ToolUseBlock(
        id="toolu_blocked",
        name="bash",
        input={"command": "python blocked.py"},
    )
    background_id = background.start_background_task(
        block, {"bash": uncooperative_handler}
    )
    assert entered.wait(timeout=1)

    timed_out = background.shutdown_background_tasks(timeout=0.01)

    assert not timed_out.stopped
    assert timed_out.live_job_ids == (background_id,)
    assert background.background_tasks[background_id]["thread"].is_alive()

    release.set()
    completed = background.shutdown_background_tasks(timeout=1)
    assert completed.stopped
    assert completed.live_job_ids == ()


def test_literal_five_field_cron_validation_and_standard_matching(
    isolated_scheduling_state,
):
    cron = _cron_module()

    assert cron.validate_cron("15 10 1 8 5") is None
    assert cron.validate_cron("60 10 1 8 5") == (
        "minute: Value 60 out of bounds [0-59]"
    )
    assert cron.validate_cron("15 10 1 8") == "Expected 5 fields, got 4"

    assert cron.cron_matches(
        "15 10 1 8 5", datetime(2026, 8, 7, 10, 15)
    )
    assert cron.cron_matches(
        "15 10 1 8 5", datetime(2026, 8, 1, 10, 15)
    )
    assert not cron.cron_matches(
        "15 10 1 8 5", datetime(2026, 8, 8, 10, 15)
    )


def test_durable_jobs_save_and_reload_below_selected_workspace(
    isolated_scheduling_state,
):
    cron = _cron_module()

    durable_job = cron.schedule_job(
        "17 4 2 3 1", "durable check", recurring=True, durable=True
    )
    session_job = cron.schedule_job(
        "18 4 2 3 1", "session check", recurring=True, durable=False
    )

    assert isinstance(durable_job, cron.CronJob)
    assert isinstance(session_job, cron.CronJob)
    assert config.DURABLE_PATH == config.WORKDIR / ".scheduled_tasks.json"
    assert json.loads(config.DURABLE_PATH.read_text(encoding="utf-8")) == [
        {
            "id": durable_job.id,
            "cron": "17 4 2 3 1",
            "prompt": "durable check",
            "recurring": True,
            "durable": True,
        }
    ]

    with cron.cron_lock:
        cron.scheduled_jobs.clear()
    cron.load_durable_jobs()

    assert cron.scheduled_jobs == {durable_job.id: durable_job}


def test_loading_workspace_replaces_all_workspace_scoped_cron_state(
    isolated_scheduling_state, monkeypatch
):
    cron = _cron_module()
    workspace_a = config.WORKDIR / "workspace-a"
    workspace_b = config.WORKDIR / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    config.configure_workspace(workspace_a)
    config.DURABLE_PATH.write_text(
        json.dumps(
            [
                {
                    "id": "cron_a",
                    "cron": "17 4 2 3 1",
                    "prompt": "workspace A prompt",
                    "recurring": True,
                    "durable": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    cron.load_durable_jobs()
    cron.cron_queue.append(cron.scheduled_jobs["cron_a"])

    config.configure_workspace(workspace_b)
    config.DURABLE_PATH.write_text(
        json.dumps(
            [
                {
                    "id": "cron_b",
                    "cron": "18 4 2 3 1",
                    "prompt": "workspace B prompt",
                    "recurring": True,
                    "durable": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    cron.load_durable_jobs()

    assert list(cron.scheduled_jobs) == ["cron_b"]
    assert cron.consume_cron_queue() == []

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 3, 2, 4, 17)

    sleeps = 0

    def stop_after_one_cycle(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise StopIteration

    monkeypatch.setattr(cron, "datetime", FixedDateTime)
    monkeypatch.setattr(cron.time, "sleep", stop_after_one_cycle)
    with pytest.raises(StopIteration):
        cron.cron_scheduler_loop()

    assert cron.consume_cron_queue() == []
    cron.save_durable_jobs()
    assert json.loads(config.DURABLE_PATH.read_text(encoding="utf-8")) == [
        {
            "id": "cron_b",
            "cron": "18 4 2 3 1",
            "prompt": "workspace B prompt",
            "recurring": True,
            "durable": True,
        }
    ]


def test_scheduler_removes_exhausted_durable_job_and_handlers_are_fixed(
    isolated_scheduling_state, monkeypatch
):
    cron = _cron_module()
    from simple_cc.tools import TOOL_DEFINITIONS, TOOL_HANDLERS

    names = [definition["name"] for definition in TOOL_DEFINITIONS]
    assert {"schedule_cron", "list_crons", "cancel_cron"} <= set(names)
    assert set(names) == set(TOOL_HANDLERS)

    result = TOOL_HANDLERS["schedule_cron"](
        cron="17 4 2 3 1",
        prompt="one shot",
        recurring=False,
        durable=True,
    )
    job = next(iter(cron.scheduled_jobs.values()))
    assert result == f"Scheduled {job.id}: '17 4 2 3 1' -> one shot"

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 3, 2, 4, 17)

    sleeps = 0

    def stop_after_one_cycle(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise StopIteration

    monkeypatch.setattr(cron, "datetime", FixedDateTime)
    monkeypatch.setattr(cron.time, "sleep", stop_after_one_cycle)

    with pytest.raises(StopIteration):
        cron.cron_scheduler_loop()

    assert job.id not in cron.scheduled_jobs
    assert cron.consume_cron_queue() == [job]
    assert cron.consume_cron_queue() == []
    assert json.loads(config.DURABLE_PATH.read_text(encoding="utf-8")) == []
    assert TOOL_HANDLERS["list_crons"]() == "No cron jobs."
    assert TOOL_HANDLERS["cancel_cron"](job_id=job.id) == (
        f"Job {job.id} not found"
    )
