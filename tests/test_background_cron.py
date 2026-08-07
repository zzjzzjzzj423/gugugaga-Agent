import time
from datetime import datetime

from simple_cc.background import BackgroundManager, CronScheduler, cron_matches, validate_cron
from simple_cc.models import ToolCall


def wait_until(predicate, timeout=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_background_completion_is_one_independent_notification():
    manager = BackgroundManager()
    job_id = manager.start(ToolCall("c1", "bash", {}), lambda: "done")
    wait_until(manager.has_pending)
    notifications = manager.drain()
    assert job_id in notifications[0]
    assert "done" in notifications[0]
    assert manager.drain() == []


def test_cron_validation_and_matching():
    assert validate_cron("*/5 * * * *") is None
    assert validate_cron("70 * * * *") is not None
    assert cron_matches("*/5 * * * *", datetime(2026, 8, 7, 10, 15))
    assert not cron_matches("*/5 * * * *", datetime(2026, 8, 7, 10, 16))


def test_cron_day_of_month_and_week_use_standard_or_semantics():
    expression = "0 10 1 * 5"
    assert cron_matches(expression, datetime(2026, 8, 7, 10, 0))  # Friday
    assert cron_matches(expression, datetime(2026, 8, 1, 10, 0))  # first day


def test_one_shot_cron_removed_after_drain(tmp_path):
    scheduler = CronScheduler(tmp_path / "cron.json")
    job = scheduler.schedule("* * * * *", "run tests", recurring=False)
    scheduler.fire_due(datetime(2026, 8, 7, 10, 15))
    assert scheduler.drain() == ["run tests"]
    assert job.id not in {item.id for item in scheduler.list()}
