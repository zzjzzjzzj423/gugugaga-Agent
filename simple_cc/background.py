from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .models import ToolCall


class BackgroundManager:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self._notifications: list[str] = []
        self._lock = threading.Lock()

    def start(self, call: ToolCall, function: Callable[[], str]) -> str:
        job_id = f"bg_{uuid.uuid4().hex[:8]}"
        self.jobs[job_id] = {"status": "running", "tool": call.name}

        def run():
            try:
                output = str(function())
                status = "completed"
            except Exception as error:
                output, status = f"Error: {type(error).__name__}: {error}", "failed"
            with self._lock:
                self.jobs[job_id].update(status=status, result=output)
                self._notifications.append(f"<task_notification id='{job_id}' status='{status}'>{output}</task_notification>")

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._notifications)

    def drain(self) -> list[str]:
        with self._lock:
            items, self._notifications = self._notifications, []
            return items


@dataclass
class CronJob:
    id: str
    expression: str
    prompt: str
    recurring: bool = True
    last_fired: str = ""


def _field_matches(field: str, value: int, low: int, high: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    return any(part.isdigit() and int(part) == value for part in field.split(","))


def validate_cron(expression: str) -> str | None:
    fields = expression.split()
    if len(fields) != 5:
        return "cron must contain five fields"
    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    for field, (low, high) in zip(fields, ranges):
        try:
            probes = field[2:] if field.startswith("*/") else field.replace("*", str(low))
            values = [int(part) for part in probes.split(",")]
        except ValueError:
            return f"invalid cron field: {field}"
        if any(value < (1 if field.startswith("*/") else low) or value > high for value in values):
            return f"cron field out of range: {field}"
    return None


def cron_matches(expression: str, value: datetime) -> bool:
    fields = expression.split()
    values = (value.minute, value.hour, value.day, value.month, (value.weekday() + 1) % 7)
    ranges = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))
    return len(fields) == 5 and all(_field_matches(field, number, *bounds) for field, number, bounds in zip(fields, values, ranges))


class CronScheduler:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, CronJob] = {}
        self._queue: list[str] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    def _load(self):
        if self.path.exists():
            self._jobs = {raw["id"]: CronJob(**raw) for raw in json.loads(self.path.read_text(encoding="utf-8"))}

    def _save(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps([asdict(job) for job in self._jobs.values()], indent=2), encoding="utf-8")
        os.replace(temp, self.path)

    def schedule(self, expression: str, prompt: str, recurring: bool = True) -> CronJob:
        error = validate_cron(expression)
        if error:
            raise ValueError(error)
        with self._lock:
            job = CronJob(f"cron_{uuid.uuid4().hex[:8]}", expression, prompt, recurring)
            self._jobs[job.id] = job
            self._save()
            return job

    def list(self) -> list[CronJob]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> str:
        with self._lock:
            if self._jobs.pop(job_id, None) is None:
                return f"Error: unknown cron {job_id}"
            self._save()
            return f"Cancelled {job_id}"

    def fire_due(self, now: datetime | None = None):
        now = now or datetime.now()
        marker = now.strftime("%Y-%m-%dT%H:%M")
        with self._lock:
            remove = []
            for job in self._jobs.values():
                if job.last_fired != marker and cron_matches(job.expression, now):
                    self._queue.append(job.prompt)
                    job.last_fired = marker
                    if not job.recurring:
                        remove.append(job.id)
            for job_id in remove:
                self._jobs.pop(job_id, None)
            if remove or self._jobs:
                self._save()

    def drain(self) -> list[str]:
        with self._lock:
            items, self._queue = self._queue, []
            return items

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        def loop():
            while not self._stop.wait(1):
                self.fire_due()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

