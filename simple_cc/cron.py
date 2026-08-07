from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import config


@dataclass
class CronJob:
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool


scheduled_jobs: dict[str, CronJob] = {}
cron_queue: list[CronJob] = []
cron_lock = threading.Lock()
_last_fired: dict[str, str] = {}
_cron_lifecycle_lock = threading.RLock()
_cron_stop_event = threading.Event()
_cron_thread: threading.Thread | None = None


def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(
            _cron_field_matches(part.strip(), value) for part in field.split(",")
        )
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7
    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)
    if not (m and h and month_ok):
        return False
    if dom == "*" and dow == "*":
        return True
    if dom == "*":
        return dow_ok
    if dow == "*":
        return dom_ok
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) <= 0:
            return f"Invalid step: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None
    if "-" in field:
        left, right = field.split("-", 1)
        if not left.isdigit() or not right.isdigit():
            return f"Invalid range: {field}"
        a, b = int(left), int(right)
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < lo or value > hi:
        return f"Value {value} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for field, (lo, hi), name in zip(fields, bounds, names):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    durable = [asdict(job) for job in scheduled_jobs.values() if job.durable]
    config.DURABLE_PATH.write_text(json.dumps(durable, indent=2))


def load_durable_jobs():
    loaded: dict[str, CronJob] = {}
    with cron_lock:
        if config.DURABLE_PATH.exists():
            try:
                for item in json.loads(config.DURABLE_PATH.read_text()):
                    job = CronJob(**item)
                    if not validate_cron(job.cron):
                        loaded[job.id] = job
            except Exception:
                pass
        scheduled_jobs.clear()
        scheduled_jobs.update(loaded)
        cron_queue.clear()
        _last_fired.clear()


def schedule_job(
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = True,
) -> CronJob | str:
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron,
        prompt=prompt,
        recurring=recurring,
        durable=durable,
    )
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()
    return job


def cancel_job(job_id: str) -> str:
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()
    return f"Cancelled {job_id}"


def cron_scheduler_loop(stop_event: threading.Event | None = None):
    while True:
        if stop_event is None:
            time.sleep(1)
        elif stop_event.wait(1):
            return
        now = datetime.now()
        marker = now.strftime("%Y-%m-%d %H:%M")
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if (
                        cron_matches(job.cron, now)
                        and _last_fired.get(job.id) != marker
                    ):
                        cron_queue.append(job)
                        _last_fired[job.id] = marker
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as error:
                    print(f"  \033[31m[cron error] {job.id}: {error}\033[0m")


def consume_cron_queue() -> list[CronJob]:
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


def run_schedule_cron(
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = True,
) -> str:
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' -> {prompt}"


def run_list_crons() -> str:
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs."
    return "\n".join(
        f"  {job.id}: '{job.cron}' -> {job.prompt[:40]} "
        f"[{'recurring' if job.recurring else 'one-shot'}, "
        f"{'durable' if job.durable else 'session'}]"
        for job in jobs
    )


def run_cancel_cron(job_id: str) -> str:
    return cancel_job(job_id)


def initialize_cron() -> threading.Thread:
    """Load and start cron only after the runtime selects its workspace."""
    global _cron_stop_event, _cron_thread

    shutdown_cron()
    load_durable_jobs()
    with _cron_lifecycle_lock:
        _cron_stop_event = threading.Event()
        _cron_thread = threading.Thread(
            target=cron_scheduler_loop,
            args=(_cron_stop_event,),
            name="simple-cc-cron-scheduler",
            daemon=True,
        )
        thread = _cron_thread
        thread.start()
        return thread


def shutdown_cron(timeout: float = 2.0) -> bool:
    global _cron_thread

    with _cron_lifecycle_lock:
        thread = _cron_thread
        _cron_stop_event.set()
    if (
        thread is not None
        and thread is not threading.current_thread()
        and thread.is_alive()
    ):
        thread.join(timeout=timeout)
    with _cron_lifecycle_lock:
        stopped = thread is None or not thread.is_alive()
        if _cron_thread is thread and stopped:
            _cron_thread = None
    return stopped


class CronScheduler:
    """Compatibility adapter for the pre-migration runtime."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, CronJob] = {}
        self._queue: list[str] = []
        self._last_fired: dict[str, str] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        jobs = {}
        for raw in json.loads(self.path.read_text(encoding="utf-8")):
            if "expression" in raw:
                raw["cron"] = raw.pop("expression")
                raw.pop("last_fired", None)
            raw.setdefault("durable", True)
            job = CronJob(**raw)
            jobs[job.id] = job
        self._jobs = jobs

    def _save(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                [asdict(job) for job in self._jobs.values()], indent=2
            ),
            encoding="utf-8",
        )
        os.replace(temp, self.path)

    def schedule(
        self, expression: str, prompt: str, recurring: bool = True
    ) -> CronJob:
        error = validate_cron(expression)
        if error:
            raise ValueError(error)
        with self._lock:
            job = CronJob(
                f"cron_{uuid.uuid4().hex[:8]}",
                expression,
                prompt,
                recurring,
                True,
            )
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
                if (
                    self._last_fired.get(job.id) != marker
                    and cron_matches(job.cron, now)
                ):
                    self._queue.append(job.prompt)
                    self._last_fired[job.id] = marker
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

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._queue)

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
