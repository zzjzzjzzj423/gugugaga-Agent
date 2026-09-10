from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gugugaga.observability import sanitize
from gugugaga.stateio import atomic_write_text


PHASES = [
    {"id": "preparing", "label": "准备数据"},
    {"id": "replay", "label": "回放对话"},
    {"id": "consolidating", "label": "整合记忆"},
    {"id": "indexing", "label": "建立索引"},
    {"id": "retrieving", "label": "检索记忆"},
    {"id": "answering", "label": "生成回答"},
    {"id": "scoring", "label": "评分汇总"},
]
ACTIVE = {"queued", "running", "stopping"}
RESUMABLE = {"stopped", "interrupted", "failed", "partial_complete"}
RUN_ID = re.compile(r"eval-[a-f0-9]{24}\Z")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_json(path: Path, default: Any = None) -> Any:
    for attempt in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            # Windows may briefly deny opening the target during atomic replacement.
            if attempt < 4:
                time.sleep(.01)
        except (OSError, ValueError):
            break
    return copy.deepcopy(default)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class _OwnerLease:
    """One execution owner per workspace, including across Web processes."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> bool:
        if self.handle is not None:
            return True
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        handle, self.handle = self.handle, None
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class EvaluationManager:
    """Durable, isolated experiment queue. Reads never start model calls."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        repo_root: Path | str | None = None,
        settings_getter: Callable[[], dict] | None = None,
        publish: Callable[[dict], Any] | None = None,
        provider_factory: Callable[[dict], Any] | None = None,
        executor: Callable[..., dict] | None = None,
    ):
        self.workspace = Path(workspace).resolve()
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
        self.root = (self.workspace / ".gugugaga" / "evaluations").resolve()
        if not self.root.is_relative_to(self.workspace):
            raise ValueError("测评目录必须位于当前工作区内")
        self.root.mkdir(parents=True, exist_ok=True)
        self._settings = settings_getter or (lambda: {})
        self._publish = publish or (lambda event: None)
        self._provider_factory = provider_factory
        self._executor = executor
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._lease = _OwnerLease(self.root / ".owner.lock")
        self._owned = self._lease.acquire()
        self._closed = False
        self._worker: threading.Thread | None = None
        self._queue: list[str] = []
        self._cancel: dict[str, threading.Event] = {}
        self._factories: dict[str, Callable] = {}
        self._secrets: set[str] = set()
        if self._owned:
            self._recover()

    def _directory(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise KeyError("测评记录不存在")
        target = (self.root / run_id).resolve()
        if target.parent != self.root:
            raise ValueError("测评路径越界")
        return target

    def _safe(self, value: Any) -> Any:
        value = sanitize(value)
        if isinstance(value, dict):
            return {key: self._safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._safe(item) for item in value]
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        return value

    def _read_record(self, run_id: str) -> dict:
        with self._lock:
            value = read_json(self._directory(run_id) / "run.json")
        if not isinstance(value, dict):
            raise KeyError("测评记录不存在或文件损坏")
        return value

    def _persist(self, record: dict) -> None:
        record["updated_at"] = now()
        write_json(self._directory(record["id"]) / "run.json", self._safe(record))

    def _recover(self) -> None:
        for folder in self.root.iterdir():
            if not RUN_ID.fullmatch(folder.name) or folder.resolve().parent != self.root:
                continue
            record = read_json(folder / "run.json")
            if isinstance(record, dict) and record.get("status") in ACTIVE:
                record.update(status="interrupted", resumable=True, error="服务已重启；已保存进度，可检查后继续")
                self._persist(record)

    def _ensure_owner(self) -> None:
        if self._closed:
            raise RuntimeError("测评服务正在关闭")
        if not self._owned:
            self._owned = self._lease.acquire()
            if not self._owned:
                raise RuntimeError("另一个控制台进程正在管理此工作区的测评，请在该窗口操作")
            self._recover()

    def _history(self):
        from .history import LegacyCatalog
        return LegacyCatalog(self.repo_root)

    def _defaults(self) -> dict:
        effective = self._settings()
        return {
            "answer_model": effective.get("model", ""),
            "consolidation_model": effective.get("consolidation_model") or effective.get("model", ""),
            "intent_model": effective.get("intent_gate_model") or effective.get("model", ""),
            "embedding_model": effective.get("embedding_model", ""),
        }

    def _snapshots(self) -> list[dict]:
        values = list(self._history().snapshots())
        # Only completed, isolated experiments are offered as reusable sources.
        for folder in self.root.iterdir():
            if not RUN_ID.fullmatch(folder.name) or folder.resolve().parent != self.root:
                continue
            record = read_json(folder / "run.json", {})
            if record.get("status") != "completed":
                continue
            # Each arm/repeat remains a separate snapshot, never mixing databases.
            groups: dict[str, dict[str, str]] = {}
            state = read_json(folder / "checkpoint.json", {})
            arm_configs = {arm["id"]: arm for arm in record.get("config", {}).get("arms", [])}
            for key, entry in state.get("databases", {}).items():
                parts = key.split(":", 2)
                if len(parts) != 3 or not entry.get("ready"):
                    continue
                arm_id, repeat, sample_id = parts
                resolved = (folder / entry.get("path", "")).resolve()
                if not resolved.is_relative_to(folder.resolve()) or not resolved.is_file():
                    continue
                groups.setdefault(f"{arm_id}:{repeat}", {})[sample_id] = str(resolved)
            for group, databases in groups.items():
                arm_id = group.split(":", 1)[0]
                values.append({"id": f"{folder.name}:{group}", "name": f"{record.get('name', folder.name)} · {group}",
                               "databases": databases, "sample_ids": list(databases), "config": arm_configs.get(arm_id, {}),
                               "construction_metadata_complete": True})
        return values

    def catalog(self) -> dict:
        from .config import catalog
        from .data import dataset_catalog
        result = catalog(self._defaults())
        result.update(
            dataset=dataset_catalog(self.repo_root),
            snapshots=[{key: value for key, value in item.items() if key != "databases"} for item in self._snapshots()],
            phases=PHASES,
            execution_available=self._provider_factory is not None or bool(self._settings().get("siliconflow_api_key")),
            isolation="每个实验及方案使用独立数据库；用户记忆和历史源产物保持只读",
        )
        return self._safe(result)

    def _prepare(self, payload: dict) -> tuple[dict, dict, list[dict], list[str]]:
        from .config import ARM_DEFAULTS, BUILD_FIELDS, normalize_config
        from .data import load_dataset
        config = normalize_config(payload, self._defaults())
        dataset = load_dataset(self.repo_root, config["dataset"])
        warnings = []
        snapshots = {item["id"]: item for item in self._snapshots()}
        samples = {str(item["sample_id"]) for item in dataset["questions"]}
        for arm in config["arms"]:
            if arm.get("memory_source") != "snapshot" or arm["mode"] != "memory":
                continue
            selected = snapshots.get(arm.get("snapshot_id"))
            if not selected:
                raise ValueError(f"方案 {arm['name']} 的记忆快照不存在")
            if not samples.issubset(selected.get("databases", {})):
                raise ValueError(f"方案 {arm['name']} 的快照未覆盖所选会话")
            source = selected.get("config") or {}
            aliases = {"threshold": "consolidation_exchange_threshold", "fact_min_importance": "fact_min_importance", "episode_min_importance": "episode_min_importance"}
            for field in BUILD_FIELDS:
                known = source.get(field, source.get(aliases.get(field, field)))
                if field == "consolidation_model" and not known:
                    known = source.get("answer_model")
                requested = arm[field] or (arm["answer_model"] if field == "consolidation_model" else arm[field])
                if known is not None and requested != known:
                    raise ValueError(f"方案 {arm['name']} 的 {field} 与快照构建配置不同；请使用来源值或选择重新构建")
                if known is None and field != "consolidation_model" and arm[field] != ARM_DEFAULTS[field]:
                    raise ValueError(f"快照未记录 {field}，无法验证构建改动；请选择重新构建")
            if not selected.get("construction_metadata_complete"):
                warnings.append("历史快照构建信息不完整；已核对可用字段，未知项不会被视为已验证。构建策略实验应重新构建。")
        base = config["arms"][0]
        differences = []
        for arm in config["arms"][1:]:
            changed = [{"field": key, "baseline": base.get(key), "value": value}
                       for key, value in arm.items() if key not in {"id", "name"} and base.get(key) != value]
            differences.append({"arm_id": arm["id"], "baseline_id": base["id"], "changes": changed,
                                "kind": "single_factor" if len(changed) == 1 else "multiple_factors" if changed else "identical"})
            if any(item["field"] == "final_limit" for item in changed):
                warnings.append("Top K 变化会同时缩放现有类型配额；实际注入仍受预算限制。")
            if any(item["field"] in {"use_facts", "use_episodes", "use_chat"} for item in changed):
                warnings.append("检索内容开关在候选检索前生效，BM25 的语料统计也可能变化。")
        if config.get("type") == "stability":
            warnings.append("稳定性重复独立生成答案；完整流程与固定上下文重复分别统计。")
        return config, dataset, differences, list(dict.fromkeys(warnings))

    def preview(self, payload: dict) -> dict:
        config, dataset, differences, warnings = self._prepare(payload)
        return {"config": config, "differences": differences, "warnings": warnings,
                "question_count": len(dataset["questions"]), "dataset_fingerprint": dataset["fingerprint"]}

    def _protocol(self) -> dict:
        files = list(Path(__file__).parent.glob("*.py"))
        files += [self.repo_root / "eval/locomo_refined" / name for name in ("evaluate.py", "adapter.py", "run_smoke.py", "diagnose_run.py")]
        files += list((self.repo_root / "gugugaga/memory").glob("*.py"))
        return {"version": 1, "scoring": "locomo-token-f1-bleu1-v1", "files": {
            str(path.relative_to(self.repo_root)) if path.is_relative_to(self.repo_root) else path.name:
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files if path.is_file()
        }}

    def _capture_factory(self, run_dir: Path) -> Callable:
        if self._provider_factory is not None:
            return self._provider_factory
        from gugugaga.config import Settings
        from gugugaga.provider import SiliconFlowProvider
        effective = dict(self._settings())
        key = effective.get("siliconflow_api_key", "")
        if not key:
            raise ValueError("请先在控制台配置模型 API Key，再启动测评")
        self._secrets.add(key)
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

        def factory(arm: dict):
            # Construct settings directly: from_env would mutate global workspace state.
            settings = Settings(
                workspace=run_dir, state_dir=run_dir, tasks_dir=run_dir / "tasks",
                memory_dir=run_dir / "memory", mailboxes_dir=run_dir / "mailboxes",
                transcripts_dir=run_dir / "transcripts", outputs_dir=run_dir / "outputs",
                skills_dir=run_dir / "skills", api_key=key, model=arm["answer_model"], base_url=base_url,
            )
            return SiliconFlowProvider(settings, enable_thinking=not arm.get("thinking_disabled", True), temperature=arm.get("temperature", 0))
        return factory

    def _freeze_snapshots(self, folder: Path, config: dict, dataset: dict) -> dict:
        from .runner import _backup
        requested = {arm["snapshot_id"] for arm in config["arms"]
                     if arm["mode"] == "memory" and arm["memory_source"] == "snapshot"}
        sources = {item["id"]: item for item in self._snapshots()} if requested else {}
        samples = {str(qa["sample_id"]) for qa in dataset["questions"]}
        frozen = {"paths": {}, "hashes": {}}
        for snapshot_id in sorted(requested):
            source = sources.get(snapshot_id)
            if source is None:
                raise ValueError("来源快照已不可用，请重新预检")
            frozen["paths"][snapshot_id] = {}
            for sample in sorted(samples):
                path = Path(source["databases"][sample]).resolve()
                relative = Path("sources") / fingerprint(snapshot_id)[:24] / (fingerprint(sample)[:24] + ".db")
                target = (folder / relative).resolve()
                if not target.is_relative_to(folder):
                    raise ValueError("来源备份路径越界")
                _backup(path, target)
                frozen["paths"][snapshot_id][sample] = relative.as_posix()
                frozen["hashes"][relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
        write_json(folder / "snapshots.json", frozen)
        return frozen

    def _frozen_sources(self, run_id: str) -> dict:
        folder = self._directory(run_id)
        record = self._read_record(run_id)
        frozen = read_json(folder / "snapshots.json", {"paths": {}, "hashes": {}})
        if record.get("snapshot_fingerprint") != fingerprint(frozen):
            raise ValueError("冻结的快照清单已变化，请创建新实验")
        mapping = {}
        for snapshot_id, samples in frozen["paths"].items():
            mapping[snapshot_id] = {}
            for sample, relative in samples.items():
                path = (folder / relative).resolve()
                if not path.is_relative_to(folder) or not path.is_file():
                    raise ValueError("冻结的来源数据库缺失或路径越界")
                if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["hashes"].get(relative):
                    raise ValueError("冻结的来源数据库已变化，请创建新实验")
                mapping[snapshot_id][sample] = path
        return mapping

    def create(self, payload: dict) -> dict:
        with self._condition:
            self._ensure_owner()
            config, dataset, differences, warnings = self._prepare(payload)
            run_id = "eval-" + uuid.uuid4().hex[:24]
            folder = self._directory(run_id)
            factory = self._capture_factory(folder)
            folder.mkdir()
            write_json(folder / "config.json", config)
            write_json(folder / "dataset.json", dataset)
            write_json(folder / "protocol.json", self._protocol())
            frozen_sources = self._freeze_snapshots(folder, config, dataset)
            record = {
                "id": run_id, "name": config["name"], "type": config["type"], "status": "queued",
                "config": config, "config_fingerprint": fingerprint(config),
                "dataset_fingerprint": fingerprint(dataset), "created_at": now(), "updated_at": now(),
                "snapshot_fingerprint": fingerprint(frozen_sources),
                "progress": {"phase": "preparing", "completed": 0, "total": len(dataset["questions"]) * len(config["arms"]) * config["repeats"], "message": "已排队，等待执行"},
                "differences": differences, "warnings": warnings, "resumable": False, "error": None,
            }
            self._persist(record)
            self._enqueue(run_id, factory)
            return self.get(run_id)

    def _enqueue(self, run_id: str, factory: Callable) -> None:
        self._factories[run_id] = factory
        self._cancel[run_id] = threading.Event()
        self._queue.append(run_id)
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="memory-evaluation", daemon=True)
            self._worker.start()
        self._condition.notify_all()

    def _event(self, run_id: str, payload: dict) -> None:
        with self._lock:
            record = self._read_record(run_id)
            event_id = int(record.get("event_id", 0)) + 1
            event = self._safe({**payload, "event_id": event_id, "timestamp": now()})
            record["event_id"] = event_id
            if event.get("phase") in {phase["id"] for phase in PHASES}:
                record["phases_seen"] = list(dict.fromkeys([*record.get("phases_seen", []), event["phase"]]))
            previous = dict(record.get("progress", {}))
            if "phase" in event:
                # Counts and question ids belong to one event's phase/scope only.
                # A previous indexing count must never appear as answering progress.
                for key in ("stage_completed", "stage_total", "stage", "qa_id", "coverage", "cached_context"):
                    previous.pop(key, None)
                if any(previous.get(key) != event.get(key) for key in ("arm_id", "sample_id", "repeat")):
                    for key in ("arm_id", "arm_name", "sample_id", "repeat"):
                        previous.pop(key, None)
            record["progress"] = {**previous, **event}
            self._persist(record)
            with (self._directory(run_id) / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._publish({"type": "evaluation", "run_id": run_id, "status": record["status"], "progress": event})

    def _work(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    run_id = self._queue.pop(0)
                    record = self._read_record(run_id)
                    if record["status"] != "queued":
                        self._factories.pop(run_id, None)
                        continue
                    record.update(status="running", started_at=now(), error=None, resumable=False)
                    self._persist(record)
                    attempt_cancel = self._cancel[run_id]
                    factory = self._factories[run_id]
                final_status, final_error = "failed", None
                try:
                    from .runner import EvaluationStopped, execute_run
                    source_map = self._frozen_sources(run_id)
                    result = (self._executor or execute_run)(
                        self._directory(run_id), record["config"], repo_root=self.repo_root,
                        provider_factory=factory, progress=lambda event: self._event(run_id, event),
                        should_stop=lambda: attempt_cancel.is_set() or self._closed, snapshots=source_map,
                    )
                    with self._lock:
                        write_json(self._directory(run_id) / "result.json", self._safe(result))
                        if attempt_cancel.is_set() or self._closed:
                            status = "interrupted" if self._closed else "stopped"
                        else:
                            status = result.get("status", "completed")
                            if status == "complete":
                                status = "completed"
                            if status not in {"completed", "partial_complete", "failed"}:
                                status = "completed"
                        final_status = status
                except Exception as error:
                    from .runner import EvaluationStopped
                    with self._lock:
                        stopped = isinstance(error, EvaluationStopped) or attempt_cancel.is_set()
                        final_status = "interrupted" if self._closed else "stopped" if stopped else "failed"
                        final_error = None if stopped else self._safe(str(error))[:2000]
                finally:
                    with self._lock:
                        current = self._read_record(run_id)
                        self._factories.pop(run_id, None)
                        current.update(status=final_status, error=final_error, resumable=final_status in RESUMABLE, finished_at=now())
                        self._persist(current)
                        self._event(run_id, {"message": {"completed": "测评完成", "partial_complete": "部分题目失败，可重试", "stopped": "已停止，进度已保存", "interrupted": "服务中断，进度已保存", "failed": "测评失败，可查看错误并重试"}.get(current["status"], current["status"]), "run_status": current["status"]})
        finally:
            if self._closed:
                self._lease.release()
                self._owned = False

    def list_runs(self) -> list[dict]:
        values = []
        for folder in self.root.iterdir():
            if RUN_ID.fullmatch(folder.name) and folder.resolve().parent == self.root:
                try:
                    values.append(self.get(folder.name))
                except (KeyError, ValueError):
                    continue
        values.extend(self._history().list_runs())
        return sorted(values, key=lambda item: str(item.get("created_at") or item.get("id", "")), reverse=True)

    def _result(self, run_id: str) -> dict:
        if run_id.startswith("legacy:"):
            return self._history().get_result(run_id)
        self._read_record(run_id)
        value = read_json(self._directory(run_id) / "result.json", {})
        return value if isinstance(value, dict) else {}

    def get(self, run_id: str) -> dict:
        if run_id.startswith("legacy:"):
            return self._history().get_run(run_id)
        record = self._read_record(run_id)
        result = self._result(run_id)
        record["result"] = {key: value for key, value in result.items() if key != "questions"}
        record["resumable"] = record["status"] in RESUMABLE
        return self._safe(record)

    def questions(self, run_id: str, *, offset: int = 0, limit: int = 50, filter: str = "all") -> dict:
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("题目分页范围无效")
        if filter not in {"all", "improved", "regressed", "unchanged", "missing", "failed"}:
            raise ValueError("未知题目筛选条件")
        rows = self._result(run_id).get("questions", [])
        record = self.get(run_id)
        result = record.get("result", {})
        expected_arms = [arm["id"] for arm in result.get("arms", [])]
        if not expected_arms:
            expected_arms = [arm["id"] for arm in record.get("config", {}).get("arms", [])]
        repeat_count = int(record.get("config", {}).get("repeats", 1))
        items = []
        for row in rows:
            answers = row.get("answers") or []
            by_key = {(answer.get("arm_id"), answer.get("repeat", 1)): answer for answer in answers}
            scores = [by_key.get((arm, 1), {}).get("f1") for arm in expected_arms]
            missing = any((arm, repeat) not in by_key or by_key[(arm, repeat)].get("f1") is None
                          for arm in expected_arms for repeat in range(1, repeat_count + 1))
            failed = any(answer.get("status") == "failed" for answer in answers)
            delta = None if len(scores) < 2 or scores[0] is None or scores[1] is None else float(scores[1]) - float(scores[0])
            match = (filter == "all" or filter == "missing" and missing or filter == "failed" and failed
                     or filter == "improved" and delta is not None and delta > 1e-9
                     or filter == "regressed" and delta is not None and delta < -1e-9
                     or filter == "unchanged" and delta is not None and abs(delta) <= 1e-9)
            if match:
                item = {key: value for key, value in row.items() if key not in {"answers", "gold_evidence"}}
                item["answers"] = [{key: value for key, value in answer.items() if key not in {"trace", "memory"}} for answer in answers]
                item["delta_f1"] = delta
                items.append(item)
        return self._safe({"items": items[offset:offset + limit], "total": len(items)})

    def question(self, run_id: str, qa_id: str) -> dict:
        result = (self._history().get_result(run_id, qa_id=qa_id, include_traces=True)
                  if run_id.startswith("legacy:") else self._result(run_id))
        for row in result.get("questions", []):
            if str(row.get("qa_id")) == qa_id:
                return self._safe(row)
        raise KeyError("题目结果尚未生成或不存在")

    def events(self, run_id: str, after: int = 0) -> dict:
        if run_id.startswith("legacy:"):
            self._history().get_run(run_id)
            return {"items": [], "recorded": False}
        self._read_record(run_id)
        events = []
        path = self._directory(run_id) / "events.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if int(event.get("event_id", 0)) > after:
                        events.append(event)
                        if len(events) >= 200:
                            break
        return self._safe({"items": events, "recorded": True})

    def stop(self, run_id: str) -> dict:
        with self._condition:
            self._ensure_owner()
            record = self._read_record(run_id)
            if record["status"] not in ACTIVE:
                return self.get(run_id)
            self._cancel.setdefault(run_id, threading.Event()).set()
            if record["status"] == "queued":
                self._queue = [item for item in self._queue if item != run_id]
                self._factories.pop(run_id, None)
            record.update(status="stopped" if record["status"] == "queued" else "stopping", resumable=True)
            self._persist(record)
            self._event(run_id, {"message": "等待当前调用结束，后续调用将停止" if record["status"] == "stopping" else "已停止排队"})
            return self.get(run_id)

    def resume(self, run_id: str) -> dict:
        with self._condition:
            self._ensure_owner()
            record = self._read_record(run_id)
            if record["status"] not in RESUMABLE:
                raise RuntimeError("只有已停止、中断、失败或部分完成的实验可以继续")
            folder = self._directory(run_id)
            config = read_json(folder / "config.json")
            dataset = read_json(folder / "dataset.json")
            if fingerprint(config) != record.get("config_fingerprint") or fingerprint(dataset) != record.get("dataset_fingerprint"):
                raise ValueError("冻结配置或题目已变化，请复制配置创建新实验")
            if read_json(folder / "protocol.json") != self._protocol():
                raise ValueError("执行代码或评分协议已变化，请创建新实验以保持可比性")
            self._frozen_sources(run_id)
            factory = self._capture_factory(folder)
            record.update(status="queued", error=None, resumable=False)
            self._persist(record)
            self._enqueue(run_id, factory)
            return self.get(run_id)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            for event in self._cancel.values():
                event.set()
            for run_id in self._queue:
                record = self._read_record(run_id)
                if record["status"] == "queued":
                    record.update(status="interrupted", resumable=True)
                    self._persist(record)
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2)
        if self._worker is None or not self._worker.is_alive():
            self._lease.release()
            self._owned = False
