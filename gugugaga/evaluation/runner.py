"""Restartable local evaluation runner. Only run-directory database copies are writable."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
import time
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eval.locomo_refined.adapter import conversation_exchanges, message_content
from eval.locomo_refined.diagnose_run import evidence_turn_id, render_oracle_evidence, retrieved_source_turn_ids
from eval.locomo_refined.evaluate import _best, bleu1, summarize, token_f1
from eval.locomo_refined.run_smoke import ANSWER_SYSTEM, response_text, token_cost
from gugugaga.memory import MemoryService
from gugugaga.memory.service import _CONSOLIDATION_SYSTEM
from gugugaga.memory.validation import parse_consolidation_result
from .config import normalize_config
from .retrieval import retrieve


class EvaluationStopped(RuntimeError):
    """The current model call may finish; no subsequent work is started."""


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.01)


def _inside(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("测评写入路径超出独立运行目录")
    return path


def _database_hash(path: Path) -> str:
    # SQL content is stable across WAL checkpoints and does not open the source for writing.
    digest = hashlib.sha256()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        for line in connection.iterdump():
            digest.update(line.encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _backup(source: Path, destination: Path) -> str:
    source = source.resolve()
    if not source.is_file():
        raise ValueError("所选来源记忆数据库不存在")
    if source == destination.resolve():
        raise ValueError("来源数据库与测评目标不能相同")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".backup.tmp")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as origin:
        with closing(sqlite3.connect(temporary)) as target:
            origin.backup(target)
    fingerprint = _database_hash(temporary)
    temporary.replace(destination)
    return fingerprint


class _EvaluationMemoryService(MemoryService):
    """Keep admission/commit semantics, with independent Fact/Episode generation switches."""
    def __init__(self, *args: Any, arm: dict, check_stop: Callable[[], None], **kwargs: Any):
        self.arm = arm
        self.check_stop = check_stop
        super().__init__(*args, **kwargs)

    def _consolidate(self, batch: Any) -> Any:
        self.check_stop()
        constraints = ""
        if not self.arm["generate_facts"]:
            constraints += "\nFor this evaluation configuration facts MUST be []."
        if not self.arm["generate_episodes"]:
            constraints += "\nFor this evaluation configuration episodes MUST be []."
        response = self.provider.create(
            model=self.model, system=_CONSOLIDATION_SYSTEM + constraints,
            messages=[{"role": "user", "content": self._batch_prompt(batch)}], tools=[], max_tokens=2400,
        )
        parsed = parse_consolidation_result(response_text(response), max_facts=10, min_importance=self.min_importance,
                                            max_episodes=5, episode_min_importance=self.episode_min_importance)
        return replace(parsed, facts=parsed.facts if self.arm["generate_facts"] else (),
                       episodes=parsed.episodes if self.arm["generate_episodes"] else ())


def _configure_indices(service: MemoryService, arm: dict) -> dict:
    """Exclude disabled layers before BM25 corpus statistics and vector candidate limits."""
    database = service.repository.path
    disabled = [kind for kind, field in (("fact", "use_facts"), ("episode", "use_episodes"), ("chat", "use_chat")) if not arm[field]]
    with closing(sqlite3.connect(database)) as connection, connection:
        for kind in disabled:
            connection.execute("DELETE FROM memory_fts WHERE kind=?", (kind,))
            connection.execute("DELETE FROM memory_embeddings WHERE memory_kind=?", (kind,))
            connection.execute("DELETE FROM memory_index_outbox WHERE memory_kind=?", (kind,))
        rows = connection.execute("SELECT turn_id, MAX(COALESCE(completed_at,created_at)) AS latest, MAX(id) AS last_id FROM chat_log WHERE is_final=1 GROUP BY turn_id ORDER BY latest DESC,last_id DESC").fetchall()
        active_turns = [row[0] for row in (rows[:arm["evidence_window"]] if arm["evidence_window"] else rows)]
        allowed_keys = set()
        if arm["use_chat"]:
            allowed_turns = set(active_turns)
            allowed_keys = {f"chat:{row[0]}" for row in connection.execute("SELECT id,turn_id FROM chat_log WHERE is_final=1") if row[1] in allowed_turns}
        for table in ("memory_embeddings", "memory_index_outbox"):
            keys = [row[0] for row in connection.execute(f"SELECT memory_key FROM {table} WHERE memory_kind='chat'")]
            connection.executemany(f"DELETE FROM {table} WHERE memory_key=?", [(key,) for key in keys if key not in allowed_keys])
        counts = dict(connection.execute("SELECT kind,COUNT(*) FROM memory_fts GROUP BY kind").fetchall())
    return {"fts_counts": counts, "total_exchanges": len(rows), "vector_chat_exchanges": len(active_turns) if arm["use_chat"] else 0,
            "vector_chat_messages": len(allowed_keys), "evidence_window": arm["evidence_window"]}


def _recover_private_work(database: Path) -> None:
    # The manager runs one worker and calls this only after interruption, never while another owns this DB.
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE consolidation_batches SET status='failed',lease_expires_at=NULL,error_code='evaluation_resumed' WHERE status='processing'")
        connection.execute("UPDATE chat_log SET consolidation_status='pending',batch_id=NULL,lease_expires_at=NULL,next_retry_at=NULL WHERE consolidation_status IN ('processing','failed')")
        connection.execute("UPDATE memory_index_outbox SET status='pending',lease_expires_at=NULL,next_retry_at=NULL WHERE status IN ('processing','failed')")


def _source_database(snapshots: dict, snapshot_id: str, sample_id: str) -> Path:
    source = snapshots.get(snapshot_id)
    if isinstance(source, dict):
        source = source.get("databases", source)
        source = source.get(sample_id) if isinstance(source, dict) else source
    if not source:
        raise ValueError(f"快照中没有会话 {sample_id} 的数据库")
    path = Path(source).resolve()
    if path.is_dir():
        matches = [p for p in (path / f"{sample_id}.db", path / sample_id / "memory.db", path / "memory.db") if p.is_file()]
        if len(matches) != 1:
            raise ValueError("无法唯一确定快照数据库")
        path = matches[0]
    if not path.is_file():
        raise ValueError("来源快照数据库不存在")
    return path


def _evidence(qa: dict, memory: str, trace: dict) -> dict:
    messages = qa.get("evidence_messages") or []
    if not messages:
        return {"source_hit": None, "source_recall": None, "full_text_hit": None, "full_text_recall": None}
    gold = {evidence_turn_id(str(qa["sample_id"]), item) for item in messages}
    found = retrieved_source_turn_ids(memory)
    texts = [message_content(item).strip() for item in messages]
    full_count = sum(bool(value) and value in memory for value in texts)
    return {"source_hit": bool(gold & found), "source_recall": len(gold & found) / len(gold),
            "full_text_hit": bool(full_count), "full_text_recall": full_count / len(texts),
            "gold_turn_ids": sorted(gold), "matched_turn_ids": sorted(gold & found),
            "rule": "来源标记交集；完整原文为实际注入字符串包含标准消息全文，不判断语义正确性"}


def _metrics(predictions: list[dict]) -> dict:
    if not predictions:
        return {"overall_f1": None, "overall_bleu1": None, "question_count": 0, "categories": {}}
    return summarize(predictions)


def _result(config: dict, dataset: dict, state: dict) -> dict:
    questions = []
    for qa in dataset["questions"]:
        questions.append({"qa_id": qa["qa_id"], "sample_id": qa["sample_id"], "category": str(qa.get("category", "")),
                          "question": qa["question"], "gold_answer": qa["answer"], "gold_evidence": qa.get("evidence_messages") or [],
                          "answers": [v for v in state["answers"].values() if v["qa_id"] == qa["qa_id"]]})
    arms, stability = [], []
    for arm in config["arms"]:
        predictions, repeat_rows, costs, source_metrics = [], [], [], []
        complete_repeats = []
        failed_count = 0
        for repeat in range(1, config["repeats"] + 1):
            current = []
            failed = 0
            for qa in questions:
                answer = next((v for v in qa["answers"] if v["arm_id"] == arm["id"] and v["repeat"] == repeat), None)
                if not answer:
                    continue
                if answer["status"] != "completed":
                    failed += 1
                    continue
                prediction = {"category": qa["category"], "gold_answer": qa["gold_answer"], "predicted_answer": answer["answer"], "sample_id": qa["sample_id"]}
                current.append(prediction)
                predictions.append(prediction)
                costs.append(answer.get("token_cost"))
                source_metrics.append(answer.get("evidence", {}))
            metrics = _metrics(current)
            complete = len(current) == len(questions)
            if complete:
                complete_repeats.append(metrics["overall_f1"])
            failed_count += failed
            repeat_rows.append({"repeat": repeat, "metrics": metrics, "completed": len(current), "planned": len(questions), "failed": failed,
                                "status": "completed" if complete else "partial_complete" if current else "pending"})
        metrics = _metrics(predictions)
        metrics["samples"] = {sample: _metrics([p for p in predictions if p["sample_id"] == sample]) for sample in sorted({p["sample_id"] for p in predictions})}
        for metric in ("source_hit", "source_recall", "full_text_hit", "full_text_recall"):
            known = [m[metric] for m in source_metrics if m.get(metric) is not None]
            metrics[metric] = sum(known) / len(known) * 100 if known else None
        metrics["evidence_question_count"] = sum(m.get("source_hit") is not None for m in source_metrics)
        planned = len(questions) * config["repeats"]
        arms.append({"id": arm["id"], "name": arm["name"], "metrics": metrics, "completed": len(predictions), "planned": planned,
                     "failed": failed_count, "token_cost": sum(costs) if costs and all(c is not None for c in costs) else None,
                     "known_token_cost": sum(c for c in costs if c is not None), "cost_scope": "answer_only", "repeats": repeat_rows})
        if config["type"] == "stability":
            stability.append({"arm_id": arm["id"], "scope": config["repeat_scope"], "completed_repeats": len(complete_repeats),
                              "planned_repeats": config["repeats"], "failed_answers": failed_count,
                              "mean": statistics.mean(complete_repeats) if complete_repeats else None,
                              "stddev": statistics.stdev(complete_repeats) if len(complete_repeats) >= 2 else None,
                              "min": min(complete_repeats) if complete_repeats else None, "max": max(complete_repeats) if complete_repeats else None,
                              "question_count": len(questions), "rule": "仅统计全部题目完成的独立重复"})
    comparison = []
    baseline = config["arms"][0]["id"]
    for arm in config["arms"][1:]:
        deltas = []
        for question in questions:
            for repeat in range(1, config["repeats"] + 1):
                left = next((a for a in question["answers"] if a["arm_id"] == baseline and a["repeat"] == repeat and a["status"] == "completed"), None)
                right = next((a for a in question["answers"] if a["arm_id"] == arm["id"] and a["repeat"] == repeat and a["status"] == "completed"), None)
                if left and right:
                    deltas.append({"qa_id": question["qa_id"], "repeat": repeat, "delta_f1": right["f1"] - left["f1"]})
        comparison.append({"baseline_id": baseline, "arm_id": arm["id"], "paired_count": len(deltas),
                           "delta_f1": statistics.mean([v["delta_f1"] for v in deltas]) if deltas else None, "questions": deltas})
    total = sum(arm["planned"] for arm in arms)
    completed = sum(arm["completed"] for arm in arms)
    failed = sum(arm["failed"] for arm in arms)
    return {"arms": arms, "questions": questions, "stability": stability, "comparison": comparison,
            "warnings": ["Token F1/BLEU-1 是文本重合指标；成本仅覆盖回答，其他阶段未记录。", "预算按 1 token ≈ 4 字符执行。"],
            "completed": completed, "planned": total, "failed": failed,
            "status": "completed" if completed == total else "partial_complete", "dataset_fingerprint": dataset["fingerprint"]}


def execute_run(run_dir: Path | str, config: dict, *, repo_root: Path | str,
                provider_factory: Callable[[dict], Any], progress: Callable[[dict], None],
                should_stop: Callable[[], bool], snapshots: dict | None = None) -> dict:
    from .data import load_dataset
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = normalize_config(config)
    dataset_path = _inside(root, "dataset.json")
    if dataset_path.exists():
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        dataset = load_dataset(Path(repo_root), config["dataset"])
        _write(dataset_path, dataset)
    fingerprint = _hash({"config": config, "dataset": dataset, "protocol": "memory-evaluation-v1"})
    checkpoint_path = _inside(root, "checkpoint.json")
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ValueError("配置或冻结数据已变化，不能复用已有检查点")
    else:
        state = {"fingerprint": fingerprint, "answers": {}, "contexts": {}, "databases": {}, "snapshots": {},
                 "retrieval_clock": datetime.now(timezone.utc).isoformat()}
    total = len(dataset["questions"]) * len(config["arms"]) * config["repeats"]
    context: dict = {}

    def persist() -> None:
        _write(checkpoint_path, state)
        _write(_inside(root, "result.json"), _result(config, dataset, state))

    def check_stop() -> None:
        if should_stop():
            persist()
            raise EvaluationStopped("已停止；完成的上下文、答案和数据库保留在检查点")

    def emit(phase: str, message: str, **extra: Any) -> None:
        progress({"phase": phase, **context, "completed": len(state["answers"]), "total": total,
                  "message": message, **extra})

    persist()
    emit("preparing", "验证冻结配置、题目与独立运行目录")
    check_stop()
    conversations = {str(c["sample_id"]): c for c in dataset["conversations"]}
    for arm in config["arms"]:
        context = {"arm_id": arm["id"], "arm_name": arm["name"]}
        check_stop()
        provider = provider_factory(dict(arm))
        for repeat in range(1, config["repeats"] + 1):
            for sample_id, conversation in conversations.items():
                selected = [qa for qa in dataset["questions"] if str(qa["sample_id"]) == sample_id]
                if not selected:
                    continue
                context = {"arm_id": arm["id"], "arm_name": arm["name"], "repeat": repeat, "sample_id": sample_id}
                # Paths use digests rather than trusting externally sourced sample/qa identifiers.
                build_repeat = 1 if config["repeat_scope"] == "answers" else repeat
                build_key = f"{arm['id']}:{build_repeat}:{sample_id}"
                database = _inside(root, "databases", arm["id"], str(build_repeat), _hash(sample_id)[:24] + ".db")
                service = None
                try:
                    incomplete = [qa for qa in selected if state["answers"].get(f"{arm['id']}:{repeat}:{qa['qa_id']}", {}).get("status") != "completed"]
                    if not incomplete:
                        continue
                    need_memory = arm["mode"] == "memory" and any(f"{arm['id']}:{build_repeat}:{qa['qa_id']}" not in state["contexts"] for qa in incomplete)
                    if need_memory:
                        check_stop()
                        emit("preparing", "准备测评专用记忆数据库", stage_completed=0, stage_total=1)
                        if arm["memory_source"] == "snapshot":
                            source = _source_database(snapshots or {}, arm["snapshot_id"], sample_id)
                            source_key = arm["snapshot_id"] + ":" + sample_id
                            previous = state["snapshots"].get(source_key)
                            if previous and _database_hash(source) != previous:
                                raise ValueError("来源快照内容已改变，不能从检查点继续")
                            if not database.exists():
                                state["snapshots"][source_key] = _backup(source, database)
                                persist()
                        service = _EvaluationMemoryService(
                            database, provider, arm=arm, check_stop=check_stop,
                            consolidation_enabled=arm["generate_facts"] or arm["generate_episodes"], threshold=arm["threshold"],
                            model=arm["consolidation_model"] or None, min_importance=arm["fact_min_importance"],
                            episode_min_importance=arm["episode_min_importance"], evidence_hot_exchanges=10000,
                            recall_token_budget=min(8000, arm["token_budget"]), intent_gate_enabled=arm["gate_enabled"],
                            intent_gate_model=arm["intent_model"] or None,
                            embedding_model=arm["embedding_model"] if arm["retrieval_mode"] != "bm25" else None,
                            retrieval_candidate_limit=arm["candidate_limit"], retrieval_final_limit=arm["final_limit"],
                            retrieval_min_score=arm["min_score"], start_worker=False,
                        )
                        _recover_private_work(database)
                        if not state["databases"].get(build_key, {}).get("ready") and arm["memory_source"] == "rebuild":
                            exchanges = list(conversation_exchanges(conversation))
                            with closing(sqlite3.connect(database)) as connection:
                                recorded = {row[0] for row in connection.execute("SELECT DISTINCT turn_id FROM chat_log")}
                            expected = {e.turn_id for e in exchanges}
                            if recorded - expected:
                                raise ValueError("测评数据库包含其他会话的记录")

                            def drain(tail: bool = False) -> None:
                                if not service.consolidation_enabled:
                                    return
                                while True:
                                    pending = int(service.status().get("pending", 0))
                                    if not pending or pending < arm["threshold"] and not tail:
                                        return
                                    check_stop()
                                    service.threshold = min(pending, arm["threshold"])
                                    consolidated = int(service.status().get("consolidated", 0))
                                    emit("consolidating", f"整合 {service.threshold} 个 Exchange", stage_completed=consolidated, stage_total=len(exchanges))
                                    if not service.process_pending(max_batches=1):
                                        check_stop()
                                        raise RuntimeError("摘要整合失败；可从检查点重试")
                                    persist()
                                    emit("consolidating", "已保存本批整合结果", stage_completed=int(service.status().get("consolidated", 0)), stage_total=len(exchanges))
                                    check_stop()

                            for index, exchange in enumerate(exchanges, 1):
                                check_stop()
                                emit("replay", "回放历史对话到测评库", stage_completed=index - 1, stage_total=len(exchanges))
                                if exchange.turn_id not in recorded:
                                    service.repository.record_exchange(session_id=exchange.session_id, turn_id=exchange.turn_id,
                                        user_content=exchange.user_content, assistant_content=exchange.assistant_content,
                                        source="locomo_refined_evaluation", completed_at=exchange.completed_at)
                                drain()
                            drain(tail=True)
                            emit("replay", "历史对话回放完成", stage_completed=len(exchanges), stage_total=len(exchanges))
                        coverage = _configure_indices(service, arm)
                        if arm["retrieval_mode"] != "bm25":
                            with closing(sqlite3.connect(database)) as connection:
                                index_total = connection.execute("SELECT COUNT(*) FROM memory_index_outbox WHERE status<>'completed'").fetchone()[0]
                            processed = 0
                            while True:
                                check_stop()
                                emit("indexing", "构建实验向量索引", stage_completed=processed, stage_total=index_total, coverage=coverage)
                                count = service.process_index_pending(max_jobs=32)
                                if not count:
                                    with closing(sqlite3.connect(database)) as connection:
                                        remaining = connection.execute("SELECT COUNT(*) FROM memory_index_outbox WHERE status<>'completed'").fetchone()[0]
                                    if remaining:
                                        raise RuntimeError("向量索引未完成；请检查 Embedding 模型后重试")
                                    break
                                processed += count
                            emit("indexing", "向量索引完成", stage_completed=index_total, stage_total=index_total, coverage=coverage)
                        else:
                            emit("indexing", "BM25 索引已准备完成", stage_completed=1, stage_total=1, coverage=coverage)
                        state["databases"][build_key] = {"path": str(database.relative_to(root)), "ready": True, "coverage": coverage}
                        persist()
                    for qa in selected:
                        context["qa_id"] = qa["qa_id"]
                        answer_key = f"{arm['id']}:{repeat}:{qa['qa_id']}"
                        context_key = f"{arm['id']}:{build_repeat}:{qa['qa_id']}"
                        if state["answers"].get(answer_key, {}).get("status") == "completed":
                            continue
                        check_stop()
                        started = time.monotonic()
                        try:
                            cached_context = state["contexts"].get(context_key)
                            if cached_context is None:
                                emit("retrieving", "检索本题证据", stage="gate")
                                if arm["mode"] == "oracle":
                                    memory = render_oracle_evidence(qa, conversation, include_image_context=True)
                                    trace = {"schema_version": 2, "mode": "oracle", "stages": {}, "final": {"content": memory}}
                                elif arm["mode"] == "no_memory":
                                    memory, trace = "", {"schema_version": 2, "mode": "no_memory", "stages": {}, "final": {"content": ""}}
                                else:
                                    memory, trace = retrieve(service, str(qa["question"]), arm, check_stop=check_stop,
                                        stage=lambda name: emit("retrieving", f"检索阶段：{name}", stage=name),
                                        now=datetime.fromisoformat(state["retrieval_clock"]))
                                cached_context = {"memory": memory, "trace": trace}
                                state["contexts"][context_key] = cached_context
                                persist()
                            else:
                                emit("retrieving", "复用已冻结的实际注入上下文", cached_context=True)
                            check_stop()
                            memory, trace = cached_context["memory"], cached_context["trace"]
                            emit("answering", "模型正在生成答案")
                            response = provider.create(messages=[{"role": "user", "content": f"Memory:\n{memory.strip() or '(none)'}\n\nQuestion:\n{qa['question']}"}],
                                                       system=ANSWER_SYSTEM, tools=[], max_tokens=arm["max_tokens"], model=arm["answer_model"])
                            text = response_text(response)
                            if not text:
                                raise RuntimeError("回答模型返回空内容")
                            emit("scoring", "计算文本匹配分数与证据指标")
                            state["answers"][answer_key] = {"qa_id": qa["qa_id"], "arm_id": arm["id"], "repeat": repeat,
                                "answer": text, "f1": _best(token_f1, text, qa["answer"]) * 100,
                                "bleu1": _best(bleu1, text, qa["answer"]) * 100,
                                "memory": memory, "trace": trace, "status": "completed", "error": None,
                                "token_cost": token_cost(response), "cost_scope": "answer_only", "cached_answer": False,
                                "context_reused": repeat > 1 and config["repeat_scope"] == "answers",
                                "evidence": _evidence(qa, memory, trace), "elapsed_seconds": round(time.monotonic() - started, 3)}
                        except EvaluationStopped:
                            raise
                        except Exception as error:
                            # Do not persist provider exception bodies: they may include credentials or request headers.
                            state["answers"][answer_key] = {"qa_id": qa["qa_id"], "arm_id": arm["id"], "repeat": repeat,
                                "answer": None, "f1": None, "bleu1": None, "status": "failed", "token_cost": None,
                                "memory": state["contexts"].get(context_key, {}).get("memory"),
                                "trace": state["contexts"].get(context_key, {}).get("trace"),
                                "error": f"{type(error).__name__}：本题执行失败，可从检查点重试"}
                        persist()
                        emit("scoring", "已保存本题结果")
                        check_stop()
                finally:
                    if service is not None:
                        service.close()
    result = _result(config, dataset, state)
    _write(_inside(root, "result.json"), result)
    emit("completed", "测评完成" if result["status"] == "completed" else "部分题目失败，可重试", completed=result["completed"], total=result["planned"])
    return result
