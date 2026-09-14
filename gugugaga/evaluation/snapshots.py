"""Manufacture and publish isolated, reusable memory snapshots.

Snapshot manifests contain no credentials. Sources are read only and database
paths are accepted only within their exact run or published snapshot directory.
"""
from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.locomo_refined.adapter import conversation_exchanges
from .config import ARM_DEFAULTS, normalize_config
from .runner import _backup, _database_hash, _hash


BUILD_OPTIONS = {"generate_facts", "generate_episodes", "threshold", "fact_min_importance",
                 "episode_min_importance", "consolidation_model", "embedding_model", "temperature", "thinking_disabled"}


def normalize_snapshot_config(payload: dict, defaults: dict | None = None) -> dict:
    if not isinstance(payload, dict) or set(payload) - {"name", "dataset", "build"}:
        raise ValueError("快照配置仅支持 name、dataset 和 build")
    build = payload.get("build", {})
    if not isinstance(build, dict) or set(build) - BUILD_OPTIONS:
        raise ValueError("快照 build 包含未知的构建参数")
    values = (defaults or {}).get("default_arm", defaults or {})
    arm = copy.deepcopy(ARM_DEFAULTS)
    arm.update({key: value for key, value in values.items() if key in BUILD_OPTIONS})
    arm.update(copy.deepcopy(build))
    arm.update(id="memory", name="快照记忆", mode="memory", memory_source="rebuild", snapshot_id="",
               use_chat=True, use_facts=arm["generate_facts"], use_episodes=arm["generate_episodes"],
               gate_enabled=False, route_mode="fixed", fixed_route="mixed", intent_model="",
               evidence_window=0, retrieval_mode="hybrid" if arm["embedding_model"] else "bm25",
               answer_model=arm["consolidation_model"] or "snapshot-build-no-answer")
    return normalize_config({"name": payload.get("name", "记忆快照"), "dataset": copy.deepcopy(payload.get("dataset", {})),
                             "type": "basic", "repeats": 1, "repeat_scope": "full", "arms": [arm]})


def snapshot_estimate(dataset: dict, config: dict) -> dict:
    arm = config["arms"][0]
    summaries = arm["generate_facts"] or arm["generate_episodes"]
    samples = []
    for conversation in dataset["conversations"]:
        count = sum(1 for _ in conversation_exchanges(conversation))
        samples.append({"sample_id": str(conversation["sample_id"]), "exchange_count": count,
                        "batch_count": math.ceil(count / arm["threshold"]) if summaries else 0})
    return {"conversation_count": len(samples), "exchange_count": sum(item["exchange_count"] for item in samples),
            "batch_count": sum(item["batch_count"] for item in samples), "samples": samples,
            "sample_ids": [item["sample_id"] for item in samples], "builds_answers": False,
            "embedding_enabled": arm["retrieval_mode"] != "bm25"}


def database_stats(path: Path | str) -> dict:
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        return {"exchanges": connection.execute("SELECT COUNT(DISTINCT turn_id) FROM chat_log WHERE is_final=1").fetchone()[0],
                "messages": connection.execute("SELECT COUNT(*) FROM chat_log WHERE is_final=1").fetchone()[0],
                "facts": connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
                "episodes": connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须是对象")
    return value


def _contained_file(root: Path, path: Path) -> Path:
    """Reject lexical traversal and every symlink/junction beneath the root."""
    root = root.absolute()
    path = path.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("快照数据库路径超出允许目录") from error
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("快照数据库路径包含非法跳转")
    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("快照数据库路径不能经过符号链接或目录联接")
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("快照数据库不存在或超出允许目录")
    return resolved


def _source_metadata(run_dir: Path, arm: dict) -> tuple[dict, bool, dict]:
    if arm.get("memory_source") == "rebuild":
        return copy.deepcopy(arm), set(ARM_DEFAULTS).issubset(arm), {}
    frozen = _read(run_dir / "snapshots.json") if (run_dir / "snapshots.json").is_file() else {}
    metadata = frozen.get("metadata", {}).get(arm.get("snapshot_id"), {})
    original = metadata.get("config") if isinstance(metadata, dict) else None
    return copy.deepcopy(original if isinstance(original, dict) else arm), bool(
        isinstance(original, dict) and metadata.get("construction_metadata_complete")), {
            "source_snapshot_id": arm.get("snapshot_id"), "evaluation_config": copy.deepcopy(arm)}


def export_snapshot(destination: Path | str, *, run_dir: Path | str, record: dict, arm_id: str,
                    repeat: int = 1, name: str, snapshot_id: str) -> dict:
    if not isinstance(snapshot_id, str) or not re.fullmatch(r"snapshot-[a-f0-9]{24}", snapshot_id):
        raise ValueError("快照 id 无效")
    if not isinstance(name, str) or not name.strip() or len(name) > 160:
        raise ValueError("快照名称不能为空且不超过 160 字符")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("快照轮次必须是正整数")
    source_root = Path(run_dir).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("快照已存在，不允许覆盖")
    if destination.resolve().is_relative_to(source_root):
        raise ValueError("发布的快照必须位于来源运行目录之外")
    arm = next((value for value in record.get("config", {}).get("arms", []) if value.get("id") == arm_id), None)
    if not arm or arm.get("mode") != "memory":
        raise ValueError("只能导出记忆方案的数据库")
    config = record["config"]
    if repeat > config.get("repeats", 1):
        raise ValueError("所选轮次超出来源实验范围")
    build_repeat = 1 if config.get("repeat_scope") == "answers" else repeat
    state = _read(source_root / "checkpoint.json")
    dataset = _read(source_root / "dataset.json")
    expected = [str(item["sample_id"]) for item in dataset["conversations"]]
    if len(set(expected)) != len(expected) or not expected:
        raise ValueError("来源冻结数据的会话列表无效")
    paths = {}
    for sample in expected:
        entry = state.get("databases", {}).get(f"{arm_id}:{build_repeat}:{sample}", {})
        if not entry.get("ready"):
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("检查点数据库必须使用运行目录内的相对路径")
        paths[sample] = _contained_file(source_root / "databases", source_root / relative)
    if not paths:
        raise ValueError("所选方案尚无完成的会话记忆，不能制造快照")
    original, metadata_complete, lineage = _source_metadata(source_root, arm)
    protocol = _read(source_root / "protocol.json") if (source_root / "protocol.json").is_file() else {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name("." + destination.name + ".building-" + uuid.uuid4().hex)
    staging.mkdir()
    generated: list[Path] = []
    try:
        databases, hashes, per_sample = {}, {}, {}
        for sample, source in paths.items():
            relative = "databases/" + _hash(sample)[:24] + ".db"
            target = staging / relative
            hashes[sample] = _backup(source, target)
            generated.append(target)
            databases[sample] = relative
            per_sample[sample] = database_stats(target)
        manifest = {"schema_version": 1, "kind": "memory_snapshot", "status": "ready", "id": snapshot_id,
                    "name": name.strip(), "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_run_id": record.get("id", source_root.name), "source_arm_id": arm_id,
                    "source_repeat": build_repeat, "requested_repeat": repeat,
                    "config": original, "construction_metadata_complete": metadata_complete, "protocol": protocol,
                    "dataset_fingerprint": dataset.get("fingerprint"), "sample_ids": list(paths), "expected_sample_ids": expected,
                    "coverage_complete": len(paths) == len(expected), "databases": databases, "hashes": hashes,
                    "stats": {key: sum(value[key] for value in per_sample.values()) for key in ("exchanges", "messages", "facts", "episodes")},
                    "per_sample": per_sample, **lineage}
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        generated.append(manifest_path)
        # New directory only: no published snapshot can ever be overwritten.
        staging.rename(destination)
    except Exception:
        for path in reversed(generated):
            path.unlink(missing_ok=True)
        # Any incomplete SQLite backup is our private staging artifact as well.
        if (staging / "databases").is_dir():
            for path in (staging / "databases").iterdir():
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            (staging / "databases").rmdir()
        staging.rmdir()
        raise
    return read_snapshot(destination)


def read_snapshot(folder: Path | str) -> dict:
    root = Path(folder).absolute()
    manifest = _read(_contained_file(root, root / "manifest.json"))
    if manifest.get("kind") != "memory_snapshot" or manifest.get("schema_version") != 1 or manifest.get("status") != "ready":
        raise ValueError("快照清单格式无效或尚未完成")
    identifier = manifest.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"snapshot-[a-f0-9]{24}", identifier) or identifier != root.name:
        raise ValueError("快照清单 id 与发布目录不一致")
    databases = manifest.get("databases")
    samples = manifest.get("sample_ids")
    hashes = manifest.get("hashes")
    if not isinstance(databases, dict) or not databases or not isinstance(samples, list) or not isinstance(hashes, dict):
        raise ValueError("快照清单缺少会话数据库或指纹")
    if any(not isinstance(sample, str) or not sample for sample in samples):
        raise ValueError("快照会话 id 必须是非空字符串")
    if len(set(samples)) != len(samples) or set(samples) != set(databases) or set(samples) != set(hashes):
        raise ValueError("快照清单的会话、数据库与指纹不一致")
    expected = manifest.get("expected_sample_ids")
    if (not isinstance(expected, list) or not expected or
            any(not isinstance(sample, str) or not sample for sample in expected) or len(set(expected)) != len(expected)):
        raise ValueError("快照清单的预期会话列表无效")
    if (not set(samples).issubset(expected) or not isinstance(manifest.get("coverage_complete"), bool) or
            manifest["coverage_complete"] != (set(samples) == set(expected))):
        raise ValueError("快照覆盖状态与实际会话列表不一致")
    resolved = {}
    for sample, relative in databases.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("快照数据库必须使用相对路径")
        if not isinstance(hashes[sample], str) or not re.fullmatch(r"[a-f0-9]{64}", hashes[sample]):
            raise ValueError("快照数据库指纹无效")
        resolved[sample] = str(_contained_file(root / "databases", root / relative))
    return {**manifest, "databases": resolved}


def validate_snapshot_hashes(manifest: dict) -> None:
    for sample, database in manifest["databases"].items():
        if _database_hash(Path(database)) != manifest["hashes"][sample]:
            raise ValueError(f"快照会话 {sample} 的数据库内容已变化")
