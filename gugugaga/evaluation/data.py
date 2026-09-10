"""Read and freeze local LoCoMo questions without touching memory databases."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FROZEN_RUN = "qwen36-full-200-oracle-20260905-021445"
CATEGORY_ORDER = ("4", "1", "3", "2")
CATEGORY_LABELS = {"1": "Multi-hop", "2": "Temporal", "3": "Open-domain", "4": "Single-hop"}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"dataset file is missing: {path.name}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read dataset {path.name}: {error}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"dataset {path.name} must contain JSON objects")
    return rows


def _dataset(repo_root: Path | str) -> tuple[list[dict], list[dict]]:
    folder = Path(repo_root).resolve() / "eval" / "locomo_refined" / "data"
    conversations = _read_rows(folder / "conversations.jsonl")
    questions = _read_rows(folder / "questions.jsonl")
    samples: set[str] = set()
    for item in conversations:
        sample = item.get("sample_id")
        if not isinstance(sample, str) or not sample or sample in samples:
            raise ValueError("dataset has missing or duplicate sample_id")
        if not isinstance(item.get("sessions"), list):
            raise ValueError(f"conversation {sample} has no sessions array")
        samples.add(sample)
    ids: set[str] = set()
    for item in questions:
        qa_id = item.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id or qa_id in ids:
            raise ValueError("dataset has missing or duplicate qa_id")
        if item.get("sample_id") not in samples:
            raise ValueError(f"question {qa_id} references an unknown conversation")
        if not isinstance(item.get("question"), str) or "answer" not in item:
            raise ValueError(f"question {qa_id} is missing question/answer")
        ids.add(qa_id)
    return conversations, questions


def _ids(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must be an array of nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return list(value)


def _portable_frozen_ids(repo_root: Path | str) -> list[str]:
    """Wheel installations keep the original question order, without run DBs."""
    path = Path(__file__).with_name("frozen200.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("unsupported portable frozen question manifest")
        ids = _ids(manifest.get("qa_ids"), "portable frozen qa_ids")
        samples = _ids(manifest.get("sample_ids"), "portable frozen sample_ids")
        expected = manifest.get("qa_ids_sha256")
        actual = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if len(ids) != 200 or len(samples) != 10 or actual != expected:
            raise ValueError("portable frozen question count/order checksum does not match")
        if manifest.get("dataset_hash_format") != "canonical-json-records-v1":
            raise ValueError("unknown portable dataset fingerprint format")
        data = Path(repo_root).resolve() / "eval" / "locomo_refined" / "data"
        for name in ("conversations.jsonl", "questions.jsonl"):
            rows = _read_rows(data / name)
            digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if digest != manifest.get("dataset_sha256", {}).get(name):
                raise ValueError(f"portable frozen question source does not match {name}; use custom for a different dataset")
        return ids
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"frozen 200-question source is unavailable: {error}") from error


def _frozen_ids(repo_root: Path | str) -> list[str]:
    folder = Path(repo_root).resolve() / "eval" / "locomo_refined" / "runs" / FROZEN_RUN
    summary = folder / "batch-summary.json"
    if not summary.is_file():
        return _portable_frozen_ids(repo_root)
    try:
        value = json.loads(summary.read_text(encoding="utf-8-sig"))
        entries = value.get("per_sample") or []
        sample_ids = [entry["sample_id"] for entry in entries]
        if not sample_ids:
            sample_ids = sorted(child.name for child in folder.iterdir() if child.is_dir() and child.name.startswith("conv-"))
        ids: list[str] = []
        for sample_id in sample_ids:
            # Never follow sample names from JSON outside the frozen run.
            child = (folder / str(sample_id)).resolve()
            if child.parent != folder.resolve():
                raise ValueError("unsafe sample_id in frozen question source")
            path = child / "details.json"
            if not path.is_file():
                path = child / "predictions_current_memory.json"
            rows = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(rows, list) or len(rows) != 20:
                raise ValueError(f"frozen sample {sample_id} must contain 20 questions")
            ids.extend(str(row["qa_id"]) for row in rows)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid frozen question source: {error}") from error
    if len(sample_ids) != 10 or len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("frozen source must contain 10 conversations and 200 unique question IDs")
    return ids


def _sample(rows: list[dict], maximum: int) -> list[dict]:
    """Use the existing smoke protocol's category order, then fill shortages."""
    per_category = max(1, maximum // 4)
    selected = [item for category in CATEGORY_ORDER for item in [row for row in rows if str(row.get("category")) == category][:per_category]]
    selected_ids = {item["qa_id"] for item in selected}
    selected.extend(item for item in rows if item["qa_id"] not in selected_ids)
    return selected[:maximum]


def load_dataset(repo_root: Path | str, selection: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise ValueError("dataset selection must be an object")
    mode = selection.get("mode", "smoke")
    if mode not in {"smoke", "frozen200", "custom"}:
        raise ValueError("dataset mode must be smoke, frozen200 or custom")
    samples = _ids(selection.get("sample_ids"), "sample_ids")
    requested = _ids(selection.get("qa_ids"), "qa_ids")
    maximum = selection.get("max_questions", 20)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 10000:
        raise ValueError("max_questions must be an integer between 1 and 10000")
    conversations, all_questions = _dataset(repo_root)
    by_sample = {item["sample_id"]: item for item in conversations}
    by_id = {item["qa_id"]: item for item in all_questions}
    if any(sample not in by_sample for sample in samples):
        raise ValueError("selection contains an unknown sample_id")
    if mode == "frozen200":
        frozen = _frozen_ids(repo_root)
        if requested and requested != frozen:
            raise ValueError("frozen200 question IDs/order must match the frozen source; use custom for a subset")
        requested = frozen
        actual_samples = list(dict.fromkeys(by_id[item]["sample_id"] for item in requested if item in by_id))
        if samples and samples != actual_samples:
            raise ValueError("frozen200 sample_ids/order must match the frozen source")
        samples = actual_samples
    if requested:
        unknown = [qa_id for qa_id in requested if qa_id not in by_id]
        if unknown:
            raise ValueError(f"unknown qa_id: {unknown[0]}")
        selected = [by_id[qa_id] for qa_id in requested]
        if samples and any(item["sample_id"] not in samples for item in selected):
            raise ValueError("qa_ids contain questions outside selected sample_ids")
        used_samples = list(dict.fromkeys(item["sample_id"] for item in selected))
    else:
        available = [item["sample_id"] for item in sorted(conversations, key=lambda item: (item.get("conversation_idx", 0), item["sample_id"])) if any(q["sample_id"] == item["sample_id"] for q in all_questions)]
        if not samples:
            if mode == "custom":
                raise ValueError("custom selection requires sample_ids or qa_ids")
            samples = available[:1]
        if mode == "smoke" and len(samples) != 1:
            raise ValueError("smoke selection requires exactly one conversation; use custom for multiple")
        selected = []
        for sample in samples:
            rows = [item for item in all_questions if item["sample_id"] == sample]
            if not rows:
                raise ValueError(f"conversation {sample} has no questions")
            selected.extend(_sample(rows, maximum))
        used_samples = list(samples)
    if not selected:
        raise ValueError("selection contains no questions")
    chosen_conversations = [by_sample[sample] for sample in used_samples]
    frozen_selection = {"mode": mode, "sample_ids": used_samples, "max_questions": maximum, "qa_ids": [item["qa_id"] for item in selected]}
    fingerprint = hashlib.sha256(json.dumps({"conversations": chosen_conversations, "questions": selected}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"conversations": chosen_conversations, "questions": selected, "fingerprint": fingerprint, "selection": frozen_selection}


def dataset_catalog(repo_root: Path | str) -> dict[str, Any]:
    result: dict[str, Any] = {"id": "locomo_refined", "name": "LoCoMo-Refined", "available": False, "samples": [], "modes": ["smoke", "frozen200", "custom"], "max_questions_scope": "per_conversation"}
    try:
        conversations, questions = _dataset(repo_root)
        for conversation in conversations:
            rows = [item for item in questions if item["sample_id"] == conversation["sample_id"]]
            counts = Counter(CATEGORY_LABELS.get(str(item.get("category")), str(item.get("category"))) for item in rows)
            result["samples"].append({"sample_id": conversation["sample_id"], "question_count": len(rows), "categories": dict(counts), "questions": [{"qa_id": item["qa_id"], "question": item["question"], "category": CATEGORY_LABELS.get(str(item.get("category")), str(item.get("category")))} for item in rows]})
        result.update(available=True, question_count=len(questions), conversation_count=len(conversations))
    except ValueError as error:
        result["error"] = str(error)
    try:
        frozen = _frozen_ids(repo_root)
        known_ids = {item["qa_id"] for item in questions} if result["available"] else set()
        if not set(frozen).issubset(known_ids):
            raise ValueError("frozen question IDs are not available in the local dataset")
        result["frozen200"] = {"available": True, "question_count": 200, "qa_ids": frozen}
    except ValueError as error:
        result["frozen200"] = {"available": False, "error": str(error)}
    return result
