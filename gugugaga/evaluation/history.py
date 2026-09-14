"""Bounded, read-only adapters for the existing LoCoMo experiment artifacts.

Legacy files are evidence, not executable configuration. In particular, this
module never opens a memory service or a database and never follows arbitrary
paths from a manifest outside the known runs directory.
"""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.locomo_refined.evaluate import CATEGORY_LABELS, bleu1, token_f1


SUMMARY_NAMES = ("batch-summary.json", "repeat-summary.json", "summary.json")
PENDING_TOPK = "topk-answers-20260906-step4"
MISSING_WARNING = "历史字段缺失按未记录展示；来源命中不等于答案证据充分。"
SCORE_WARNING = "Token F1 是文本重合度；Oracle 是相同回答协议下的参考条件。历史评分器版本未完整记录。"


def _number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _score(value: Any) -> int | float | None:
    number = _number(value)
    return number if number is not None and 0 <= number <= 100.000001 else None


def _metrics(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    categories = {}
    for key, item in (value.get("categories") or {}).items():
        if isinstance(item, dict):
            categories[str(key)] = {"count": _number(item.get("count")), "f1": _score(item.get("f1")), "bleu1": _score(item.get("bleu1"))}
    return {"question_count": _number(value.get("question_count")), "overall_f1": _score(value.get("overall_f1")), "overall_bleu1": _score(value.get("overall_bleu1")), "categories": categories}


def _computed(answer: str, gold: Any, metric: Any) -> float | None:
    if gold is None:
        return None
    candidates = gold if isinstance(gold, list) else [gold]
    if not candidates:
        return None
    return max(metric(answer, candidate) for candidate in candidates) * 100.0


class LegacyCatalog:
    def __init__(self, repo_root: Path | str):
        self.repo_root = Path(repo_root).resolve()
        self.root = (self.repo_root / "eval" / "locomo_refined" / "runs").resolve()

    def _safe(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved == self.root or not resolved.is_relative_to(self.root):
            raise ValueError("legacy path must stay inside the evaluation runs directory")
        return resolved

    def _id(self, path: Path) -> str:
        return "legacy:" + self._safe(path).relative_to(self.root).as_posix()

    def _path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not run_id.startswith("legacy:"):
            raise ValueError("invalid legacy run id")
        relative = run_id[len("legacy:"):]
        if not relative or "\\" in relative or ":" in relative or any(part in {"", ".", ".."} for part in relative.split("/")):
            raise ValueError("invalid legacy run path")
        path = self._safe(self.root / relative)
        if not path.is_dir() and relative != PENDING_TOPK:
            raise KeyError("legacy run not found")
        return path

    def _read(self, path: Path, warnings: list[str] | None = None, default: Any = None) -> Any:
        try:
            path = self._safe(path)
            if not path.is_file():
                return default
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("file exceeds the 64 MiB per-artifact read limit")
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            if warnings is not None:
                warnings.append(f"无法读取 {path.name}: {error}")
            return default

    def _summary(self, folder: Path, warnings: list[str] | None = None) -> tuple[dict, str | None]:
        for name in SUMMARY_NAMES:
            if (folder / name).is_file():
                value = self._read(folder / name, warnings)
                return (value if isinstance(value, dict) else {}), name
        return {}, None

    def _children(self, folder: Path) -> list[Path]:
        try:
            children = sorted(folder.iterdir(), key=lambda path: path.name)
        except OSError:
            return []
        output = []
        for child in children:
            try:
                if child.is_dir() and not child.name.startswith(".") and self._safe(child).parent == folder.resolve():
                    output.append(child)
            except (OSError, ValueError):
                continue
        return output

    def _repeat_folders(self, folder: Path, summary: dict) -> list[Path]:
        children = [child for child in self._children(folder) if (child / "summary.json").is_file()]
        by_name = {child.name: child for child in children}
        for child in children:
            child_summary, _ = self._summary(child)
            if child_summary.get("run_id"):
                by_name[str(child_summary["run_id"])] = child
        output: list[Path] = []
        for label in summary.get("runs") or []:
            if not isinstance(label, str):
                continue
            child = by_name.get(label)
            if child is None and "/" not in label and "\\" not in label and ":" not in label and label not in {".", ".."}:
                candidate = self.root / label
                try:
                    if candidate.is_dir() and (self._safe(candidate) / "summary.json").is_file():
                        child = candidate
                except (ValueError, OSError):
                    pass
            if child is not None and child not in output:
                output.append(child)
        # Older summaries store generated run IDs rather than directory names.
        for child in children:
            if child not in output:
                output.append(child)
        return output

    @staticmethod
    def _specs(summary: dict) -> list[tuple[str, str, str]]:
        """(arm id, displayed name, summary metric key), without reference duplicates."""
        if "full_vectors" in summary:
            return [("baseline", "原历史向量覆盖", "baseline_replay"), ("full_vectors", "全量原文向量", "full_vectors")]
        if "no_type_quota" in summary:
            return [("baseline", "全量原文＋摘要＋配额", "baseline"), ("no_type_quota", "关闭类型配额", "no_type_quota")]
        if "raw_only" in summary:
            return [("baseline", "全量原文＋摘要＋配额", "baseline"), ("raw_only", "仅全量原文", "raw_only")]
        output = []
        for arm_id, name, keys in [("memory", "Memory", ("memory", "current_memory")), ("no_memory", "No-Memory", ("no_memory",)), ("oracle", "Gold Evidence Oracle", ("oracle_evidence", "oracle"))]:
            key = next((key for key in keys if isinstance(summary.get(key), dict)), None)
            if key:
                output.append((arm_id, name, key))
        return output

    def _pending_exists(self) -> bool:
        try:
            return PENDING_TOPK in (self.repo_root / "README.md").read_text(encoding="utf-8-sig") and not (self.root / PENDING_TOPK / "batch-summary.json").is_file() and not (self.root / PENDING_TOPK / "summary.json").is_file()
        except OSError:
            return False

    def list_runs(self) -> list[dict[str, Any]]:
        folders = self._children(self.root)
        nested_sources: set[Path] = set()
        for folder in folders:
            summary, filename = self._summary(folder)
            if filename == "repeat-summary.json":
                nested_sources.update(child.resolve() for child in self._repeat_folders(folder, summary))
        result = []
        for folder in folders:
            if folder.resolve() in nested_sources:
                continue
            _, filename = self._summary(folder)
            if not filename:
                continue
            try:
                result.append(self.get_run(self._id(folder)))
            except (OSError, ValueError, KeyError) as error:
                result.append({"id": self._id(folder), "name": folder.name, "type": "development", "status": "failed", "error": str(error), "resumable": False, "legacy": True, "result": {"arms": [], "warnings": [str(error)]}})
        if self._pending_exists() and not any(item["id"] == "legacy:" + PENDING_TOPK for item in result):
            result.append(self.get_run("legacy:" + PENDING_TOPK))
        return sorted(result, key=lambda row: str(row.get("created_at") or row["name"]), reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any]:
        folder = self._path(run_id)
        warnings = [MISSING_WARNING, SCORE_WARNING]
        summary, filename = self._summary(folder, warnings)
        if folder.name == PENDING_TOPK and not filename:
            if not self._pending_exists():
                raise KeyError("legacy run not found")
            return {"id": run_id, "name": "Top K / 预算实验（产物待补）", "type": "ablation", "status": "pending_artifacts", "config": {}, "progress": {}, "result": {"arms": [], "stability": [], "comparison": [], "warnings": ["README 引用了该实验，但当前报告及运行产物缺失；不展示或比较未经核验的成绩。"]}, "created_at": None, "updated_at": None, "error": None, "resumable": False, "legacy": True}
        if filename is None:
            raise KeyError("legacy summary not found")
        manifest = self._read(folder / "manifest.json", warnings, {})
        manifest = manifest if isinstance(manifest, dict) else {}
        warnings.extend(str(item) for item in manifest.get("limitations", []) if isinstance(item, str))
        config = copy.deepcopy(summary.get("config") or manifest.get("protocol") or {})
        if not isinstance(config, dict):
            config = {}
        kind = "stability" if filename == "repeat-summary.json" else "ablation" if any(key in summary for key in ("full_vectors", "no_type_quota", "raw_only")) else "basic" if filename == "batch-summary.json" else "development"
        specs = self._specs(summary)
        planned = summary.get("question_count", summary.get("question_count_per_run"))
        arms = []
        for arm_id, name, key in specs:
            metric = _metrics(summary.get(key))
            arms.append({"id": arm_id, "name": name, "metrics": metric, "completed": metric["question_count"], "planned": planned, "token_cost": None, "repeats": []})
        stability = []
        if kind == "stability":
            for arm_id, name, key in [("memory", "Memory", "memory_f1"), ("no_memory", "No-Memory", "no_memory_f1")]:
                entry = (summary.get("metrics") or {}).get(key)
                if not isinstance(entry, dict):
                    continue
                values = [_score(value) for value in entry.get("values", [])]
                entry = {"arm_id": arm_id, "name": name, "scope": "unknown", "independence": "unknown", "mean": _score(entry.get("mean")), "sample_std": _number(entry.get("sample_std")), "values": values, "completed_repeats": sum(value is not None for value in values), "planned_repeats": summary.get("run_count")}
                stability.append(entry)
                arms.append({"id": arm_id, "name": name, "metrics": {"question_count": planned, "overall_f1": entry["mean"], "overall_bleu1": None, "categories": {}}, "completed": entry["completed_repeats"] * planned if isinstance(planned, int) else None, "planned": summary.get("run_count", 0) * planned if isinstance(planned, int) else None, "token_cost": None, "repeats": [{"repeat": index + 1, "f1": value} for index, value in enumerate(values)]})
            warnings.append("历史重复记录的完整构建范围与独立性未完整记录，不把缓存答案当作新独立样本。")
        if kind == "development":
            warnings.append("早期单会话开发实验可能同时改变多个条件，不作为受控组件消融。")
        costs = summary.get("answer_token_totals") or {}
        for arm in arms:
            arm["token_cost"] = _number(costs.get(arm["id"]))
        try:
            modified = datetime.fromtimestamp((folder / filename).stat().st_mtime, timezone.utc).isoformat()
        except OSError:
            modified = None
        status = "completed" if summary else "failed"
        if summary.get("completed_samples", 0) < summary.get("planned_samples", 0):
            status = "partial"
        config["legacy_protocol"] = True
        result = {"arms": arms, "stability": stability, "comparison": [], "warnings": warnings, "evidence": summary.get("stage_evidence") or summary.get("evidence_retrieval"), "per_sample": summary.get("per_sample") or summary.get("samples"), "score_scale": "0-100", "cost_scope": "answer_only", "scoring_protocol": "legacy_version_unknown"}
        return {"id": run_id, "name": folder.name, "type": kind, "status": status, "config": config, "progress": {"completed": planned if status == "completed" else None, "total": planned}, "result": result, "created_at": modified, "updated_at": modified, "error": None if summary else "无法读取实验汇总", "resumable": False, "legacy": True, "source": folder.relative_to(self.root).as_posix()}

    def _gold_questions(self, warnings: list[str]) -> dict[str, dict]:
        path = self.repo_root / "eval" / "locomo_refined" / "data" / "questions.jsonl"
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            return {row["qa_id"]: row for row in rows if isinstance(row, dict) and isinstance(row.get("qa_id"), str)}
        except (OSError, ValueError):
            warnings.append("本地题库未记录或无法读取；标准证据只显示历史产物已保存的内容。")
            return {}

    def _trace_reference(self, value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        normalized = value.replace("\\", "/")
        marker = "/eval/locomo_refined/runs/"
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
        elif normalized.startswith("eval/locomo_refined/runs/"):
            normalized = normalized[len("eval/locomo_refined/runs/"):]
        elif Path(value).is_absolute():
            try:
                return self._safe(Path(value))
            except ValueError:
                return None
        if any(part in {"", ".", ".."} for part in normalized.split("/")) or ":" in normalized:
            return None
        try:
            return self._safe(self.root / normalized)
        except ValueError:
            return None

    def _trace(self, folder: Path, qa_id: str, arm_id: str, warnings: list[str], enabled: bool) -> tuple[Any, bool]:
        if any(character in qa_id for character in "/\\:") or not qa_id:
            return None, False
        trace_path = folder / "traces" / (qa_id.replace("#", "_") + ".json")
        try:
            exists = self._safe(trace_path).is_file()
        except (ValueError, OSError):
            return None, False
        if not exists or not enabled:
            return None, exists
        raw = self._read(trace_path, warnings, {})
        if not isinstance(raw, dict) or (raw.get("qa_id") and raw["qa_id"] != qa_id):
            warnings.append(f"检索追踪与 qa_id 不匹配: {qa_id}")
            return None, False
        key = {"memory": "memory", "baseline": "baseline"}.get(arm_id, arm_id)
        if isinstance(raw.get(key), dict):
            return raw[key], True
        if "stages" in raw:
            return raw, True
        if arm_id == "baseline":
            source = self._trace_reference(raw.get("source_trace"))
            source_value = self._read(source, warnings, {}) if source else {}
            if isinstance(source_value, dict) and source_value.get("qa_id") in {None, qa_id}:
                value = source_value.get(raw.get("source_arm", "full_vectors"))
                if isinstance(value, dict):
                    return value, True
        return None, False

    def _sample_result(self, folder: Path, specs: list[tuple[str, str, str]], gold: dict[str, dict], warnings: list[str], qa_id: str | None, include_traces: bool) -> list[dict]:
        detail_path = folder / "details_verified.json"
        if not detail_path.is_file():
            detail_path = folder / "details.json"
        rows = self._read(detail_path, warnings, [])
        rows = rows if isinstance(rows, list) else []
        details = {row["qa_id"]: row for row in rows if isinstance(row, dict) and isinstance(row.get("qa_id"), str)}
        diagnostics = self._read(folder / "diagnostics.json", warnings, {})
        diagnosis_rows = diagnostics.get("details", []) if isinstance(diagnostics, dict) else []
        diagnosis_by_id = {item["qa_id"]: item for item in diagnosis_rows if isinstance(item, dict) and isinstance(item.get("qa_id"), str)}
        prediction_maps = {}
        for arm_id, filename in [("memory", "predictions_current_memory.json"), ("no_memory", "predictions_no_memory.json"), ("oracle", "predictions_oracle_evidence.json")]:
            predictions = self._read(folder / filename, warnings, [])
            prediction_maps[arm_id] = {item["qa_id"]: item for item in predictions if isinstance(item, dict) and isinstance(item.get("qa_id"), str)} if isinstance(predictions, list) else {}
        order = list(details)
        for predictions in prediction_maps.values():
            order.extend(key for key in predictions if key not in order)
        if qa_id is not None:
            order = [qa_id] if qa_id in order else []
        output = []
        for key in order:
            detail = details.get(key, {})
            datum = gold.get(key, {})
            metadata = next((mapping[key] for mapping in prediction_maps.values() if key in mapping), {})
            item = {"qa_id": key, "sample_id": detail.get("sample_id", metadata.get("sample_id", datum.get("sample_id", folder.name))), "category": str(detail.get("category", metadata.get("category", datum.get("category", "unknown")))), "question": detail.get("question", metadata.get("question", datum.get("question", ""))), "gold_answer": detail.get("gold_answer", metadata.get("gold_answer", datum.get("answer"))), "gold_evidence": datum.get("evidence_messages"), "gold_evidence_ids": datum.get("evidence", diagnosis_by_id.get(key, {}).get("gold_evidence_ids")), "answers": []}
            if datum:
                if detail.get("gold_answer", metadata.get("gold_answer", datum.get("answer"))) != datum.get("answer"):
                    warnings.append(f"{key} 历史标准答案与当前题库不同；保留历史答案，不宣称评分协议一致。")
            for arm_id, _, _ in specs:
                prediction_id = arm_id if arm_id in {"memory", "no_memory", "oracle"} else "memory"
                prediction = prediction_maps[prediction_id].get(key, {}) if arm_id != "baseline" else {}
                answer_field = "baseline_answer" if arm_id == "baseline" else "no_memory_answer" if arm_id == "no_memory" else "oracle_answer" if arm_id == "oracle" else "current_memory_answer"
                answer = detail.get(answer_field, prediction.get("predicted_answer"))
                diagnosis = diagnosis_by_id.get(key, {})
                if answer is None and arm_id == "oracle":
                    answer = diagnosis.get("oracle_answer")
                f1_key = "baseline_f1" if arm_id == "baseline" else "no_memory_f1" if arm_id == "no_memory" else "oracle_f1" if arm_id == "oracle" else "memory_f1"
                recorded_f1 = _score(detail.get(f1_key, diagnosis.get(f1_key)))
                recorded_bleu = _score(prediction.get("bleu1"))
                memory_field = "baseline_retrieved_memories" if arm_id == "baseline" else "retrieved_memories"
                memory = detail.get(memory_field, prediction.get("retrieved_memories")) if arm_id not in {"no_memory", "oracle"} else prediction.get("retrieved_memories")
                cost_field = "baseline_token_cost" if arm_id == "baseline" else "no_memory_token_cost" if arm_id == "no_memory" else "oracle_token_cost" if arm_id == "oracle" else "current_memory_token_cost"
                error = prediction.get("error")
                is_answer = isinstance(answer, str)
                trace, trace_available = self._trace(folder, key, arm_id, warnings, include_traces) if arm_id not in {"no_memory", "oracle"} else (None, False)
                answer_record = {"arm_id": arm_id, "repeat": 1, "answer": answer if is_answer else None, "f1": recorded_f1 if recorded_f1 is not None else _computed(answer, item["gold_answer"], token_f1) if is_answer and not error else None, "bleu1": recorded_bleu if recorded_bleu is not None else _computed(answer, item["gold_answer"], bleu1) if is_answer and not error else None, "memory": memory, "trace": trace, "trace_available": trace_available, "status": "failed" if error else "completed" if is_answer else "missing", "token_cost": _number(detail.get(cost_field, prediction.get("token_cost"))), "error": error, "score_source": "recorded" if recorded_f1 is not None else "computed_from_stored_answer" if is_answer and item["gold_answer"] is not None else "missing"}
                if diagnosis:
                    answer_record["diagnostics"] = copy.deepcopy(diagnosis)
                evidence_key = "baseline_stage_evidence" if arm_id == "baseline" else "full_vector_stage_evidence" if arm_id == "full_vectors" else "no_quota_stage_evidence" if arm_id == "no_type_quota" else "raw_only_stage_evidence" if arm_id == "raw_only" else None
                if evidence_key and evidence_key in detail:
                    answer_record["stage_evidence"] = detail[evidence_key]
                answer_record["cache_reused"] = bool(detail.get("answer_reused_for_identical_context") or detail.get("answer_reused_from_preliminary_run")) if arm_id not in {"baseline", "no_memory", "oracle"} else None
                item["answers"].append(answer_record)
            output.append(item)
        return output

    def get_result(self, run_id: str, *, qa_id: str | None = None, include_traces: bool = False) -> dict[str, Any]:
        metadata = self.get_run(run_id)
        result = copy.deepcopy(metadata["result"])
        result["questions"] = []
        if metadata["status"] == "pending_artifacts":
            return result
        folder = self._path(run_id)
        summary, filename = self._summary(folder)
        warnings = result["warnings"]
        if filename == "repeat-summary.json":
            merged: dict[str, dict] = {}
            repeat_folders = self._repeat_folders(folder, summary)
            for repeat, child in enumerate(repeat_folders, 1):
                child_result = self.get_result(self._id(child), qa_id=qa_id, include_traces=include_traces)
                for question in child_result["questions"]:
                    entry = merged.setdefault(question["qa_id"], {**question, "answers": []})
                    entry["answers"].extend({**answer, "repeat": repeat} for answer in question["answers"])
            result["questions"] = list(merged.values())
            if len(repeat_folders) != summary.get("run_count"):
                warnings.append("部分重复运行的逐题文件不可用，聚合成绩与可检查的子记录数量可能不同。")
        else:
            specs = self._specs(summary)
            if not specs:
                warnings.append("该历史格式没有可识别的评分分组。")
            gold = self._gold_questions(warnings)
            samples = [folder] if filename == "summary.json" else [child for child in self._children(folder) if child.name.startswith("conv-")]
            # Some single runs acquired Oracle diagnostics after summary.json was written.
            if filename == "summary.json" and not any(arm[0] == "oracle" for arm in specs):
                diagnostic = self._read(folder / "diagnostics.json", warnings, {})
                if isinstance(diagnostic, dict) and isinstance(diagnostic.get("oracle_evidence"), dict):
                    specs.append(("oracle", "Gold Evidence Oracle", "oracle_evidence"))
                    metric = _metrics(diagnostic["oracle_evidence"])
                    result["arms"].append({"id": "oracle", "name": "Gold Evidence Oracle", "metrics": metric, "completed": metric["question_count"], "planned": summary.get("question_count"), "token_cost": None, "repeats": []})
            for sample in samples:
                result["questions"].extend(self._sample_result(sample, specs, gold, warnings, qa_id, include_traces))
        if qa_id is None:
            for arm in result["arms"]:
                answers = [answer for item in result["questions"] for answer in item["answers"] if answer["arm_id"] == arm["id"]]
                completed = [answer for answer in answers if answer["status"] == "completed"]
                arm["available_answers"] = len(completed)
                if arm.get("completed") is not None and len(completed) != arm["completed"]:
                    warnings.append(f"{arm['name']} 汇总记录 {arm['completed']} 个答案，当前可查看 {len(completed)} 个；未以缺失答案补零。")
                costs = [answer["token_cost"] for answer in completed]
                arm["recorded_token_cost"] = sum(value for value in costs if value is not None) if any(value is not None for value in costs) else None
                arm["token_cost_complete"] = bool(costs) and all(value is not None for value in costs)
                if arm.get("token_cost") is None and arm["token_cost_complete"]:
                    arm["token_cost"] = sum(costs)
        result["comparison"] = self._comparison(result)
        result["warnings"] = list(dict.fromkeys(warnings))
        return result

    @staticmethod
    def _comparison(result: dict) -> list[dict]:
        arms = result.get("arms") or []
        if len(arms) < 2:
            return []
        baseline = arms[0]["id"]
        output = []
        for arm in arms[1:]:
            deltas = []
            for question in result.get("questions", []):
                answers = {(answer["arm_id"], answer["repeat"]): answer for answer in question["answers"]}
                for (arm_id, repeat), answer in answers.items():
                    original = answers.get((baseline, repeat))
                    if arm_id == arm["id"] and original and answer["status"] == original["status"] == "completed" and answer["f1"] is not None and original["f1"] is not None:
                        deltas.append(answer["f1"] - original["f1"])
            output.append({"baseline_arm_id": baseline, "arm_id": arm["id"], "paired_count": len(deltas), "delta_f1": sum(deltas) / len(deltas) if deltas else None, "improved": sum(value > 1e-9 for value in deltas), "regressed": sum(value < -1e-9 for value in deltas), "unchanged": sum(abs(value) <= 1e-9 for value in deltas), "causal_claim": False})
        return output

    def snapshots(self) -> list[dict[str, Any]]:
        """Enumerate only known database layouts; do not connect to SQLite."""
        output = []
        for folder in self._children(self.root):
            summary, filename = self._summary(folder)
            if not filename:
                continue
            sample_folders = [folder] if filename == "summary.json" else [child for child in self._children(folder) if child.name.startswith("conv-")]
            variants: dict[str, dict[str, str]] = {}
            sample_configs = []
            for sample_folder in sample_folders:
                sample_summary, _ = self._summary(sample_folder)
                sample_id = sample_summary.get("sample_id") or (sample_folder.name if sample_folder.name.startswith("conv-") else None)
                if not isinstance(sample_id, str) or not sample_id:
                    continue
                sample_configs.append(sample_summary.get("config") or {})
                database_dir = sample_folder / "databases"
                try:
                    files = sorted(database_dir.iterdir()) if database_dir.is_dir() else []
                except OSError:
                    continue
                for database in files:
                    try:
                        safe = self._safe(database)
                        if safe.parent != database_dir.resolve() or not safe.is_file() or safe.suffix != ".db":
                            continue
                    except (OSError, ValueError):
                        continue
                    variant = "memory" if database.stem == sample_id else database.stem
                    variants.setdefault(variant, {})[sample_id] = str(safe)
            config = copy.deepcopy(summary.get("config") or (sample_configs[0] if sample_configs else {}))
            if not isinstance(config, dict):
                config = {}
            for variant, databases in variants.items():
                output.append({"id": self._id(folder) + ":snapshot:" + variant, "name": folder.name + " / " + variant, "databases": databases, "sample_ids": list(databases), "config": config, "construction_metadata_complete": False})
        return output
