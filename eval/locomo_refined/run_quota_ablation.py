"""Isolate removal of type quotas by replaying frozen full-vector candidates.

No retrieval, embeddings, memory writes, or oracle calls are needed. Existing
reranking (including subject diversity), final K, rendering and budget are held
fixed. Only changed contexts require a new short-answer model call.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from eval.locomo_refined.evaluate import summarize
from eval.locomo_refined.quota_ablation import replay_selection
from eval.locomo_refined.run_smoke import ANSWER_SYSTEM, answer
from eval.locomo_refined.run_vector_ablation import (
    digest, prediction, read_json, score, stage_evidence, write_json,
)
from gugugaga.config import Settings
from gugugaga.provider import SiliconFlowProvider


class AnswerOnlyProvider:
    def __init__(self, provider):
        self.provider = provider
        self.last_response = None

    def create(self, **kwargs):
        self.last_response = self.provider.create(**kwargs)
        return self.last_response

    def embed(self, *args, **kwargs):
        raise RuntimeError("Embedding calls are disabled for the quota-only ablation")


def detail_path(source: Path, sample: str) -> Path:
    verified = source / sample / "details_verified.json"
    return verified if verified.exists() else source / sample / "details.json"


def trace_path(source: Path, sample: str, qa_id: str) -> Path:
    return source / sample / "traces" / (qa_id.replace("#", "_") + ".json")


def prepare(source: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prior_manifest = read_json(source / "manifest.json")
    protocol = prior_manifest["protocol"]
    if read_json(source / "run-status.json")["status"] != "complete":
        raise ValueError("Source ablation must be complete")
    if protocol["final_limit"] != 5 or protocol["experimental_hot_exchanges"] != 10000:
        raise ValueError("Expected the full-vector Top 5 source ablation")
    if hashlib.sha256(ANSWER_SYSTEM.encode()).hexdigest() != protocol["answer_system_sha256"]:
        raise ValueError("Answer prompt changed from source experiment")
    files = [source / name for name in ("manifest.json", "batch-summary.json", "run-status.json")]
    counts = Counter()
    for sample in prior_manifest["sample_ids"]:
        path = detail_path(source, sample)
        files.append(path)
        files.append(source / sample / "predictions_current_memory.json")
        rows = read_json(path)
        if len({row["qa_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate question IDs")
        predictions = read_json(source / sample / "predictions_current_memory.json")
        if [row["qa_id"] for row in rows] != [row["qa_id"] for row in predictions]:
            raise ValueError("Question order differs from the frozen baseline")
        for row, prior in zip(rows, predictions):
            path = trace_path(source, sample, row["qa_id"])
            files.append(path)
            original = read_json(path)["full_vectors"]
            routed = replay_selection(original, policy="routed")
            changed = replay_selection(original, policy="no_type_quota")
            if routed["final"]["content"] != row["retrieved_memories"]:
                raise ValueError(f"Baseline context mismatch: {row['qa_id']}")
            if prior["predicted_answer"] != row["current_memory_answer"]:
                raise ValueError("Baseline answer mismatch")
            counts["questions"] += 1
            counts["baseline_context_matches"] += 1
            counts["unchanged_contexts"] += changed["final"]["content"] == routed["final"]["content"]
            counts["changed_contexts"] += changed["final"]["content"] != routed["final"]["content"]
            counts["post_budget_less_than_5"] += len(changed["final"]["candidates"]) < 5
    code_files = (
        "eval/locomo_refined/run_quota_ablation.py", "eval/locomo_refined/quota_ablation.py",
        "eval/locomo_refined/run_vector_ablation.py", "eval/locomo_refined/run_smoke.py",
        "eval/locomo_refined/evaluate.py", "gugugaga/memory/retrieval.py", "gugugaga/provider.py",
    )
    manifest = {
        "schema_version": 1, "source_run": str(source), "sample_ids": prior_manifest["sample_ids"],
        "frozen_source_sha256": {str(path.relative_to(source)): digest(path) for path in files},
        "code_sha256": {name: digest(ROOT / name) for name in code_files},
        "protocol": {
            "answer_model": protocol["answer_model"], "temperature": protocol["temperature"],
            "enable_thinking": protocol["enable_thinking"], "answer_max_tokens": protocol["answer_max_tokens"],
            "answer_system_sha256": protocol["answer_system_sha256"],
            "candidate_limit": protocol["candidate_limit"], "final_limit": 5,
            "recall_token_budget": protocol["recall_token_budget"], "min_score": protocol["min_score"],
            "vector_coverage": "full historical evidence, inherited unchanged from source",
            "baseline_selection": "existing type quotas",
            "experimental_selection": "first 5 of the exact same frozen rerank candidates",
            "rerank_subject_diversity": "unchanged, not re-sorted by final_score",
            "retrieval_route_query_vectors": "all upstream trace stages frozen; no new retrieval calls",
            "answer_reuse": "baseline answers reused; experiment reuses baseline only for byte-identical contexts",
            "no_memory_oracle": "reuse source summary, no new calls", "database_writes": False,
        },
        "preflight": dict(counts),
    }
    return manifest, dict(counts)


def run_sample(source: Path, output: Path, settings: Settings, sample: str, protocol: dict[str, Any]):
    folder = output / sample
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        return read_json(folder / "summary.json")
    provider = AnswerOnlyProvider(SiliconFlowProvider(
        settings, enable_thinking=protocol["enable_thinking"], temperature=protocol["temperature"]
    ))
    rows = read_json(detail_path(source, sample))
    checkpoint = folder / "details.json"
    completed = read_json(checkpoint) if checkpoint.exists() else []
    ids = {row["qa_id"] for row in completed}
    for row in rows:
        if row["qa_id"] in ids:
            continue
        original_path = trace_path(source, sample, row["qa_id"])
        original = read_json(original_path)["full_vectors"]
        baseline = replay_selection(original, policy="routed")
        experiment = replay_selection(original, policy="no_type_quota")
        content = experiment["final"]["content"]
        reused = content == baseline["final"]["content"]
        write_json(folder / "traces" / original_path.name, {
            "qa_id": row["qa_id"], "source_trace": str(original_path),
            "source_arm": "full_vectors", "no_type_quota": experiment,
        })
        if reused:
            value, cost = row["current_memory_answer"], None
            stop_reason = row["answer_stop_reason"]
        else:
            value, cost = answer(provider, row["question"], content, model=protocol["answer_model"])
            stop_reason = provider.last_response.stop_reason
        gold = set(row["gold_evidence_turn_ids"])
        completed.append({
            **{key: row[key] for key in ("qa_id", "sample_id", "question", "category", "gold_answer")},
            "baseline_answer": row["current_memory_answer"],
            "baseline_f1": score(row, row["current_memory_answer"]),
            "baseline_retrieved_memories": baseline["final"]["content"],
            "baseline_retrieved_count": len(baseline["final"]["candidates"]),
            "current_memory_answer": value, "memory_f1": score(row, value),
            "current_memory_token_cost": cost, "answer_stop_reason": stop_reason,
            "answer_reused_for_identical_context": reused,
            "retrieved_memories": content, "retrieved_count": len(experiment["final"]["candidates"]),
            "retrieval_route": experiment["final"]["route"],
            "retrieved_kinds": experiment["final"]["kinds"],
            "retrieved_item_kinds": [item["kind"] for item in experiment["final"]["candidates"]],
            "gold_evidence_turn_ids": sorted(gold),
            "baseline_stage_evidence": stage_evidence(baseline, gold),
            "no_quota_stage_evidence": stage_evidence(experiment, gold),
        })
        write_json(checkpoint, completed)
        write_json(folder / "progress.json", {"completed": len(completed), "total": len(rows)})
        print(f"{sample} {len(completed)}/{len(rows)} quotas={score(row,row['current_memory_answer']):.1f} no_quotas={score(row,value):.1f}{' cached' if reused else ''}", flush=True)
    predictions = [prediction(row, row["current_memory_answer"], row["retrieved_memories"], row["current_memory_token_cost"]) for row in completed]
    baseline_predictions = [prediction(row, row["baseline_answer"], row["baseline_retrieved_memories"]) for row in completed]
    write_json(folder / "predictions_current_memory.json", predictions)
    result = {
        "sample_id": sample, "baseline": summarize(baseline_predictions), "no_type_quota": summarize(predictions),
        "reused_answers": sum(row["answer_reused_for_identical_context"] for row in completed),
    }
    write_json(folder / "summary.json", result)
    return result


def aggregate(source: Path, output: Path, samples: list[str]):
    rows = [row for sample in samples if (output / sample / "summary.json").exists()
            for row in read_json(output / sample / "details.json")]
    baseline = summarize([prediction(row, row["baseline_answer"], row["baseline_retrieved_memories"]) for row in rows])
    experiment = summarize([prediction(row, row["current_memory_answer"], row["retrieved_memories"]) for row in rows])
    stages = {}
    for arm in ("baseline", "no_quota"):
        stats = {}
        for row in rows:
            gold = row["gold_evidence_turn_ids"]
            if not gold:
                continue
            for name, stage in row[f"{arm}_stage_evidence"].items():
                count = stats.setdefault(name, Counter())
                count["scorable_questions"] += 1
                count["gold_turn_pairs"] += len(gold)
                for kind in ("source", "direct"):
                    hits = len(stage[f"{kind}_gold_turns"])
                    count[f"{kind}_hit_questions"] += hits > 0
                    count[f"{kind}_complete_questions"] += hits == len(gold)
                    count[f"{kind}_hit_turn_pairs"] += hits
        stages[arm] = {name: dict(count) for name, count in stats.items()}
    original = read_json(source / "batch-summary.json")
    result = {
        "question_count": len(rows), "baseline": baseline, "no_type_quota": experiment,
        "delta_f1": experiment["overall_f1"] - baseline["overall_f1"],
        "historical_memory": original["historical_memory"], "oracle": original["oracle"],
        "improved_questions": sum(row["memory_f1"] > row["baseline_f1"] for row in rows),
        "regressed_questions": sum(row["memory_f1"] < row["baseline_f1"] for row in rows),
        "unchanged_questions": sum(row["memory_f1"] == row["baseline_f1"] for row in rows),
        "answer_calls": sum(not row["answer_reused_for_identical_context"] for row in rows),
        "reused_answers": sum(row["answer_reused_for_identical_context"] for row in rows),
        "stop_reasons": dict(Counter(row["answer_stop_reason"] for row in rows)),
        "final_item_count": dict(Counter(row["retrieved_count"] for row in rows)),
        "stage_evidence": stages,
    }
    write_json(output / "batch-summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the frozen source")
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1 to 4")
    manifest, counts = prepare(source, output)
    if args.validate_only:
        print(counts)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Source or code changed since this output was created")
    write_json(manifest_path, manifest)
    for name, expected in manifest["code_sha256"].items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Code changed during preparation")
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    load_dotenv(ROOT / ".env")
    settings = Settings.from_env(ROOT, model_override=manifest["protocol"]["answer_model"])
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_sample, source, output, settings, sample, manifest["protocol"]): sample
                   for sample in manifest["sample_ids"]}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                future.result()
                current = aggregate(source, output, manifest["sample_ids"])
                print(f"DONE {sample}: {current['question_count']} questions delta={current['delta_f1']:+.3f}", flush=True)
            except Exception as error:
                errors.append({"sample_id": sample, "error_type": type(error).__name__, "error": str(error)[:500]})
                print(f"FAILED {sample}: {type(error).__name__}: {str(error)[:250]}", flush=True)
                write_json(output / "errors.json", errors)
    aggregate(source, output, manifest["sample_ids"])
    changed = [name for name, expected in manifest["frozen_source_sha256"].items()
               if digest(source / name) != expected]
    if changed:
        errors.append({"source_files_changed": changed})
    write_json(output / "run-status.json", {
        "status": "failed" if errors else "complete", "errors": errors,
        "source_files_verified_unchanged": len(manifest["frozen_source_sha256"]) - len(changed),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
