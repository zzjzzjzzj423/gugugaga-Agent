"""Compare the frozen full-vector hybrid memory against raw-conversation only.

Fact/Episode indexes are removed on isolated copies before either retrieval
branch takes Top 20. Query vectors, rerank clock, final K, budget and answer
protocol are reused from the source experiment. No new embeddings are needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from eval.locomo_refined.evaluate import summarize
from eval.locomo_refined.quota_ablation import replay_selection
from eval.locomo_refined.raw_only_ablation import prepare_raw_only_indexes, retrieve_raw_only
from eval.locomo_refined.run_quota_ablation import AnswerOnlyProvider, detail_path, trace_path
from eval.locomo_refined.run_smoke import ANSWER_SYSTEM, answer
from eval.locomo_refined.run_vector_ablation import (
    coverage, digest, prediction, read_json, score, snapshot, stage_evidence, write_json,
)
from gugugaga.config import Settings
from gugugaga.memory.repository import MemoryRepository
from gugugaga.provider import SiliconFlowProvider


def prepare(source: Path):
    prior = read_json(source / "manifest.json")
    protocol = prior["protocol"]
    if read_json(source / "run-status.json")["status"] != "complete":
        raise ValueError("Expected a complete source run")
    if protocol["final_limit"] != 5 or protocol["experimental_hot_exchanges"] != 10000:
        raise ValueError("Expected full-vector Top 5 baseline")
    if hashlib.sha256(ANSWER_SYSTEM.encode()).hexdigest() != protocol["answer_system_sha256"]:
        raise ValueError("Answer prompt has changed")
    files = [source / name for name in ("manifest.json", "batch-summary.json", "run-status.json")]
    counts = Counter()
    for sample in prior["sample_ids"]:
        rows = read_json(detail_path(source, sample))
        queries = read_json(source / sample / "query_vectors.json")
        files.extend([detail_path(source, sample), source / sample / "query_vectors.json",
                      source / sample / "predictions_current_memory.json",
                      source / sample / "databases" / "full_vectors.db",
                      source / sample / "databases" / "baseline.db"])
        predictions = read_json(source / sample / "predictions_current_memory.json")
        if [row["qa_id"] for row in rows] != [row["qa_id"] for row in predictions]:
            raise ValueError("Question ID/order mismatch")
        for row in rows:
            path = trace_path(source, sample, row["qa_id"])
            files.append(path)
            original = read_json(path)["full_vectors"]
            routed = replay_selection(original)
            if routed["final"]["content"] != row["retrieved_memories"]:
                raise ValueError("Baseline content mismatch")
            if not queries.get(row["question"].strip()):
                raise ValueError("Missing cached query vector")
            counts["questions"] += 1
        state = coverage(source / sample / "databases" / "full_vectors.db", protocol["embedding_model"])
        if state["chat_vectors"] != state["chat_rows"]:
            raise ValueError("Raw vector coverage is incomplete")
        counts["chat_rows"] += state["chat_rows"]
        counts["summary_records"] += state["memory_counts"]["facts"] + state["memory_counts"]["episodes"]
        counts["consolidation_batches"] += state["memory_counts"]["consolidation_batches"]
    code_files = (
        "eval/locomo_refined/run_raw_only_ablation.py", "eval/locomo_refined/raw_only_ablation.py",
        "eval/locomo_refined/run_vector_ablation.py", "eval/locomo_refined/run_quota_ablation.py",
        "eval/locomo_refined/quota_ablation.py", "eval/locomo_refined/run_smoke.py",
        "eval/locomo_refined/evaluate.py", "gugugaga/memory/repository.py",
        "gugugaga/memory/retrieval.py", "gugugaga/provider.py",
    )
    manifest = {
        "schema_version": 1, "source_run": str(source), "sample_ids": prior["sample_ids"],
        "frozen_source_sha256": {str(path.relative_to(source)): digest(path) for path in files},
        "code_sha256": {name: digest(ROOT / name) for name in code_files}, "preflight": dict(counts),
        "protocol": {
            **{key: protocol[key] for key in (
                "answer_model", "embedding_model", "temperature", "enable_thinking", "answer_max_tokens",
                "answer_system_sha256", "candidate_limit", "final_limit", "recall_token_budget", "min_score",
            )},
            "baseline": "full raw evidence plus Facts and Episodes with existing type quotas",
            "experiment": "raw chat only, excluded summaries before BM25/vector candidate limits",
            "raw_vectors": "reuse all 6022 existing message vectors; no new embedding calls",
            "lexical_corpus": "only chat FTS rows, preserving lexical baseline row IDs; IDF changes naturally",
            "rerank": "existing algorithm, true embeddings for dedup; clock frozen per source question",
            "route": "reuse source route; existing selector fills all 5 from chat-only candidates",
            "answer_reuse": "reuse baseline answers for identical injected contexts only",
            "database_writes": "isolated copies only; no service/worker/consolidation/usage aggregation",
        },
    }
    return manifest


def run_sample(source: Path, output: Path, settings: Settings, sample: str, protocol):
    folder = output / sample
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        return read_json(folder / "summary.json")
    target_db = folder / "databases" / "raw_only.db"
    snapshot(source / sample / "databases" / "full_vectors.db", target_db)
    # Initializing a repository rebuilds FTS, so the raw-only filter must follow.
    repository = MemoryRepository(target_db)
    index_report = prepare_raw_only_indexes(target_db, source / sample / "databases" / "baseline.db")
    write_json(folder / "index-verification.json", index_report)
    provider = AnswerOnlyProvider(SiliconFlowProvider(
        settings, enable_thinking=protocol["enable_thinking"], temperature=protocol["temperature"]
    ))
    queries = read_json(source / sample / "query_vectors.json")
    rows = read_json(detail_path(source, sample))
    checkpoint = folder / "details.json"
    completed = read_json(checkpoint) if checkpoint.exists() else []
    completed_ids = {row["qa_id"] for row in completed}
    for row in rows:
        if row["qa_id"] in completed_ids:
            continue
        original_path = trace_path(source, sample, row["qa_id"])
        original = read_json(original_path)["full_vectors"]
        raw = retrieve_raw_only(repository, original, queries[row["question"].strip()])
        content = raw["final"]["content"]
        write_json(folder / "traces" / original_path.name, {
            "qa_id": row["qa_id"], "source_trace": str(original_path), "source_arm": "full_vectors", "raw_only": raw,
        })
        reused = content == row["retrieved_memories"]
        if reused:
            value, cost, usage, stop = row["current_memory_answer"], None, None, row["answer_stop_reason"]
        else:
            value, cost = answer(provider, row["question"], content, model=protocol["answer_model"])
            usage, stop = provider.last_response.usage, provider.last_response.stop_reason
        gold = set(row["gold_evidence_turn_ids"])
        completed.append({
            **{key: row[key] for key in ("qa_id", "sample_id", "question", "category", "gold_answer")},
            "baseline_answer": row["current_memory_answer"], "baseline_f1": score(row, row["current_memory_answer"]),
            "baseline_retrieved_memories": row["retrieved_memories"],
            "current_memory_answer": value, "memory_f1": score(row, value),
            "current_memory_token_cost": cost, "answer_usage": usage, "answer_stop_reason": stop,
            "answer_reused_for_identical_context": reused, "retrieved_memories": content,
            "retrieved_count": raw["final"]["hit_count"], "retrieved_kinds": raw["final"]["kinds"],
            "retrieval_route": raw["final"]["route"], "gold_evidence_turn_ids": sorted(gold),
            "baseline_stage_evidence": stage_evidence(original, gold),
            "raw_only_stage_evidence": stage_evidence(raw, gold),
        })
        write_json(checkpoint, completed)
        write_json(folder / "progress.json", {"completed": len(completed), "total": len(rows)})
        print(f"{sample} {len(completed)}/{len(rows)} mixed={score(row,row['current_memory_answer']):.1f} raw={score(row,value):.1f}{' cached' if reused else ''}", flush=True)
    predictions = [prediction(row, row["current_memory_answer"], row["retrieved_memories"], row["current_memory_token_cost"]) for row in completed]
    baseline = [prediction(row, row["baseline_answer"], row["baseline_retrieved_memories"]) for row in completed]
    write_json(folder / "predictions_current_memory.json", predictions)
    summary = {"sample_id": sample, "baseline": summarize(baseline), "raw_only": summarize(predictions), "index_verification": index_report}
    write_json(folder / "summary.json", summary)
    return summary


def aggregate(source: Path, output: Path, samples):
    rows = [row for sample in samples if (output / sample / "summary.json").exists()
            for row in read_json(output / sample / "details.json")]
    baseline = summarize([prediction(row, row["baseline_answer"], row["baseline_retrieved_memories"]) for row in rows])
    raw = summarize([prediction(row, row["current_memory_answer"], row["retrieved_memories"]) for row in rows])
    stages = {}
    for arm in ("baseline", "raw_only"):
        stats = {}
        for row in rows:
            gold = row["gold_evidence_turn_ids"]
            if not gold:
                continue
            for name, evidence in row[f"{arm}_stage_evidence"].items():
                count = stats.setdefault(name, Counter())
                count["scorable_questions"] += 1
                count["gold_turn_pairs"] += len(gold)
                for kind in ("source", "direct"):
                    hits = len(evidence[f"{kind}_gold_turns"])
                    count[f"{kind}_hit_questions"] += hits > 0
                    count[f"{kind}_complete_questions"] += hits == len(gold)
                    count[f"{kind}_hit_turn_pairs"] += hits
        stages[arm] = {name: dict(value) for name, value in stats.items()}
    prior = read_json(source / "batch-summary.json")
    result = {
        "question_count": len(rows), "baseline": baseline, "raw_only": raw,
        "delta_f1": raw["overall_f1"] - baseline["overall_f1"],
        "historical_memory": prior["historical_memory"], "oracle": prior["oracle"],
        "improved_questions": sum(row["memory_f1"] > row["baseline_f1"] for row in rows),
        "regressed_questions": sum(row["memory_f1"] < row["baseline_f1"] for row in rows),
        "unchanged_questions": sum(row["memory_f1"] == row["baseline_f1"] for row in rows),
        "answer_calls": sum(not row["answer_reused_for_identical_context"] for row in rows),
        "reused_answers": sum(row["answer_reused_for_identical_context"] for row in rows),
        "recorded_answer_tokens": sum(row["current_memory_token_cost"] or 0 for row in rows),
        "baseline_context_characters": sum(len(row["baseline_retrieved_memories"]) for row in rows),
        "raw_context_characters": sum(len(row["retrieved_memories"]) for row in rows),
        "stop_reasons": dict(Counter(row["answer_stop_reason"] for row in rows)),
        "final_item_count": dict(Counter(row["retrieved_count"] for row in rows)), "stage_evidence": stages,
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
        raise ValueError("Output must be separate from source")
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1 to 4")
    manifest = prepare(source)
    if args.validate_only:
        print(manifest["preflight"])
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Code or source inputs changed since this output was created")
    write_json(manifest_path, manifest)
    for name, expected in manifest["code_sha256"].items():
        content = (ROOT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Code changed during preparation")
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
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
                summary = aggregate(source, output, manifest["sample_ids"])
                print(f"DONE {sample}: {summary['question_count']} questions delta={summary['delta_f1']:+.3f}", flush=True)
            except Exception as error:
                errors.append({"sample_id": sample, "error_type": type(error).__name__, "error": str(error)[:500]})
                print(f"FAILED {sample}: {type(error).__name__}: {str(error)[:250]}", flush=True)
                write_json(output / "errors.json", errors)
    aggregate(source, output, manifest["sample_ids"])
    changed = [name for name, expected in manifest["frozen_source_sha256"].items() if digest(source / name) != expected]
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
