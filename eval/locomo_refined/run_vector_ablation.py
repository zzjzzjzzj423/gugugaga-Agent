"""Replay frozen LoCoMo questions with only historical vector coverage changed.

Source databases are opened read-only and copied with SQLite backup. The original
router decisions and query vectors are shared by both arms. No consolidation,
usage aggregation, no-memory answering, or oracle answering is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from eval.locomo_refined.evaluate import summarize, token_f1
from eval.locomo_refined.run_smoke import ANSWER_SYSTEM, answer
from gugugaga.config import Settings
from gugugaga.memory.service import MemoryService
from gugugaga.provider import SiliconFlowProvider


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def snapshot(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".snapshot.db")
    with closing(connect_readonly(source)) as origin, closing(sqlite3.connect(temporary)) as destination:
        origin.backup(destination)
    temporary.replace(target)


def coverage(path: Path, model: str) -> dict[str, Any]:
    with closing(connect_readonly(path)) as db:
        total, exchanges = db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT turn_id) FROM chat_log WHERE is_final=1"
        ).fetchone()
        rows = list(db.execute(
            "SELECT c.id, c.turn_id FROM chat_log c JOIN memory_embeddings e "
            "ON e.memory_key='chat:'||c.id WHERE c.is_final=1 AND e.model=?", (model,)
        ))
        return {
            "chat_rows": total, "exchanges": exchanges,
            "chat_vectors": len(rows),
            "vector_covered_exchanges": len({row[1] for row in rows}),
            "index_jobs": dict(db.execute(
                "SELECT status, COUNT(*) FROM memory_index_outbox GROUP BY status"
            )),
            "memory_counts": {
                table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("facts", "episodes", "chat_log", "consolidation_batches")
            },
        }


class CachedQueryProvider:
    def __init__(self, provider: SiliconFlowProvider, queries: dict[str, Any]):
        self.provider = provider
        self.queries = queries
        self.last_response: Any = None

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if len(texts) == 1 and texts[0] in self.queries:
            return [self.queries[texts[0]]]
        return self.provider.embed(texts, model=model)

    def create(self, **kwargs: Any) -> Any:
        self.last_response = self.provider.create(**kwargs)
        return self.last_response


class FrozenRouteService(MemoryService):
    def __init__(self, *args: Any, frozen_routes: dict[str, Any], **kwargs: Any):
        self.frozen_routes = frozen_routes
        super().__init__(*args, **kwargs)

    def _resolve_memory_intent(self, query: str, *, fallback_route: str):
        original = self.frozen_routes[query]
        return (
            original["gate_decision"] == "retrieve",
            original["retrieval_route"], original["retrieval_route_source"],
            original.get("retrieval_route_confidence"),
        )


def prediction(row: dict[str, Any], value: str, content: str, cost: Any = None):
    return {
        **{key: row[key] for key in ("qa_id", "sample_id", "question", "category", "gold_answer")},
        "predicted_answer": value, "retrieved_memories": content, "token_cost": cost,
    }


def score(row: dict[str, Any], text: str) -> float:
    gold = row["gold_answer"]
    return 100 * max(token_f1(text, item) for item in (gold if isinstance(gold, list) else [gold]))


def stage_evidence(trace: dict[str, Any], gold: set[str]) -> dict[str, Any]:
    result = {}
    for name, stage in trace.get("stages", {}).items():
        items = stage.get("candidates", [])
        all_turns = {turn for item in items for turn in item.get("source_turn_ids", [])}
        direct = {turn for item in items if item.get("kind") == "chat"
                  for turn in item.get("source_turn_ids", [])}
        result[name] = {
            "source_gold_turns": sorted(gold & all_turns),
            "direct_gold_turns": sorted(gold & direct),
        }
    final = trace.get("final", {})
    content = final.get("content", "")
    items = final.get("candidates", [])
    # Full raw text must actually survive rendering to count as complete evidence.
    direct = {turn for item in items if item.get("kind") == "chat"
              and item.get("text") and item["text"] in content
              for turn in item.get("source_turn_ids", [])}
    source = set()
    for line in content.splitlines():
        match = re.match(r"^- \[.*?; source=([^\]]+)\]", line)
        if match:
            source.update(match[1].split(","))
    result["injected"] = {
        "source_gold_turns": sorted(gold & source),
        "direct_gold_turns": sorted(gold & direct),
    }
    return result


def run_sample(source: Path, output: Path, settings: Settings, sample_id: str,
               reuse_run: Path | None = None) -> dict[str, Any]:
    folder, target = source / sample_id, output / sample_id
    target.mkdir(parents=True, exist_ok=True)
    if (target / "summary.json").exists():
        return read_json(target / "summary.json")
    details = read_json(folder / "details.json")
    original_summary = read_json(folder / "summary.json")
    config = original_summary["config"]
    model = config["answer_model"]
    source_db = folder / "databases" / f"{sample_id}.db"
    with closing(connect_readonly(source_db)) as db:
        models = [row[0] for row in db.execute("SELECT DISTINCT model FROM memory_embeddings")]
    if len(models) != 1:
        raise ValueError(f"{sample_id}: expected one frozen embedding model")
    embedding_model = models[0]
    before = coverage(source_db, embedding_model)
    source_hash = digest(source_db)
    a_db, b_db = target / "databases" / "baseline.db", target / "databases" / "full_vectors.db"
    snapshot(source_db, a_db)
    reusable = reuse_run / sample_id if reuse_run else None
    cached_db = reusable / "databases" / "full_vectors.db" if reusable else None
    snapshot(cached_db if cached_db and cached_db.exists() else source_db, b_db)
    raw_provider = SiliconFlowProvider(settings, enable_thinking=False, temperature=0.0)
    queries_path = target / "query_vectors.json"
    if queries_path.exists():
        queries = read_json(queries_path)
    elif reusable and (reusable / "query_vectors.json").exists():
        queries = read_json(reusable / "query_vectors.json")
        write_json(queries_path, queries)
    else:
        texts = [row["question"].strip() for row in details]
        vectors = raw_provider.embed(texts, model=embedding_model)
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise ValueError("query embedding count mismatch")
        queries = dict(zip(texts, vectors))
        write_json(queries_path, queries)
    provider = CachedQueryProvider(raw_provider, queries)
    answer_cache = {}
    if reusable and (reusable / "details.json").exists():
        for prior in read_json(reusable / "details.json"):
            answer_cache[(prior["question"], prior["retrieved_memories"])] = (
                prior["current_memory_answer"], prior.get("answer_stop_reason")
            )
    options = dict(
        provider=provider, frozen_routes={row["question"].strip(): row for row in details},
        model=model, consolidation_enabled=False, start_worker=False,
        embedding_model=embedding_model, recall_token_budget=2000,
        retrieval_candidate_limit=20, retrieval_final_limit=5, retrieval_min_score=0.20,
    )
    diagnostics = {row["qa_id"]: row for row in read_json(folder / "diagnostics.json")["details"]}
    checkpoint = target / "details.json"
    completed = read_json(checkpoint) if checkpoint.exists() else []
    completed_ids = {row["qa_id"] for row in completed}
    a_service = FrozenRouteService(a_db, evidence_hot_exchanges=30, **options)
    b_service = FrozenRouteService(b_db, evidence_hot_exchanges=10000, **options)
    # Warming rewrites FTS row IDs, which can change BM25 tie ordering. Hold the
    # lexical arm fixed as well as routing, so only vector coverage is varied.
    def frozen_bm25(query: str, *, limit: int):
        return deepcopy(a_service.repository.bm25_candidates(query, limit=limit))
    b_service.repository.bm25_candidates = frozen_bm25
    try:
        progress_path = target / "progress.json"
        write_json(progress_path, {"phase": "indexing", "before": before})
        while b_service.process_index_pending(max_jobs=32):
            current = coverage(b_db, embedding_model)
            write_json(progress_path, {"phase": "indexing", "coverage": current})
            print(f"{sample_id} vectors {current['chat_vectors']}/{current['chat_rows']}", flush=True)
        after = coverage(b_db, embedding_model)
        if after["chat_vectors"] != after["chat_rows"]:
            raise RuntimeError(f"{sample_id}: incomplete vector index: {after}")
        if any(count for status, count in after["index_jobs"].items() if status != "completed"):
            raise RuntimeError(f"{sample_id}: unfinished index jobs: {after['index_jobs']}")
        if after["memory_counts"] != before["memory_counts"]:
            raise RuntimeError("memory records changed during vector-only preparation")
        for row in details:
            if row["qa_id"] in completed_ids:
                continue
            trace_a, trace_b = {}, {}
            started = time.monotonic()
            a = a_service.recall_for_turn(row["question"], trace=trace_a)
            a_ms = (time.monotonic() - started) * 1000
            started = time.monotonic()
            b = b_service.recall_for_turn(row["question"], trace=trace_b)
            b_ms = (time.monotonic() - started) * 1000
            if a.route != row["retrieval_route"] or b.route != a.route:
                raise RuntimeError("route replay mismatch")
            if a.strategy != "hybrid" or b.strategy != "hybrid":
                raise RuntimeError("hybrid retrieval unexpectedly failed")
            if trace_a["stages"]["bm25"]["candidates"] != trace_b["stages"]["bm25"]["candidates"]:
                raise RuntimeError("BM25 candidates differ between arms")
            trace_path = target / "traces" / (row["qa_id"].replace("#", "_") + ".json")
            write_json(trace_path, {"qa_id": row["qa_id"], "baseline": trace_a, "full_vectors": trace_b})
            matches = a.content == row["retrieved_memories"]
            a_cost, a_stop = None, "reused_frozen_prediction"
            if matches:
                a_answer = row["current_memory_answer"]
            else:
                a_answer, a_cost = answer(provider, row["question"], a.content, model=model)
                a_stop = provider.last_response.stop_reason
            b_reused = (row["question"], b.content) in answer_cache
            if b_reused:
                b_answer, b_stop = answer_cache[(row["question"], b.content)]
                b_cost = None
            else:
                b_answer, b_cost = answer(provider, row["question"], b.content, model=model)
                b_stop = provider.last_response.stop_reason
            gold = set(diagnostics[row["qa_id"]]["gold_evidence_turn_ids"])
            completed.append({
                **row,
                "historical_memory_answer": row["current_memory_answer"],
                "historical_memory_f1": score(row, row["current_memory_answer"]),
                "baseline_context_matches_frozen": matches,
                "baseline_answer": a_answer, "baseline_f1": score(row, a_answer),
                "baseline_token_cost": a_cost, "baseline_stop_reason": a_stop,
                "baseline_retrieved_memories": a.content,
                "current_memory_answer": b_answer, "memory_f1": score(row, b_answer),
                "current_memory_token_cost": b_cost,
                "answer_stop_reason": b_stop, "answer_reused_from_preliminary_run": b_reused,
                "retrieved_memories": b.content, "retrieved_count": b.hit_count,
                "retrieved_kinds": list(b.kinds), "retrieval_method": b.strategy,
                "gold_evidence_turn_ids": sorted(gold),
                "baseline_stage_evidence": stage_evidence(trace_a, gold),
                "full_vector_stage_evidence": stage_evidence(trace_b, gold),
                "baseline_retrieval_ms": a_ms, "full_vector_retrieval_ms": b_ms,
            })
            write_json(checkpoint, completed)
            write_json(progress_path, {"phase": "answering", "completed": len(completed), "total": len(details)})
            print(f"{sample_id} {len(completed)}/{len(details)} A={score(row,a_answer):.1f} B={score(row,b_answer):.1f}", flush=True)
    finally:
        a_service.close()
        b_service.close()
    if digest(source_db) != source_hash:
        raise RuntimeError("source database hash changed")
    predictions = [prediction(row, row["current_memory_answer"], row["retrieved_memories"], row["current_memory_token_cost"]) for row in completed]
    baseline = [prediction(row, row["baseline_answer"], row["baseline_retrieved_memories"], row["baseline_token_cost"]) for row in completed]
    write_json(target / "predictions_current_memory.json", predictions)
    write_json(target / "predictions_baseline_replay.json", baseline)
    summary = {
        "sample_id": sample_id, "question_count": len(completed),
        "source_database_sha256": source_hash, "source_database_unchanged": True,
        "before": before, "after": after,
        "baseline_context_matches_frozen": sum(row["baseline_context_matches_frozen"] for row in completed),
        "historical_memory": original_summary["current_memory"],
        "baseline_replay": summarize(baseline), "full_vectors": summarize(predictions),
        "no_memory": summarize(read_json(folder / "predictions_no_memory.json")),
        "oracle": summarize(read_json(folder / "predictions_oracle_evidence.json")),
    }
    write_json(target / "summary.json", summary)
    write_json(target / "progress.json", {"phase": "complete", "completed": len(completed)})
    return summary


def aggregate(source: Path, output: Path, sample_ids: list[str]) -> dict[str, Any]:
    all_details, full, baseline, historical, no_memory, oracle, samples = [], [], [], [], [], [], []
    for sample in sample_ids:
        folder = output / sample
        if not (folder / "summary.json").exists():
            continue
        samples.append(read_json(folder / "summary.json"))
        all_details.extend(read_json(folder / "details.json"))
        full.extend(read_json(folder / "predictions_current_memory.json"))
        baseline.extend(read_json(folder / "predictions_baseline_replay.json"))
        for dest, name in ((historical, "current_memory"), (no_memory, "no_memory"), (oracle, "oracle_evidence")):
            dest.extend(read_json(source / sample / f"predictions_{name}.json"))
    stage_stats = {}
    for arm in ("baseline", "full_vector"):
        counters: dict[str, Counter] = {}
        for row in all_details:
            gold = row["gold_evidence_turn_ids"]
            if not gold:
                continue
            for stage, evidence in row[f"{arm}_stage_evidence"].items():
                count = counters.setdefault(stage, Counter())
                count["questions"] += 1
                count["gold_turn_pairs"] += len(gold)
                for kind in ("source", "direct"):
                    hits = len(evidence[f"{kind}_gold_turns"])
                    count[f"{kind}_hit_questions"] += hits > 0
                    count[f"{kind}_complete_questions"] += hits == len(gold)
                    count[f"{kind}_hit_turn_pairs"] += hits
        stage_stats[arm] = {stage: dict(value) for stage, value in counters.items()}
    result = {
        "completed_samples": len(samples), "planned_samples": len(sample_ids),
        "question_count": len(all_details),
        "historical_memory": summarize(historical), "baseline_replay": summarize(baseline),
        "full_vectors": summarize(full), "no_memory": summarize(no_memory), "oracle": summarize(oracle),
        "baseline_context_matches_frozen": sum(row["baseline_context_matches_frozen"] for row in all_details),
        "stage_evidence": stage_stats, "samples": samples,
        "answer_stop_reasons": dict(Counter(row["answer_stop_reason"] for row in all_details)),
        "improved_questions": sum(row["memory_f1"] > row["baseline_f1"] for row in all_details),
        "regressed_questions": sum(row["memory_f1"] < row["baseline_f1"] for row in all_details),
        "unchanged_questions": sum(row["memory_f1"] == row["baseline_f1"] for row in all_details),
    }
    result["delta_f1"] = result["full_vectors"]["overall_f1"] - result["baseline_replay"]["overall_f1"]
    write_json(output / "batch-summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--reuse-run", type=Path, help="Reuse compatible embeddings and identical-context answers")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from the frozen source run")
    sample_ids = args.sample_id or sorted(folder.name for folder in source.glob("conv-*") if folder.is_dir())
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    for sample in sample_ids:
        rows = read_json(source / sample / "details.json")
        ids = [row["qa_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate question IDs")
        for branch in ("current_memory", "no_memory", "oracle_evidence"):
            if [row["qa_id"] for row in read_json(source / sample / f"predictions_{branch}.json")] != ids:
                raise ValueError("frozen prediction IDs differ")
    manifest = {
        "schema_version": 2, "source_run": str(source), "sample_ids": sample_ids,
        "reuse_run": str(args.reuse_run.resolve()) if args.reuse_run else None,
        "frozen_source_sha256": {
            str(path.relative_to(source)): digest(path)
            for sample in sample_ids
            for path in [source / sample / "databases" / f"{sample}.db",
                         *sorted((source / sample).glob("*.json"))]
        },
        "code_sha256": {
            name: digest(ROOT / name)
            for name in ("eval/locomo_refined/run_vector_ablation.py", "eval/locomo_refined/run_smoke.py",
                         "gugugaga/memory/service.py", "gugugaga/memory/retrieval.py",
                         "gugugaga/memory/repository.py", "gugugaga/provider.py")
        },
        "protocol": {
            "answer_model": "Qwen/Qwen3.6-35B-A3B", "embedding_model": "BAAI/bge-m3",
            "temperature": 0, "enable_thinking": False, "answer_max_tokens": 256,
            "baseline_hot_exchanges": 30, "experimental_hot_exchanges": 10000,
            "candidate_limit": 20, "final_limit": 5, "recall_token_budget": 2000, "min_score": 0.20,
            "route_policy": "replay frozen route decisions without gold evidence",
            "baseline_answers": "reuse if actual context matches; otherwise regenerate",
            "query_vectors": "cached identically for both arms", "usage_aggregation": False,
            "bm25_policy": "reuse unchanged baseline BM25 candidates in both arms to avoid FTS reinsertion tie drift",
            "answer_system_sha256": hashlib.sha256(ANSWER_SYSTEM.encode()).hexdigest(),
        },
        "limitations": [
            "Frozen run did not record a code hash, recall token budget, or min score; the latter two use source defaults.",
            "Source coverage does not establish answer sufficiency; direct injected coverage requires full raw text.",
            "Original Token F1 is retained and is not accuracy. No semantic judge is added in this ablation.",
        ],
    }
    if args.validate_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("existing output uses a different protocol")
    write_json(manifest_path, manifest)
    if args.reuse_run:
        prior_manifest = read_json(args.reuse_run / "manifest.json")
        if prior_manifest["source_run"] != str(source):
            raise ValueError("cache source run differs")
        for key in ("answer_model", "embedding_model", "temperature", "enable_thinking", "answer_max_tokens", "answer_system_sha256"):
            if prior_manifest["protocol"][key] != manifest["protocol"][key]:
                raise ValueError(f"cache protocol differs: {key}")
    load_dotenv(ROOT / ".env")
    settings = Settings.from_env(ROOT, model_override=manifest["protocol"]["answer_model"])
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_sample, source, output, settings, sample, args.reuse_run): sample for sample in sample_ids}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                future.result()
                result = aggregate(source, output, sample_ids)
                print(f"DONE {sample}: {result['question_count']} questions; delta={result['delta_f1']:+.3f}", flush=True)
            except Exception as error:
                errors.append({"sample_id": sample, "error_type": type(error).__name__, "error": str(error)[:500]})
                print(f"FAILED {sample}: {type(error).__name__}: {str(error)[:300]}", flush=True)
                write_json(output / "errors.json", errors)
    aggregate(source, output, sample_ids)
    write_json(output / "run-status.json", {"status": "failed" if errors else "complete", "errors": errors, "finished_at": datetime.now(timezone.utc).isoformat()})
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
