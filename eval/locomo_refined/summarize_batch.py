from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.locomo_refined.evaluate import summarize


MEMORY_ITEM_PATTERN = re.compile(r"(?m)^- \[(fact|episode|chat):")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator * 100


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple LoCoMo-Refined sample runs into one report"
    )
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to <batch_dir>/batch-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Defaults to <batch_dir>/report.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_dir = args.batch_dir.resolve()
    run_dirs = sorted(
        path
        for path in batch_dir.iterdir()
        if path.is_dir()
        and (path / "summary.json").is_file()
        and (path / "diagnostics.json").is_file()
    )
    if not run_dirs:
        raise FileNotFoundError(f"no completed runs found under {batch_dir}")

    memory_predictions: list[dict[str, Any]] = []
    no_memory_predictions: list[dict[str, Any]] = []
    oracle_predictions: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    route_source_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    diagnosis_counts: Counter[str] = Counter()
    item_kind_counts: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []
    scorable_count = 0
    evidence_hits = 0
    evidence_recall_sum = 0.0
    total_current_tokens = 0
    total_no_memory_tokens = 0
    total_oracle_tokens = 0
    total_exchanges = 0
    total_batches = 0
    total_retries = 0
    total_facts = 0
    total_episodes = 0
    total_evidence = 0

    expected_config: dict[str, Any] | None = None
    for run_dir in run_dirs:
        summary = load_json(run_dir / "summary.json")
        diagnostics = load_json(run_dir / "diagnostics.json")
        details = load_json(run_dir / "details.json")
        current = load_json(run_dir / "predictions_current_memory.json")
        baseline = load_json(run_dir / "predictions_no_memory.json")
        oracle = load_json(run_dir / "predictions_oracle_evidence.json")

        config = dict(summary.get("config") or {})
        if expected_config is None:
            expected_config = config
        elif config != expected_config:
            raise ValueError(f"configuration mismatch in {run_dir}")

        if not (len(current) == len(baseline) == len(oracle) == len(details)):
            raise ValueError(f"prediction count mismatch in {run_dir}")

        memory_predictions.extend(current)
        no_memory_predictions.extend(baseline)
        oracle_predictions.extend(oracle)
        route_counts.update(summary.get("retrieval_route_counts") or {})
        route_source_counts.update(summary.get("retrieval_route_source_counts") or {})
        strategy_counts.update(summary.get("retrieval_strategy_counts") or {})
        diagnosis_counts.update(diagnostics.get("diagnosis_counts") or {})

        for detail in details:
            total_current_tokens += int(detail.get("current_memory_token_cost") or 0)
            total_no_memory_tokens += int(detail.get("no_memory_token_cost") or 0)
            item_kind_counts.update(
                match.group(1) for match in MEMORY_ITEM_PATTERN.finditer(
                    str(detail.get("retrieved_memories") or "")
                )
            )
        total_oracle_tokens += sum(int(item.get("token_cost") or 0) for item in oracle)

        for item in diagnostics.get("details") or []:
            recall = item.get("evidence_recall_at_k")
            if recall is None:
                continue
            scorable_count += 1
            evidence_hits += int(bool(item.get("evidence_hit_at_k")))
            evidence_recall_sum += float(recall)

        status = summary.get("memory_status") or {}
        total_exchanges += int(status.get("exchange_count") or 0)
        total_batches += int(status.get("batch_count") or 0)
        total_retries += int(status.get("consolidation_retry_count") or 0)
        total_facts += int(status.get("facts") or 0)
        total_episodes += int(status.get("episodes") or 0)
        total_evidence += int(status.get("evidence") or 0)

        memory_f1 = float(summary["current_memory"]["overall_f1"])
        baseline_f1 = float(summary["no_memory"]["overall_f1"])
        oracle_f1 = float(diagnostics["oracle_evidence"]["overall_f1"])
        per_sample.append(
            {
                "sample_id": summary["sample_id"],
                "question_count": int(summary["question_count"]),
                "memory_f1": memory_f1,
                "no_memory_f1": baseline_f1,
                "memory_gain_f1": memory_f1 - baseline_f1,
                "oracle_f1": oracle_f1,
                "oracle_ratio_percent": percent(memory_f1, oracle_f1),
                "evidence_hit_at_k": diagnostics["evidence_retrieval"]["hit_at_k"],
                "evidence_recall_at_k": diagnostics["evidence_retrieval"]["recall_at_k"],
            }
        )

    memory_summary = summarize(memory_predictions)
    baseline_summary = summarize(no_memory_predictions)
    oracle_summary = summarize(oracle_predictions)
    memory_f1 = float(memory_summary["overall_f1"])
    baseline_f1 = float(baseline_summary["overall_f1"])
    oracle_f1 = float(oracle_summary["overall_f1"])
    oracle_ratio = percent(memory_f1, oracle_f1)
    oracle_adjusted_ratio = percent(memory_f1 - baseline_f1, oracle_f1 - baseline_f1)

    result = {
        "batch_id": batch_dir.name,
        "run_count": len(run_dirs),
        "question_count": len(memory_predictions),
        "config": expected_config or {},
        "memory": memory_summary,
        "no_memory": baseline_summary,
        "oracle_evidence": oracle_summary,
        "memory_gain_f1": memory_f1 - baseline_f1,
        "oracle_ratio_percent": oracle_ratio,
        "oracle_adjusted_ratio_percent": oracle_adjusted_ratio,
        "evidence_retrieval": {
            "k": 5,
            "scorable_question_count": scorable_count,
            "hit_at_k": percent(evidence_hits, scorable_count),
            "recall_at_k": (
                evidence_recall_sum / scorable_count * 100 if scorable_count else None
            ),
        },
        "retrieval_route_counts": dict(route_counts),
        "retrieval_route_source_counts": dict(route_source_counts),
        "retrieval_strategy_counts": dict(strategy_counts),
        "retrieved_item_kind_counts": dict(item_kind_counts),
        "diagnosis_counts": dict(diagnosis_counts),
        "memory_totals": {
            "exchanges": total_exchanges,
            "consolidation_batches": total_batches,
            "consolidation_retries": total_retries,
            "facts": total_facts,
            "episodes": total_episodes,
            "evidence": total_evidence,
        },
        "answer_token_totals": {
            "memory": total_current_tokens,
            "no_memory": total_no_memory_tokens,
            "oracle": total_oracle_tokens,
            "combined": total_current_tokens + total_no_memory_tokens + total_oracle_tokens,
        },
        "per_sample": per_sample,
    }

    output = (args.output or batch_dir / "batch-summary.json").resolve()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    category_rows = []
    for category, values in memory_summary["categories"].items():
        baseline_values = baseline_summary["categories"].get(category, {})
        oracle_values = oracle_summary["categories"].get(category, {})
        category_rows.append(
            "| {category} | {count} | {memory:.2f} | {baseline:.2f} | {oracle:.2f} | {ratio}% |".format(
                category=category,
                count=values["count"],
                memory=float(values["f1"]),
                baseline=float(baseline_values.get("f1", 0.0)),
                oracle=float(oracle_values.get("f1", 0.0)),
                ratio=fmt(percent(float(values["f1"]), float(oracle_values.get("f1", 0.0)))),
            )
        )
    sample_rows = [
        "| {sample_id} | {memory_f1:.2f} | {no_memory_f1:.2f} | {oracle_f1:.2f} | {oracle_ratio}% | {hit}% |".format(
            **item,
            oracle_ratio=fmt(item["oracle_ratio_percent"]),
            hit=fmt(item["evidence_hit_at_k"]),
        )
        for item in per_sample
    ]
    report = "\n".join(
        [
            "# LoCoMo-Refined 200-question report",
            "",
            f"- Runs: {len(run_dirs)}",
            f"- Questions: {len(memory_predictions)}",
            f"- Memory F1: {memory_f1:.2f}",
            f"- No-Memory F1: {baseline_f1:.2f}",
            f"- Memory Gain F1: {memory_f1 - baseline_f1:.2f}",
            f"- Oracle Evidence F1: {oracle_f1:.2f}",
            f"- Oracle ratio: {fmt(oracle_ratio)}%",
            f"- No-Memory-adjusted Oracle ratio: {fmt(oracle_adjusted_ratio)}%",
            f"- Evidence Hit@5: {fmt(result['evidence_retrieval']['hit_at_k'])}%",
            f"- Evidence Recall@5: {fmt(result['evidence_retrieval']['recall_at_k'])}%",
            "",
            "## Category F1",
            "",
            "| Category | Questions | Memory | No-Memory | Oracle | Oracle ratio |",
            "|---|---:|---:|---:|---:|---:|",
            *category_rows,
            "",
            "## Per conversation",
            "",
            "| Sample | Memory F1 | No-Memory F1 | Oracle F1 | Oracle ratio | Hit@5 |",
            "|---|---:|---:|---:|---:|---:|",
            *sample_rows,
            "",
            "## Diagnostics",
            "",
            f"- Routes: `{json.dumps(dict(route_counts), ensure_ascii=False)}`",
            f"- Route sources: `{json.dumps(dict(route_source_counts), ensure_ascii=False)}`",
            f"- Retrieved item kinds: `{json.dumps(dict(item_kind_counts), ensure_ascii=False)}`",
            f"- Diagnosis counts: `{json.dumps(dict(diagnosis_counts), ensure_ascii=False)}`",
            f"- Memory totals: `{json.dumps(result['memory_totals'], ensure_ascii=False)}`",
            f"- Answer tokens: `{json.dumps(result['answer_token_totals'], ensure_ascii=False)}`",
            "",
        ]
    )
    report_path = (args.report or batch_dir / "report.md").resolve()
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Summary: {output}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
