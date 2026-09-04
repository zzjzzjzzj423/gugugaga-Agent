from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summary_path(value: Path) -> Path:
    return value / "summary.json" if value.is_dir() else value


def load_summary(value: Path) -> dict[str, Any]:
    path = summary_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"summary was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    payload["_path"] = str(path.resolve())
    return payload


def sample_stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def ensure_comparable(summaries: list[dict[str, Any]]) -> None:
    first = summaries[0]
    required = ("sample_id", "question_count", "retrieval_method", "config")
    for key in required:
        if key not in first:
            raise ValueError(f"first summary is missing comparison field: {key}")
    expected = {key: first[key] for key in required}
    for summary in summaries[1:]:
        actual = {key: summary.get(key) for key in required}
        if actual != expected:
            raise ValueError(
                "run configuration mismatch:\n"
                f"expected {json.dumps(expected, ensure_ascii=False, sort_keys=True)}\n"
                f"actual   {json.dumps(actual, ensure_ascii=False, sort_keys=True)}\n"
                f"source   {summary['_path']}"
            )


def build_report(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(summaries) < 2:
        raise ValueError("at least two runs are required to compute a sample standard deviation")
    ensure_comparable(summaries)
    categories = sorted(
        summaries[0]["current_memory"].get("categories", {}).keys()
    )
    return {
        "run_count": len(summaries),
        "sample_id": summaries[0]["sample_id"],
        "question_count_per_run": summaries[0]["question_count"],
        "config": summaries[0]["config"],
        "runs": [summary["run_id"] for summary in summaries],
        "metrics": {
            "memory_f1": sample_stats(
                [float(summary["current_memory"]["overall_f1"]) for summary in summaries]
            ),
            "no_memory_f1": sample_stats(
                [float(summary["no_memory"]["overall_f1"]) for summary in summaries]
            ),
            "memory_gain_f1": sample_stats(
                [float(summary["memory_gain_f1"]) for summary in summaries]
            ),
        },
        "memory_f1_by_category": {
            category: sample_stats(
                [
                    float(summary["current_memory"]["categories"][category]["f1"])
                    for summary in summaries
                ]
            )
            for category in categories
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = [
        "| Metric | Mean | Sample SD | Values |",
        "|---|---:|---:|---|",
    ]
    labels = {
        "memory_f1": "Memory F1",
        "no_memory_f1": "No-Memory F1",
        "memory_gain_f1": "Memory Gain F1",
    }
    for key, label in labels.items():
        stats = report["metrics"][key]
        values = ", ".join(f"{value:.2f}" for value in stats["values"])
        rows.append(
            f"| {label} | {stats['mean']:.2f} | {stats['sample_std']:.2f} | {values} |"
        )
    for category, stats in report["memory_f1_by_category"].items():
        values = ", ".join(f"{value:.2f}" for value in stats["values"])
        rows.append(
            f"| {category} Memory F1 | {stats['mean']:.2f} | "
            f"{stats['sample_std']:.2f} | {values} |"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate comparable LoCoMo-Refined repeat runs"
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or summary files")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = build_report([load_summary(path) for path in args.runs])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
