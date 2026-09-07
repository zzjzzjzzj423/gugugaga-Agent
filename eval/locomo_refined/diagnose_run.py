from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from eval.locomo_refined.adapter import load_jsonl, message_content, normalize_timestamp
from eval.locomo_refined.evaluate import summarize
from eval.locomo_refined.run_smoke import answer
from gugugaga.config import Settings
from gugugaga.provider import SiliconFlowProvider


SOURCE_PATTERN = re.compile(r"; source=([^\]]+)\]")


def evidence_turn_id(sample_id: str, message: dict[str, Any]) -> str:
    session_index = int(message["session_index"])
    message_index = int(message["message_index"])
    if message_index < 1:
        raise ValueError("LoCoMo evidence message indexes must be one-based")
    exchange_index = (message_index - 1) // 2
    return f"{sample_id}:s{session_index:04d}:e{exchange_index:04d}"


def retrieved_source_turn_ids(rendered_memory: str) -> set[str]:
    values: set[str] = set()
    for match in SOURCE_PATTERN.finditer(rendered_memory):
        values.update(
            source.strip()
            for source in match.group(1).split(",")
            if source.strip()
        )
    return values


def session_timestamps(conversation: dict[str, Any]) -> dict[int, str]:
    return {
        int(session["session_index"]): normalize_timestamp(str(session["date_time"]))
        for session in conversation.get("sessions") or []
    }


def render_oracle_evidence(
    question: dict[str, Any],
    conversation: dict[str, Any],
    *,
    include_image_context: bool,
) -> str:
    timestamps = session_timestamps(conversation)
    lines = ["<oracle_evidence>"]
    for message in question.get("evidence_messages") or []:
        session_index = int(message["session_index"])
        timestamp = timestamps.get(session_index, "unknown time")
        speaker = str(message.get("speaker") or "unknown").strip() or "unknown"
        content = message_content(
            message,
            include_image_context=include_image_context,
        )
        lines.append(
            f"- [{message.get('dia_id')} @ {timestamp}] {speaker}: {content}"
        )
    lines.append("</oracle_evidence>")
    return "\n".join(lines)


def per_question_f1(prediction: str, question: dict[str, Any]) -> float:
    return float(
        summarize(
            [
                {
                    "category": str(question["category"]),
                    "predicted_answer": prediction,
                    "gold_answer": question["answer"],
                }
            ]
        )["overall_f1"]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add Oracle Evidence and Evidence Recall@K diagnostics to one run"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "questions.jsonl",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=ROOT / "data" / "conversations.jsonl",
    )
    parser.add_argument("--model", help="Override the answer model recorded by the run")
    parser.add_argument("--output", type=Path, help="Defaults to <run_dir>/diagnostics.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    details_path = run_dir / "details.json"
    if not summary_path.is_file() or not details_path.is_file():
        raise FileNotFoundError(f"run is missing summary.json or details.json: {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    details = json.loads(details_path.read_text(encoding="utf-8"))
    config = summary.get("config") or {}
    model = args.model or config.get("answer_model")
    if not model:
        raise ValueError("answer model is not recorded; pass --model explicitly")
    temperature = config.get("temperature")
    thinking_disabled = bool(config.get("thinking_disabled", False))
    include_image_context = bool(config.get("include_image_context", True))

    questions = {str(item["qa_id"]): item for item in load_jsonl(args.questions)}
    conversations = {
        str(item["sample_id"]): item for item in load_jsonl(args.conversations)
    }
    sample_id = str(summary["sample_id"])
    conversation = conversations.get(sample_id)
    if conversation is None:
        raise ValueError(f"conversation was not found for sample: {sample_id}")

    load_dotenv(REPO_ROOT / ".env")
    settings = Settings.from_env(REPO_ROOT, model_override=str(model))
    provider = SiliconFlowProvider(
        settings,
        enable_thinking=False if thinking_disabled else None,
        temperature=float(temperature) if temperature is not None else None,
    )

    oracle_predictions: list[dict[str, Any]] = []
    diagnostic_details: list[dict[str, Any]] = []
    for detail in details:
        qa_id = str(detail["qa_id"])
        question = questions.get(qa_id)
        if question is None:
            raise ValueError(f"question was not found: {qa_id}")
        gold_turns = {
            evidence_turn_id(sample_id, message)
            for message in question.get("evidence_messages") or []
        }
        retrieved_turns = retrieved_source_turn_ids(
            str(detail.get("retrieved_memories") or "")
        )
        matched_turns = gold_turns & retrieved_turns
        evidence_recall = len(matched_turns) / len(gold_turns) if gold_turns else None

        oracle_memory = render_oracle_evidence(
            question,
            conversation,
            include_image_context=include_image_context,
        )
        oracle_answer, oracle_tokens = answer(
            provider,
            str(question["question"]),
            oracle_memory,
            model=str(model),
        )
        oracle_prediction = {
            "qa_id": qa_id,
            "sample_id": sample_id,
            "question": question["question"],
            "category": str(question["category"]),
            "gold_answer": question["answer"],
            "predicted_answer": oracle_answer,
            "token_cost": oracle_tokens,
            "retrieved_memories": oracle_memory,
        }
        oracle_predictions.append(oracle_prediction)

        oracle_f1 = per_question_f1(oracle_answer, question)
        memory_f1 = per_question_f1(str(detail["current_memory_answer"]), question)
        if not gold_turns:
            diagnosis = "unscorable_no_gold_evidence"
        elif oracle_f1 <= 0:
            diagnosis = "oracle_answer_failure"
        elif not matched_turns:
            diagnosis = "retrieval_failure"
        elif memory_f1 <= 0:
            diagnosis = "answer_failure_with_evidence"
        else:
            diagnosis = "success_or_partial"
        diagnostic_details.append(
            {
                "qa_id": qa_id,
                "category": str(question["category"]),
                "gold_evidence_ids": list(question.get("evidence") or []),
                "gold_evidence_turn_ids": sorted(gold_turns),
                "retrieved_source_turn_ids": sorted(retrieved_turns),
                "matched_evidence_turn_ids": sorted(matched_turns),
                "evidence_hit_at_k": bool(matched_turns) if gold_turns else None,
                "evidence_recall_at_k": evidence_recall,
                "retrieval_route": detail.get("retrieval_route"),
                "retrieval_route_source": detail.get("retrieval_route_source"),
                "retrieval_route_confidence": detail.get(
                    "retrieval_route_confidence"
                ),
                "has_multimodal_evidence": any(
                    bool(message.get("has_multimodal_context"))
                    for message in question.get("evidence_messages") or []
                ),
                "memory_answer": detail["current_memory_answer"],
                "memory_f1": memory_f1,
                "oracle_answer": oracle_answer,
                "oracle_f1": oracle_f1,
                "diagnosis": diagnosis,
            }
        )

    oracle_summary = summarize(oracle_predictions)
    scorable_retrieval = [
        item for item in diagnostic_details if item["evidence_recall_at_k"] is not None
    ]
    k = max((int(item.get("retrieved_count") or 0) for item in details), default=0)
    diagnosis_counts = {
        diagnosis: sum(item["diagnosis"] == diagnosis for item in diagnostic_details)
        for diagnosis in sorted({item["diagnosis"] for item in diagnostic_details})
    }
    report = {
        "run_id": summary["run_id"],
        "sample_id": sample_id,
        "question_count": len(diagnostic_details),
        "config": {
            "answer_model": model,
            "temperature": temperature,
            "thinking_disabled": thinking_disabled,
            "include_image_context": include_image_context,
        },
        "memory": summary["current_memory"],
        "oracle_evidence": oracle_summary,
        "oracle_gain_over_memory_f1": (
            float(oracle_summary["overall_f1"])
            - float(summary["current_memory"]["overall_f1"])
        ),
        "evidence_retrieval": {
            "k": k,
            "scorable_question_count": len(scorable_retrieval),
            "hit_at_k": (
                sum(bool(item["evidence_hit_at_k"]) for item in scorable_retrieval)
                / len(scorable_retrieval)
                * 100
                if scorable_retrieval
                else None
            ),
            "recall_at_k": (
                sum(float(item["evidence_recall_at_k"]) for item in scorable_retrieval)
                / len(scorable_retrieval)
                * 100
                if scorable_retrieval
                else None
            ),
        },
        "diagnosis_counts": diagnosis_counts,
        "details": diagnostic_details,
    }
    output = (args.output or run_dir / "diagnostics.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "predictions_oracle_evidence.json").write_text(
        json.dumps(oracle_predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "run_id",
                    "sample_id",
                    "question_count",
                    "oracle_evidence",
                    "oracle_gain_over_memory_f1",
                    "evidence_retrieval",
                    "diagnosis_counts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nDiagnostics: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
