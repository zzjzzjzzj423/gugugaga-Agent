from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from eval.locomo_refined.adapter import (
    GugugagaMemoryAdapter,
    conversation_exchanges,
    load_jsonl,
)
from eval.locomo_refined.evaluate import summarize
from gugugaga.config import Settings
from gugugaga.provider import SiliconFlowProvider


DEFAULT_DATA_DIR = ROOT / "data"
CATEGORY_ORDER = ("4", "1", "3", "2")
ANSWER_SYSTEM = """Answer the question using only the supplied memory.
Return the shortest phrase that fully answers the question. Do not repeat the question or explain your reasoning unless a short temporal calculation is necessary.
For questions that explicitly ask what is likely or what someone would choose, you may make one conservative inference from clear preferences, goals, or alternatives in memory; an exact matching sentence is not required.
If the memory does not support the answer or that one inference, answer exactly: No information."""


def response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content or "").strip()
    values = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if kind == "text":
            values.append(
                str(block.get("text", ""))
                if isinstance(block, dict)
                else str(getattr(block, "text", ""))
            )
    return "".join(values).strip()


def token_cost(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if not isinstance(usage, dict):
        return None
    values = [usage.get("input_tokens"), usage.get("output_tokens")]
    known = [int(value) for value in values if isinstance(value, int)]
    return sum(known) if known else None


def answer(provider: Any, question: str, memory: str, *, model: str | None) -> tuple[str, int | None]:
    memory_block = memory.strip() or "(none)"
    response = provider.create(
        messages=[
            {
                "role": "user",
                "content": f"Memory:\n{memory_block}\n\nQuestion:\n{question}",
            }
        ],
        system=ANSWER_SYSTEM,
        tools=[],
        max_tokens=256,
        model=model,
    )
    return response_text(response), token_cost(response)


def group_questions(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        grouped[str(question["sample_id"])].append(question)
    return grouped


def choose_sample(
    conversations: list[dict[str, Any]],
    questions_by_sample: dict[str, list[dict[str, Any]]],
    requested: str | None,
) -> dict[str, Any]:
    if requested:
        for conversation in conversations:
            if str(conversation.get("sample_id")) == requested:
                if requested not in questions_by_sample:
                    raise ValueError(f"sample {requested} has no questions")
                return conversation
        raise ValueError(f"sample {requested} was not found")
    for conversation in sorted(
        conversations, key=lambda item: int(item.get("conversation_idx", 0))
    ):
        if str(conversation.get("sample_id")) in questions_by_sample:
            return conversation
    raise ValueError("no conversation has matching questions")


def select_questions(
    questions: list[dict[str, Any]], *, per_category: int = 5, maximum: int = 20
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category in CATEGORY_ORDER:
        matches = [item for item in questions if str(item.get("category")) == category]
        for item in matches[:per_category]:
            selected.append(item)
            selected_ids.add(str(item["qa_id"]))
    if len(selected) < maximum:
        for item in questions:
            if str(item["qa_id"]) in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(str(item["qa_id"]))
            if len(selected) >= maximum:
                break
    return selected[:maximum]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Gugugaga LoCoMo-Refined smoke evaluation")
    parser.add_argument("--conversations", type=Path, default=DEFAULT_DATA_DIR / "conversations.jsonl")
    parser.add_argument("--questions", type=Path, default=DEFAULT_DATA_DIR / "questions.jsonl")
    parser.add_argument("--sample-id")
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--questions-per-category", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-output-dir",
        type=Path,
        help="Resume an interrupted run from its existing output directory",
    )
    parser.add_argument("--exclude-image-context", action="store_true")
    parser.add_argument("--model", help="Answer model override")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Disable provider thinking mode for short-answer and strict-JSON calls "
            "(recommended for dual-mode reasoning models)"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature shared by consolidation and answer calls",
    )
    parser.add_argument("--max-consolidation-attempts", type=int, default=3)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize the selected data without calling a model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir and args.resume_output_dir:
        raise ValueError("--output-dir and --resume-output-dir cannot be used together")
    if args.max_questions < 1 or args.questions_per_category < 1:
        raise ValueError("question limits must be positive")
    conversations = load_jsonl(args.conversations)
    questions = load_jsonl(args.questions)
    questions_by_sample = group_questions(questions)
    conversation = choose_sample(conversations, questions_by_sample, args.sample_id)
    sample_id = str(conversation["sample_id"])
    selected = select_questions(
        questions_by_sample[sample_id],
        per_category=args.questions_per_category,
        maximum=args.max_questions,
    )
    if args.validate_only:
        exchanges = list(
            conversation_exchanges(
                conversation,
                include_image_context=not args.exclude_image_context,
            )
        )
        report = {
            "sample_id": sample_id,
            "conversation_idx": conversation.get("conversation_idx"),
            "session_count": len(conversation.get("sessions") or []),
            "exchange_count": len(exchanges),
            "partial_exchange_count": sum(exchange.partial for exchange in exchanges),
            "selected_question_count": len(selected),
            "selected_categories": dict(
                sorted(
                    {
                        category: sum(
                            str(item.get("category")) == category for item in selected
                        )
                        for category in {str(item.get("category")) for item in selected}
                    }.items()
                )
            ),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    load_dotenv(REPO_ROOT / ".env")
    settings = Settings.from_env(REPO_ROOT, model_override=args.model)
    provider = SiliconFlowProvider(
        settings,
        enable_thinking=False if args.disable_thinking else None,
        temperature=args.temperature,
    )

    if args.resume_output_dir:
        output_dir = args.resume_output_dir.resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"resume output directory was not found: {output_dir}")
        run_id = output_dir.name
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (args.output_dir or ROOT / "runs" / run_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "databases" / f"{sample_id}.db"
    if args.resume_output_dir and not database.is_file():
        raise FileNotFoundError(f"resume database was not found: {database}")
    expected_turn_ids = {
        exchange.turn_id
        for exchange in conversation_exchanges(
            conversation,
            include_image_context=not args.exclude_image_context,
        )
    }
    adapter = GugugagaMemoryAdapter(
        database,
        provider,
        threshold=settings.memory_consolidation_exchange_threshold,
        model=settings.memory_consolidation_model,
        timeout_seconds=settings.memory_consolidation_timeout_seconds,
        lease_seconds=settings.memory_consolidation_lease_seconds,
        max_facts=settings.memory_consolidation_max_facts,
        min_importance=settings.memory_consolidation_min_importance,
        max_episodes=settings.memory_consolidation_max_episodes,
        episode_min_importance=(
            settings.memory_consolidation_episode_min_importance
        ),
        recall_token_budget=settings.memory_recall_token_budget,
        embedding_model=settings.memory_embedding_model,
        retrieval_candidate_limit=settings.memory_retrieval_candidate_limit,
        retrieval_final_limit=settings.memory_retrieval_final_limit,
        retrieval_min_score=settings.memory_retrieval_min_score,
        max_consolidation_attempts=args.max_consolidation_attempts,
        allow_existing=bool(args.resume_output_dir),
    )
    try:
        adapter.ingest_conversation(
            conversation,
            include_image_context=not args.exclude_image_context,
        )
        unexpected_turn_ids = adapter.recorded_turn_ids - expected_turn_ids
        if unexpected_turn_ids:
            preview = ", ".join(sorted(unexpected_turn_ids)[:3])
            raise RuntimeError(
                "resume database contains turns from another conversation: "
                f"{preview}"
            )
        adapter.flush()
        memory_status = adapter.status()

        current_predictions = []
        baseline_predictions = []
        details = []
        for qa in selected:
            question = str(qa["question"])
            recall = adapter.recall(question)
            current_answer, current_cost = answer(
                provider,
                question,
                recall.content if recall.should_inject else "",
                model=args.model or settings.model,
            )
            baseline_answer, baseline_cost = answer(
                provider,
                question,
                "",
                model=args.model or settings.model,
            )
            common = {
                "qa_id": qa["qa_id"],
                "sample_id": sample_id,
                "question": question,
                "category": str(qa["category"]),
                "gold_answer": qa["answer"],
            }
            current = {
                **common,
                "predicted_answer": current_answer,
                "token_cost": current_cost,
                "retrieved_memories": recall.content,
            }
            baseline = {
                **common,
                "predicted_answer": baseline_answer,
                "token_cost": baseline_cost,
                "retrieved_memories": "",
            }
            current_predictions.append(current)
            baseline_predictions.append(baseline)
            details.append(
                {
                    **common,
                    "gate_decision": recall.decision,
                    "gate_reason": recall.reason,
                    "retrieval_method": recall.strategy,
                    "retrieval_route": recall.route,
                    "retrieval_route_source": recall.route_source,
                    "retrieval_route_confidence": recall.route_confidence,
                    "retrieved_count": recall.hit_count,
                    "retrieved_kinds": list(recall.kinds),
                    "retrieved_memories": recall.content,
                    "current_memory_answer": current_answer,
                    "current_memory_token_cost": current_cost,
                    "no_memory_answer": baseline_answer,
                    "no_memory_token_cost": baseline_cost,
                }
            )
    finally:
        adapter.close()

    current_summary = summarize(current_predictions)
    baseline_summary = summarize(baseline_predictions)
    gate_open = sum(item["gate_decision"] == "retrieve" for item in details)
    empty_recall = sum(
        item["gate_decision"] == "retrieve" and item["retrieved_count"] == 0
        for item in details
    )
    question_count = len(details)
    retrieval_strategy_counts = {
        strategy: sum(item["retrieval_method"] == strategy for item in details)
        for strategy in sorted({str(item["retrieval_method"]) for item in details})
    }
    retrieval_route_counts = {
        route: sum(item["retrieval_route"] == route for item in details)
        for route in sorted({str(item["retrieval_route"]) for item in details})
    }
    retrieval_route_source_counts = {
        source: sum(item["retrieval_route_source"] == source for item in details)
        for source in sorted(
            {str(item["retrieval_route_source"]) for item in details}
        )
    }
    summary = {
        "run_id": run_id,
        "sample_id": sample_id,
        "conversation_idx": conversation.get("conversation_idx"),
        "config": {
            "answer_model": args.model or settings.model,
            "consolidation_model": settings.memory_consolidation_model,
            "temperature": args.temperature,
            "thinking_disabled": args.disable_thinking,
            "fact_min_importance": settings.memory_consolidation_min_importance,
            "episode_min_importance": (
                settings.memory_consolidation_episode_min_importance
            ),
            "max_episodes_per_batch": settings.memory_consolidation_max_episodes,
            "max_questions": args.max_questions,
            "questions_per_category": args.questions_per_category,
            "include_image_context": not args.exclude_image_context,
        },
        "retrieval_method": (
            "hybrid" if settings.memory_embedding_model else "bm25"
        ),
        "retrieval_strategy_counts": retrieval_strategy_counts,
        "retrieval_route_counts": retrieval_route_counts,
        "retrieval_route_source_counts": retrieval_route_source_counts,
        "question_count": question_count,
        "memory_status": memory_status,
        "gate": {
            "open_count": gate_open,
            "skip_count": question_count - gate_open,
            "open_rate": gate_open / question_count if question_count else 0.0,
            "empty_recall_count": empty_recall,
            "average_retrieved_count": (
                sum(item["retrieved_count"] for item in details) / question_count
                if question_count
                else 0.0
            ),
        },
        "current_memory": current_summary,
        "no_memory": baseline_summary,
        "memory_gain_f1": current_summary["overall_f1"] - baseline_summary["overall_f1"],
    }
    write_json(output_dir / "predictions_current_memory.json", current_predictions)
    write_json(output_dir / "predictions_no_memory.json", baseline_predictions)
    write_json(output_dir / "details.json", details)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
