from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable


CATEGORY_LABELS = {
    "1": "Multi-hop",
    "2": "Temporal",
    "3": "Open-domain",
    "4": "Single-hop",
}
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: Any) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]


def token_f1(prediction: Any, reference: Any) -> float:
    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted or not expected:
        return 0.0
    expected_counts = Counter(expected)
    overlap = sum(
        min(count, expected_counts[token])
        for token, count in Counter(predicted).items()
    )
    if overlap <= 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: Any, reference: Any) -> float:
    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted or not expected:
        return 0.0
    expected_counts = Counter(expected)
    overlap = sum(
        min(count, expected_counts[token])
        for token, count in Counter(predicted).items()
    )
    precision = overlap / len(predicted)
    if precision <= 0:
        return 0.0
    penalty = 1.0
    if len(predicted) < len(expected):
        penalty = math.exp(1.0 - len(expected) / len(predicted))
    return penalty * precision


def _answer_candidates(gold_answer: Any) -> list[str]:
    values = gold_answer if isinstance(gold_answer, list) else [gold_answer]
    candidates = [str(value).strip() for value in values]
    return [value for value in candidates if value] or [""]


def _best(metric: Callable[[Any, Any], float], prediction: Any, gold: Any) -> float:
    return max(metric(prediction, candidate) for candidate in _answer_candidates(gold))


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    category_scores: dict[str, list[tuple[float, float]]] = {}
    for item in predictions:
        category_scores.setdefault(str(item["category"]), []).append(
            (
                _best(token_f1, item["predicted_answer"], item["gold_answer"]),
                _best(bleu1, item["predicted_answer"], item["gold_answer"]),
            )
        )
    all_scores = [score for scores in category_scores.values() for score in scores]
    if not all_scores:
        return {"question_count": 0, "overall_f1": 0.0, "overall_bleu1": 0.0, "categories": {}}
    categories = {}
    for category, scores in sorted(category_scores.items()):
        categories[CATEGORY_LABELS.get(category, category)] = {
            "count": len(scores),
            "f1": sum(score[0] for score in scores) / len(scores) * 100,
            "bleu1": sum(score[1] for score in scores) / len(scores) * 100,
        }
    return {
        "question_count": len(all_scores),
        "overall_f1": sum(score[0] for score in all_scores) / len(all_scores) * 100,
        "overall_bleu1": sum(score[1] for score in all_scores) / len(all_scores) * 100,
        "categories": categories,
    }

