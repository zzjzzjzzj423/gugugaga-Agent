"""Replay final selection from frozen retrieval traces without model or DB calls."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from gugugaga.memory.models import RecallItem
from gugugaga.memory.retrieval import (
    render_candidates,
    select_routed_candidates,
    trace_candidates,
)


SelectionPolicy = Literal["routed", "no_type_quota"]


def replay_selection(
    source_trace: dict[str, Any], *, policy: SelectionPolicy = "routed",
    final_limit: int | None = None, recall_token_budget: int | None = None,
) -> dict[str, Any]:
    """Return a new trace using the saved rerank order and original render budget.

    With unchanged settings the routed policy must reproduce the saved final
    content exactly. Overrides change only final selection and rendering. The
    no_type_quota policy takes the first K already-reranked candidates; it leaves
    rerank deduplication and subject diversity intact. No gold evidence is used.
    """
    if policy not in {"routed", "no_type_quota"}:
        raise ValueError(f"unsupported selection policy: {policy}")
    trace = deepcopy(source_trace)
    config = trace["config"]
    original_limit = int(config["final_limit"])
    original_budget = int(config["recall_token_budget"])
    limit = original_limit if final_limit is None else final_limit
    token_budget = original_budget if recall_token_budget is None else recall_token_budget
    if type(limit) is not int or type(token_budget) is not int:
        raise ValueError("limit and token budget must be integers")
    if limit < 1 or token_budget < 1:
        raise ValueError("selection replay requires positive limit and token budget")
    if limit > int(config.get("candidate_limit", 20)):
        raise ValueError("final limit must not exceed candidate limit")
    config.update(final_limit=limit, recall_token_budget=token_budget)
    # A closed pre-gate has no rerank pool. Preserve the original skip instead
    # of inventing candidates or calling the router again.
    if "rerank" not in trace["stages"]:
        if trace["final"]["decision"] == "skip" and not trace["final"]["content"]:
            if trace.get("error") or trace.get("vector_error"):
                raise ValueError("cannot replay a failed retrieval")
            return trace
        raise ValueError("frozen trace is missing rerank candidates")
    ranked = trace["stages"]["rerank"]["candidates"]
    route = trace.get("gate", {}).get("route", trace["final"]["route"])
    route_trace: dict[str, Any] = {"selection_policy": policy}
    if policy == "routed":
        selected = select_routed_candidates(
            ranked, route=route, limit=limit, trace=route_trace
        )
    else:
        selected = ranked[:limit]
        route_trace.update({
            "route": route,
            "limit": limit,
            "quotas": {},
            "selections": [
                {"memory_key": item["memory_key"], "reason": "rerank_order"}
                for item in selected
            ],
            "candidates": trace_candidates(selected),
            "removed": [
                {"candidate": item, "reason": "final_limit"}
                for item in trace_candidates(ranked[limit:], start_rank=limit + 1)
            ],
        })
    trace["selection_policy"] = policy
    trace["stages"]["route"] = route_trace

    character_budget = token_budget * 4
    rendered = render_candidates(selected)
    render_trace: dict[str, Any] = {
        "character_budget": character_budget,
        "before_characters": len(rendered),
        "removed": [],
    }
    while len(selected) > 1 and len(rendered) > character_budget:
        removed = selected.pop()
        render_trace["removed"].append({
            "candidate": trace_candidates([removed], start_rank=len(selected) + 1)[0],
            "reason": "character_budget",
        })
        rendered = render_candidates(selected)
    render_trace.update({
        "candidates": trace_candidates(selected),
        "before_slice_characters": len(rendered),
        "truncated": len(rendered) > character_budget,
        "truncated_characters": max(0, len(rendered) - character_budget),
        "content": rendered[:character_budget],
    })
    trace["stages"]["render"] = render_trace
    content = render_trace["content"]
    kind_names = {"fact": "semantic", "episode": "episodic", "chat": "evidence"}
    trace["final"].update({
        "decision": "retrieve" if selected else "skip",
        "reason": source_trace["final"]["reason"] if selected else "no_relevant_memory",
        "hit_count": len(selected),
        "kinds": list(dict.fromkeys(
            kind_names[item["kind"]] for item in selected if item.get("kind") in kind_names
        )),
        "memory_keys": [str(item["memory_key"]) for item in selected],
        "items": [
            RecallItem(
                memory_key=str(item["memory_key"]), kind=str(item["kind"]),
                subject=str(item.get("subject") or "")[:500],
                text=str(item.get("text") or "")[:4_000],
                occurred_at=str(item["occurred_at"]) if item.get("occurred_at") is not None else None,
                retrieval_sources=tuple(str(value) for value in item.get("retrieval_sources") or ()),
                source_ranks={str(key): int(value) for key, value in (item.get("source_ranks") or {}).items()},
                relevance_score=float(item.get("relevance_score") or 0.0),
                final_score=float(item.get("final_score") or 0.0),
            ).as_dict()
            for item in selected
        ],
        "candidates": trace_candidates(selected),
        "content": content,
    })
    if (policy == "routed" and limit == original_limit and token_budget == original_budget
            and content != source_trace["final"]["content"]):
        raise ValueError("routed selection does not reproduce the frozen final content")
    return trace
