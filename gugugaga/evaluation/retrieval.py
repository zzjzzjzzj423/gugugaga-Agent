"""Configurable retrieval over an experiment's private memory repository."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable

from gugugaga.memory.retrieval import (
    _cosine, _jaccard, _recency_score, classify_memory_query, render_candidates,
    rerank_candidates, rrf_fuse, select_routed_candidates, trace_candidates,
)
from gugugaga.memory.service import _DIRECT_MEMORY_REFERENCE, _TRIVIAL_QUERY


def _rank(values: list[dict], arm: dict, now: datetime) -> tuple[list[dict], dict]:
    trace = {"removed": [], "deferred": [], "enabled": arm["rerank_enabled"],
             "dedup_enabled": arm["dedup_enabled"], "diversity_enabled": arm["diversity_enabled"]}
    if all(arm[key] for key in ("rerank_enabled", "dedup_enabled", "diversity_enabled")):
        return rerank_candidates(values, limit=max(arm["final_limit"], len(values)),
                                 min_score=arm["min_score"], now=now, trace=trace), trace
    scored = []
    for candidate in values:
        helpful, irrelevant = (max(0, int(candidate.get(k) or 0)) for k in ("helpful_count", "irrelevant_count"))
        score = float(candidate.get("relevance_score") or 0)
        if arm["rerank_enabled"]:
            score = (.7 * score + .1 * min(1., max(0., float(candidate.get("importance") or 0)))
                     + .08 * (helpful + 1) / (helpful + irrelevant + 2) + .07 * _recency_score(candidate, now)
                     + .05 * (1 - math.exp(-max(0, int(candidate.get("access_count") or 0)) / 5)))
        item = dict(candidate, final_score=score)
        if score < arm["min_score"]:
            trace["removed"].append({"candidate": trace_candidates([item])[0], "reason": "min_score"})
        else:
            scored.append(item)
    if arm["rerank_enabled"]:
        scored.sort(key=lambda c: (c["final_score"], str(c.get("occurred_at") or "")), reverse=True)
    trace["scored"] = trace_candidates(scored)
    selected, deferred, subjects = [], [], {}
    for candidate in scored:
        duplicate = next((existing for existing in selected
                          if arm["dedup_enabled"] and candidate.get("kind") == existing.get("kind")
                          and (candidate.get("kind") != "fact" or candidate.get("subject") == existing.get("subject"))
                          and (_jaccard(str(candidate.get("text") or ""), str(existing.get("text") or "")) >= .8
                               or _cosine(candidate.get("embedding_vector"), existing.get("embedding_vector")) >= .86)), None)
        if duplicate:
            trace["removed"].append({"candidate": trace_candidates([candidate])[0], "reason": "duplicate", "duplicate_of": duplicate["memory_key"]})
            continue
        subject = str(candidate.get("subject") or "")
        if arm["diversity_enabled"] and candidate.get("kind") == "fact" and subjects.get(subject, 0) >= 2:
            deferred.append(candidate)
            trace["deferred"].append({"memory_key": candidate["memory_key"], "reason": "fact_subject_diversity"})
        else:
            selected.append(candidate)
            if candidate.get("kind") == "fact":
                subjects[subject] = subjects.get(subject, 0) + 1
    for candidate in deferred:
        duplicate = next((existing for existing in selected
                          if arm["dedup_enabled"] and candidate.get("kind") == existing.get("kind")
                          and (candidate.get("kind") != "fact" or candidate.get("subject") == existing.get("subject"))
                          and (_jaccard(str(candidate.get("text") or ""), str(existing.get("text") or "")) >= .8
                               or _cosine(candidate.get("embedding_vector"), existing.get("embedding_vector")) >= .86)), None)
        if duplicate:
            trace["removed"].append({"candidate": trace_candidates([candidate])[0], "reason": "duplicate", "duplicate_of": duplicate["memory_key"]})
        else:
            selected.append(candidate)
    trace["candidates"] = trace_candidates(selected)
    return selected, trace


def retrieve(service: Any, query: str, arm: dict, *, check_stop: Callable[[], None],
             stage: Callable[[str], None], now: datetime | None = None) -> tuple[str, dict]:
    repository = service.repository
    query = query.strip()
    trace: dict = {"schema_version": 2, "query": query, "config": arm, "stages": {}, "gate": {}}
    stages = trace["stages"]
    route, source, confidence = classify_memory_query(query), "rule", None

    def finish(memory: str, candidates: list[dict], reason: str) -> tuple[str, dict]:
        trace["final"] = {"content": memory, "candidates": trace_candidates(candidates), "hit_count": len(candidates),
                          "decision": "retrieve" if memory else "skip", "reason": reason, "strategy": arm["retrieval_mode"],
                          "route": route, "route_source": source, "route_confidence": confidence}
        return memory, trace

    stage("gate")
    check_stop()
    if not query or _TRIVIAL_QUERY.fullmatch(query) or not repository.has_searchable_memory():
        trace["gate"] = {"decision": "skip", "reason": "empty_or_trivial_or_no_memory", "source": "hard_rule"}
        return finish("", [], "hard_rule_skip")
    direct = bool(_DIRECT_MEMORY_REFERENCE.search(query))
    should_retrieve = True
    if direct:
        source = "hard_rule"
    elif arm["gate_enabled"] or arm["route_mode"] == "auto":
        check_stop()
        should_retrieve, resolved, source, confidence = service._resolve_memory_intent(query, fallback_route=route)
        check_stop()
        if arm["route_mode"] == "auto":
            route = resolved
        else:
            source = "rule"
        if not arm["gate_enabled"]:
            should_retrieve = True
    if arm["route_mode"] == "fixed":
        route, source = arm["fixed_route"], "fixed"
    trace["gate"] = {"decision": "retrieve" if should_retrieve else "skip", "should_retrieve": should_retrieve,
                     "gate_enabled": arm["gate_enabled"], "direct_reference": direct,
                     "route": route, "route_source": source, "route_confidence": confidence}
    if not should_retrieve:
        return finish("", [], "intent_gate_skip")
    stage("bm25")
    check_stop()
    bm25 = repository.bm25_candidates(query, limit=arm["candidate_limit"]) if arm["retrieval_mode"] in {"bm25", "hybrid"} else []
    stages["bm25"] = {"enabled": arm["retrieval_mode"] != "vector", "candidates": trace_candidates(bm25)}
    stage("vector")
    check_stop()
    vector = []
    if arm["retrieval_mode"] in {"vector", "hybrid"}:
        vectors = service.provider.embed([query], model=arm["embedding_model"])
        check_stop()
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("query_embedding_missing")
        vector = repository.vector_candidates(vectors[0], limit=arm["candidate_limit"], model=arm["embedding_model"])
    stages["vector"] = {"enabled": arm["retrieval_mode"] != "bm25", "candidates": trace_candidates(vector)}
    stage("rrf")
    fused = rrf_fuse(bm25, vector, rank_constant=arm["rrf_k"] if arm["retrieval_mode"] == "hybrid" else 60)
    stages["rrf"] = {"enabled": arm["retrieval_mode"] == "hybrid", "candidates": trace_candidates(fused)}
    stage("expanded")
    expanded = repository.expand_chat_candidates(fused) if arm["expand_exchanges"] else fused
    stages["expanded"] = {"enabled": arm["expand_exchanges"], "candidates": trace_candidates(expanded)}
    if not expanded and direct:
        allowed = {kind for kind, field in (("fact", "use_facts"), ("episode", "use_episodes"), ("chat", "use_chat")) if arm[field]}
        expanded = [dict(candidate, relevance_score=.35, retrieval_sources=["recent"], source_ranks={"recent": rank})
                    for rank, candidate in enumerate(repository.recent_candidates(limit=arm["final_limit"]), 1)
                    if candidate.get("kind") in allowed]
        stages["recent_fallback"] = {"candidates": trace_candidates(expanded)}
    stage("rerank")
    ranked, stages["rerank"] = _rank(expanded, arm, now or datetime.now(timezone.utc))
    stage("route")
    if arm["quota_mode"] == "default":
        selection_trace = {}
        selected = select_routed_candidates(ranked, route=route, limit=arm["final_limit"], trace=selection_trace)
    elif arm["quota_mode"] == "none":
        selected = ranked[:arm["final_limit"]]
        selection_trace = {"quota_mode": "none"}
    else:
        selected = []
        for kind, count in arm["custom_quota"].items():
            selected.extend([c for c in ranked if c["kind"] == kind][:count])
        keys = {c["memory_key"] for c in selected}
        selected.extend([c for c in ranked if c["memory_key"] not in keys][:arm["final_limit"] - len(selected)])
        selection_trace = {"quota_mode": "custom", "quotas": arm["custom_quota"], "fallback": "ranking_order"}
    if "removed" not in selection_trace:
        keys = {c["memory_key"] for c in selected}
        selection_trace["removed"] = [{"candidate": c, "reason": "quota_or_limit"} for c in trace_candidates(ranked) if c["memory_key"] not in keys]
    stages["route"] = dict(selection_trace, candidates=trace_candidates(selected))
    stage("render")
    ceiling = arm["token_budget"] * 4
    memory, removed = render_candidates(selected), []
    before = len(memory)
    while len(selected) > 1 and len(memory) > ceiling:
        removed.append({"candidate": trace_candidates([selected.pop()])[0], "reason": "character_budget"})
        memory = render_candidates(selected)
    stages["render"] = {"character_budget": ceiling, "before_characters": before, "removed": removed,
                        "candidates": trace_candidates(selected), "truncated": len(memory) > ceiling,
                        "truncated_characters": max(0, len(memory) - ceiling), "content": memory[:ceiling]}
    check_stop()
    return finish(memory[:ceiling], selected, "selected" if selected else "no_relevant_memory")
