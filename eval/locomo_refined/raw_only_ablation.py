"""Prepare chat-only retrieval indexes on an isolated evaluation database copy."""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import datetime
import hashlib
import math
from pathlib import Path
import sqlite3
from typing import Any

from gugugaga.memory.models import RecallItem
from gugugaga.memory.repository import MemoryRepository
from gugugaga.memory.retrieval import (
    render_candidates,
    rerank_candidates,
    rrf_fuse,
    select_routed_candidates,
    trace_candidates,
)


def _counts(db: sqlite3.Connection) -> dict[str, Any]:
    return {
        "chat_rows": db.execute("SELECT COUNT(*) FROM chat_log WHERE is_final=1").fetchone()[0],
        "fts_by_kind": dict(db.execute("SELECT kind, COUNT(*) FROM memory_fts GROUP BY kind")),
        "vectors_by_kind": dict(db.execute(
            "SELECT memory_kind, COUNT(*) FROM memory_embeddings GROUP BY memory_kind"
        )),
        "index_jobs_by_kind": dict(db.execute(
            "SELECT memory_kind, COUNT(*) FROM memory_index_outbox GROUP BY memory_kind"
        )),
    }


def prepare_raw_only_indexes(
    target_db: Path | str, lexical_source_db: Path | str
) -> dict[str, Any]:
    """Restrict the copy's BM25/vector corpora before candidate selection.

    Call this AFTER constructing MemoryRepository, because initialization
    rebuilds FTS from all memory types. Repeat after every repository restart,
    and do not run consolidation or indexing workers.
    Chat FTS row IDs come from the frozen lexical source; all other content and
    existing chat embeddings stay unchanged. Only ``target_db`` is writable.
    """
    target, source = Path(target_db).resolve(), Path(lexical_source_db).resolve()
    if not target.is_file() or not source.is_file():
        raise FileNotFoundError("both target copy and frozen lexical source must exist")
    if target == source or target.samefile(source):
        raise ValueError("raw-only indexes require a separate target database copy")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as origin:
        lexical_rows = origin.execute(
            "SELECT rowid, memory_key, kind, subject, text, occurred_at "
            "FROM memory_fts WHERE kind='chat' ORDER BY rowid"
        ).fetchall()
    with closing(sqlite3.connect(target)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        before = _counts(db)
        raw_rows = db.execute("SELECT * FROM chat_log ORDER BY id").fetchall()
        raw_vectors = db.execute(
            "SELECT * FROM memory_embeddings WHERE memory_kind='chat' ORDER BY memory_key"
        ).fetchall()
        expected_lexical = db.execute(
            "SELECT 'chat:'||id, 'chat', role, content, COALESCE(completed_at, created_at) "
            "FROM chat_log WHERE is_final=1 ORDER BY id"
        ).fetchall()
        by_key = {row[0]: row for row in expected_lexical}
        if len(lexical_rows) != len(by_key) or {row[1]: row[1:] for row in lexical_rows} != by_key:
            raise ValueError("frozen lexical source does not match the target chat corpus")
        vector_keys = {row[0] for row in raw_vectors}
        if len(raw_vectors) != len(by_key) or vector_keys != set(by_key):
            raise ValueError("target must already have one embedding for every final chat message")

        db.execute("DELETE FROM memory_fts")
        db.executemany(
            "INSERT INTO memory_fts(rowid, memory_key, kind, subject, text, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            lexical_rows,
        )
        db.execute("DELETE FROM memory_embeddings WHERE memory_kind<>'chat'")
        # Prevent a stale queued summary job from restoring excluded embeddings.
        db.execute("DELETE FROM memory_index_outbox WHERE memory_kind<>'chat'")

        after = _counts(db)
        if db.execute("SELECT * FROM chat_log ORDER BY id").fetchall() != raw_rows:
            raise RuntimeError("raw chat content changed during index preparation")
        if db.execute(
            "SELECT * FROM memory_embeddings WHERE memory_kind='chat' ORDER BY memory_key"
        ).fetchall() != raw_vectors:
            raise RuntimeError("raw chat embeddings changed during index preparation")
        rebuilt = db.execute(
            "SELECT rowid, memory_key, kind, subject, text, occurred_at "
            "FROM memory_fts ORDER BY rowid"
        ).fetchall()
        if rebuilt != lexical_rows:
            raise RuntimeError("chat FTS row ordering differs from the frozen lexical source")
        if set(after["fts_by_kind"]) - {"chat"} or set(after["vectors_by_kind"]) - {"chat"}:
            raise RuntimeError("summary candidates remain in raw-only retrieval indexes")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("frozen lexical source hash changed")
    return {
        "before": before,
        "after": after,
        "raw_content_unchanged": True,
        "raw_embeddings_unchanged": True,
        "lexical_rowids_preserved": True,
        "lexical_source_sha256": source_hash,
        "lexical_source_unchanged": True,
    }


def retrieve_raw_only(
    repository: MemoryRepository,
    source_trace: dict[str, Any],
    query_vector: list[float],
) -> dict[str, Any]:
    """Run the frozen retrieval pipeline against prepared chat-only indexes.

    The query vector, original route, rerank clock, limits and render budget are
    supplied by the baseline trace. This performs no provider or usage calls.
    Actual candidate vectors remain available for semantic deduplication and
    are removed only from the returned diagnostic snapshots.
    """
    vector = [float(value) for value in query_vector]
    if not vector or not all(math.isfinite(value) for value in vector) or not any(vector):
        raise ValueError("a finite nonzero cached query vector is required")
    trace = deepcopy(source_trace)
    trace.pop("vector_error", None)
    trace.pop("error", None)
    trace["retrieval_policy"] = "raw_only"
    trace["stages"] = {}
    config = trace["config"]
    candidate_limit, final_limit = int(config["candidate_limit"]), int(config["final_limit"])
    token_budget = int(config["recall_token_budget"])
    if not 1 <= final_limit <= candidate_limit or token_budget < 1:
        raise ValueError("invalid frozen retrieval limits or render budget")
    query = str(trace["query"]).strip()
    route = str(source_trace["final"]["route"])
    clock = datetime.fromisoformat(source_trace["stages"]["rerank"]["now"])
    if clock.tzinfo is None:
        raise ValueError("the frozen rerank clock must include a timezone")

    bm25 = repository.bm25_candidates(query, limit=candidate_limit)
    semantic = repository.vector_candidates(
        vector, limit=candidate_limit, model=config["embedding_model"]
    )
    if any(item.get("kind") != "chat" for item in [*bm25, *semantic]):
        raise RuntimeError("raw-only retrieval indexes contain summary candidates")
    trace["stages"]["bm25"] = {"candidates": trace_candidates(bm25)}
    trace["stages"]["vector"] = {"candidates": trace_candidates(semantic)}
    fused = rrf_fuse(bm25, semantic)
    trace["stages"]["rrf"] = {"candidates": trace_candidates(fused)}
    expanded = repository.expand_chat_candidates(fused)
    trace["stages"]["expanded"] = {"candidates": trace_candidates(expanded)}
    strategy = "hybrid" if bm25 and semantic else "vector" if semantic else "bm25" if bm25 else "none"
    rerank_trace: dict[str, Any] = {}
    ranked = rerank_candidates(
        expanded, limit=max(final_limit, len(expanded)), min_score=float(config["min_score"]),
        now=clock, trace=rerank_trace,
    )
    trace["stages"]["rerank"] = rerank_trace
    route_trace: dict[str, Any] = {}
    selected = select_routed_candidates(ranked, route=route, limit=final_limit, trace=route_trace)
    trace["stages"]["route"] = route_trace
    character_budget = token_budget * 4
    rendered = render_candidates(selected)
    render_trace: dict[str, Any] = {
        "character_budget": character_budget, "before_characters": len(rendered), "removed": [],
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
    trace["final"] = {
        "decision": "retrieve" if selected else "skip",
        "reason": "no_relevant_memory" if not selected else "semantic_match" if strategy == "vector" else "lexical_match",
        "strategy": strategy,
        "route": route,
        "route_source": source_trace["final"].get("route_source", "frozen"),
        "route_confidence": source_trace["final"].get("route_confidence"),
        "hit_count": len(selected),
        "kinds": ["evidence"] if selected else [],
        "memory_keys": [str(item["memory_key"]) for item in selected],
        "items": [
            RecallItem(
                memory_key=str(item["memory_key"]), kind="chat",
                subject=str(item.get("subject") or "")[:500],
                text=str(item.get("text") or "")[:4000],
                occurred_at=str(item["occurred_at"]) if item.get("occurred_at") is not None else None,
                retrieval_sources=tuple(str(value) for value in item.get("retrieval_sources") or ()),
                source_ranks={str(key): int(value) for key, value in (item.get("source_ranks") or {}).items()},
                relevance_score=float(item.get("relevance_score") or 0.0),
                final_score=float(item.get("final_score") or 0.0),
            ).as_dict()
            for item in selected
        ],
        "candidates": trace_candidates(selected),
        "content": render_trace["content"],
    }
    return trace
