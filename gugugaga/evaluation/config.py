"""Frozen, credential-free configuration for isolated memory experiments."""
from __future__ import annotations

import copy
import math
import re
from typing import Any

ARM_DEFAULTS = {
    "id": "a", "name": "基线 A", "mode": "memory", "memory_source": "rebuild", "snapshot_id": "",
    "generate_facts": True, "generate_episodes": True, "use_chat": True, "use_facts": True,
    "use_episodes": True, "threshold": 6, "fact_min_importance": .8, "episode_min_importance": .6,
    "retrieval_mode": "bm25", "evidence_window": 0, "gate_enabled": True, "route_mode": "auto",
    "fixed_route": "mixed", "quota_mode": "default", "custom_quota": {"fact": 2, "episode": 1, "chat": 2},
    "candidate_limit": 20, "final_limit": 5, "token_budget": 2000, "rrf_k": 60,
    "rerank_enabled": True, "dedup_enabled": True, "diversity_enabled": True,
    "expand_exchanges": True, "min_score": .2, "answer_model": "", "consolidation_model": "",
    "intent_model": "", "embedding_model": "", "temperature": 0.0, "thinking_disabled": True,
    "max_tokens": 256,
}
ENUMS = {"mode": ("memory", "no_memory", "oracle"), "memory_source": ("rebuild", "snapshot"),
         "retrieval_mode": ("bm25", "vector", "hybrid"), "route_mode": ("auto", "rule", "fixed"),
         "fixed_route": ("fact", "episode", "evidence", "mixed"), "quota_mode": ("default", "none", "custom")}
INT_RANGES = {"threshold": (1,100), "evidence_window": (0,10000), "candidate_limit": (1,100),
              "final_limit": (1,20), "token_budget": (1,32000), "rrf_k": (1,1000), "max_tokens": (1,8192)}
FLOAT_RANGES = {"fact_min_importance": (0,1), "episode_min_importance": (0,1), "min_score": (0,1), "temperature": (0,2)}
BUILD_FIELDS = ("generate_facts", "generate_episodes", "threshold", "fact_min_importance", "episode_min_importance", "consolidation_model")


def _integer(value: Any, low: int, high: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{field} 必须是 {low}–{high} 之间的整数")
    return value


def normalize_config(payload: dict, defaults: dict | None = None) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("实验配置必须是对象")
    unknown = set(payload) - {"name", "type", "dataset", "repeats", "repeat_scope", "arms"}
    if unknown:
        raise ValueError(f"未知实验配置字段：{', '.join(sorted(unknown))}")
    base = copy.deepcopy(ARM_DEFAULTS)
    supplied_defaults = (defaults or {}).get("default_arm", defaults or {})
    base.update({key: value for key, value in supplied_defaults.items() if key in base})
    name = payload.get("name", "记忆测评")
    if not isinstance(name, str) or not name.strip() or len(name) > 160:
        raise ValueError("实验名称不能为空，且不超过 160 字符")
    kind = payload.get("type", "basic")
    if kind not in {"basic", "ablation", "stability"}:
        raise ValueError("实验类型必须是 basic/ablation/stability")
    repeats = _integer(payload.get("repeats", 3 if kind == "stability" else 1), 1, 20, "repeats")
    scope = payload.get("repeat_scope", "full")
    if scope not in {"full", "answers"}:
        raise ValueError("repeat_scope 必须是 full/answers")
    if kind != "stability" and repeats != 1:
        raise ValueError("多次重复运行请使用 stability 类型")
    if kind == "stability" and repeats < 2:
        raise ValueError("稳定性实验至少需要 2 次重复")
    selection = copy.deepcopy(payload.get("dataset", {"mode": "smoke"}))
    if not isinstance(selection, dict) or set(selection) - {"mode", "sample_ids", "max_questions", "qa_ids"}:
        raise ValueError("dataset 包含未知字段或不是对象")
    selection.setdefault("mode", "smoke")
    if selection["mode"] not in {"smoke", "frozen200", "custom"}:
        raise ValueError("数据范围必须是 smoke/frozen200/custom")
    for field in ("sample_ids", "qa_ids"):
        selection.setdefault(field, [])
        if not isinstance(selection[field], list) or any(not isinstance(x, str) or not x for x in selection[field]):
            raise ValueError(f"dataset.{field} 必须是非空字符串的数组")
        if len(set(selection[field])) != len(selection[field]):
            raise ValueError(f"dataset.{field} 不能重复")
    selection.setdefault("max_questions", 200 if selection["mode"] == "frozen200" else 20)
    _integer(selection["max_questions"], 1, 10000, "max_questions")
    arms = payload.get("arms", [base])
    if not isinstance(arms, list) or not 1 <= len(arms) <= 8:
        raise ValueError("实验必须包含 1–8 个方案")
    normalized = []
    ids = set()
    for index, values in enumerate(arms):
        if not isinstance(values, dict) or set(values) - set(base):
            raise ValueError(f"方案 {index + 1} 包含未知配置字段或不是对象")
        arm = {**copy.deepcopy(base), "id": f"arm{index + 1}", "name": f"方案 {index + 1}", **copy.deepcopy(values)}
        if not isinstance(arm["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", arm["id"]) or arm["id"] in ids:
            raise ValueError("方案 id 必须唯一且仅含字母、数字、短横线、下划线（1–40 位）")
        ids.add(arm["id"])
        for field in ("name", "snapshot_id", "answer_model", "consolidation_model", "intent_model", "embedding_model"):
            if not isinstance(arm[field], str) or len(arm[field]) > 256:
                raise ValueError(f"{field} 必须为不超过 256 字符的字符串")
            arm[field] = arm[field].strip()
        if not arm["name"] or not arm["answer_model"]:
            raise ValueError("方案名称和回答模型不能为空")
        for field, default in ARM_DEFAULTS.items():
            if isinstance(default, bool) and not isinstance(arm[field], bool):
                raise ValueError(f"{field} 必须为布尔值")
        for field, choices in ENUMS.items():
            if arm[field] not in choices:
                raise ValueError(f"{field} 必须取值于 {'/'.join(choices)}")
        for field, bounds in INT_RANGES.items():
            _integer(arm[field], *bounds, field)
        for field, bounds in FLOAT_RANGES.items():
            value = arm[field]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{field} 必须在 {bounds[0]}–{bounds[1]} 之间")
        quota = arm["custom_quota"]
        if not isinstance(quota, dict) or set(quota) != {"fact", "episode", "chat"}:
            raise ValueError("custom_quota 必须包含 fact/episode/chat 三种配额")
        for key, value in quota.items():
            _integer(value, 0, 20, f"custom_quota.{key}")
        if arm["final_limit"] > arm["candidate_limit"]:
            raise ValueError("Top K 不能大于各路候选上限")
        if arm["mode"] == "memory":
            if not any(arm[x] for x in ("use_chat", "use_facts", "use_episodes")):
                raise ValueError("记忆方案至少启用一种检索内容")
            if arm["retrieval_mode"] != "bm25" and not arm["embedding_model"]:
                raise ValueError("Vector/Hybrid 检索必须配置 Embedding 模型")
            if arm["memory_source"] == "snapshot" and not arm["snapshot_id"]:
                raise ValueError("复用记忆必须选择快照")
            if arm["memory_source"] == "rebuild":
                if arm["use_facts"] and not arm["generate_facts"] or arm["use_episodes"] and not arm["generate_episodes"]:
                    raise ValueError("重建模式下不能检索未生成的 Fact/Episode")
                if (arm["generate_facts"] or arm["generate_episodes"]) and not arm["consolidation_model"]:
                    raise ValueError("生成摘要必须配置整合模型")
            if (arm["gate_enabled"] or arm["route_mode"] == "auto") and not arm["intent_model"]:
                raise ValueError("Gate 或自动路由需要配置意图模型")
            if arm["quota_mode"] == "custom":
                if sum(quota.values()) != arm["final_limit"]:
                    raise ValueError("自定义配额总量必须等于 Top K")
                for key, switch in (("fact", "use_facts"), ("episode", "use_episodes"), ("chat", "use_chat")):
                    if quota[key] and not arm[switch]:
                        raise ValueError(f"关闭 {key} 检索时其自定义配额必须为 0")
        normalized.append(arm)
    if kind == "ablation" and len(normalized) < 2:
        raise ValueError("对照实验至少需要两个方案")
    if kind == "stability" and scope == "full" and any(a["mode"] == "memory" and a["memory_source"] != "rebuild" for a in normalized):
        raise ValueError("完整流程稳定性必须重新构建记忆；快照请选择只重复回答")
    return {"name": name.strip(), "type": kind, "dataset": selection, "repeats": repeats, "repeat_scope": scope, "arms": normalized}


def catalog(defaults: dict | None = None) -> dict:
    base = copy.deepcopy(ARM_DEFAULTS)
    values = (defaults or {}).get("default_arm", defaults or {})
    base.update({key: value for key, value in values.items() if key in base})
    basic = [dict(base, id="no_memory", name="无记忆", mode="no_memory"), dict(base), dict(base, id="oracle", name="标准证据 Oracle", mode="oracle")]
    presets = [
        ("basic", "基础评测", "basic", basic),
        ("vector_coverage", "历史原文向量覆盖", "ablation", [dict(base, retrieval_mode="hybrid", evidence_window=30), dict(base, id="b", name="全量原文向量", retrieval_mode="hybrid", evidence_window=0)]),
        ("quota", "类型配额消融", "ablation", [dict(base), dict(base, id="b", name="取消配额", quota_mode="none")]),
        ("summaries", "摘要层消融", "ablation", [dict(base), dict(base, id="b", name="仅原文", use_facts=False, use_episodes=False)]),
        ("top_k", "最终召回数量", "ablation", [dict(base), dict(base, id="b", name="Top 10", final_limit=10)]),
        ("budget", "记忆上下文预算", "ablation", [dict(base, final_limit=10), dict(base, id="b", name="预算 16000", final_limit=10, token_budget=16000)]),
        ("stability", "稳定性实验", "stability", [dict(base)]),
    ]
    templates = [{"id": key, "name": label, "type": kind, "arms": arms,
                  "config": {"name": label, "type": kind, "dataset": {"mode": "smoke", "max_questions": 20}, "arms": arms, "repeats": 3 if kind == "stability" else 1, "repeat_scope": "full"}}
                 for key, label, kind, arms in presets]
    return {"default_arm": base, "defaults": {"name": "记忆测评", "type": "basic", "dataset": {"mode": "smoke", "max_questions": 20}, "repeats": 1, "repeat_scope": "full", "arms": basic},
            "templates": templates, "enums": ENUMS, "integer_ranges": INT_RANGES, "float_ranges": FLOAT_RANGES,
            "budget_unit": "approximate_tokens_4_characters", "build_fields": list(BUILD_FIELDS)}
