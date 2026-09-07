# LoCoMo-Refined 200题记忆系统与 Oracle 诊断报告

## 1. 实验目的

本实验用于评估 Gugugaga Agent 当前长期记忆模块在不同长对话上的整体表现，并通过 Gold Evidence Oracle 区分以下问题：

1. 记忆整合阶段是否保留了可回答的信息；
2. 检索阶段能否将标准证据召回至最终 Top 5；
3. 回答模型拿到证据后能否生成正确答案；
4. 当前系统距离“使用标准证据回答”的经验上界还有多大差距。

本次冻结 Agent 代码、路由规则和提示词，没有根据单题结果继续调参。

## 2. 实验配置

| 配置项 | 值 |
|---|---|
| 测试日期 | 2026-09-05 |
| 数据集 | LoCoMo-Refined |
| 会话数 | 10 |
| 每个会话题数 | 20 |
| 总题数 | 200 |
| 回答模型 | `Qwen/Qwen3.6-35B-A3B` |
| Temperature | `0` |
| Thinking | Disabled |
| 检索方式 | FTS5/BM25 + Vector + RRF Hybrid Recall |
| 初始候选数 | Top 20 |
| 最终召回数 | Top 5 |
| Fact 最低重要度 | `0.8` |
| Episode 最低重要度 | `0.6` |
| 每批最大 Episode 数 | `5` |

每段会话都使用全新隔离的 SQLite 数据库，完整执行对话回放、记忆整合、向量索引、Intent Gate、路由、Memory 回答和 No-Memory 回答。随后对同一批问题提供数据集标注的 Gold Evidence，运行 Oracle 回答。

运行结果中的 `consolidation_model` 字段记录为 `null`，表示未单独指定记忆整合模型；Provider 实际回退到本次 `--model` 指定的主模型，即 `Qwen/Qwen3.6-35B-A3B`。

本实验是覆盖全部 10 段会话的 200 题分层抽样评测，不是对数据集中 1382 道问题的全量遍历。部分会话缺少足够的某类问题，因此最终类别数量不是严格的每类 50 题。

## 3. 对照组定义

### No-Memory

只向模型提供问题，不提供长期记忆，用于衡量模型常识、训练数据记忆和无依据猜测形成的基线。

### Memory

运行当前完整记忆系统，根据 Intent Gate 和 Query Route 从 Fact、Episode、Conversation Evidence 中选择最终 Top 5，再交给模型回答。

### Gold Evidence Oracle

绕过记忆检索，直接将数据集标注的标准证据交给相同回答模型。Oracle 衡量的是当前回答模型和回答协议在“已获得标准证据”条件下的经验上界，不是数学意义上的 100 分上限。

## 4. 总体结果

| 模式 | Token F1 | BLEU-1 |
|---|---:|---:|
| No-Memory | 2.72 | 1.32 |
| Memory | **26.01** | **20.34** |
| Gold Evidence Oracle | **50.85** | **44.71** |

核心结果：

```text
Memory Gain F1 = 26.01 - 2.72 = 23.29
Oracle 达成率 = 26.01 / 50.85 = 51.15%
去除 No-Memory 后的 Oracle 达成率
= (26.01 - 2.72) / (50.85 - 2.72)
= 48.39%
```

10 个会话的 Memory F1 为 `26.01 ± 10.37`。这里的标准差表示不同会话之间的难度和性能差异，不是相同会话重复运行的随机性。

## 5. 分类别结果

| 类别 | 题数 | No-Memory F1 | Memory F1 | Oracle F1 | Oracle 达成率 |
|---|---:|---:|---:|---:|---:|
| Multi-hop | 52 | 4.56 | **22.18** | 54.84 | 40.44% |
| Temporal | 56 | 0.65 | **27.62** | 32.71 | **84.42%** |
| Open-domain | 42 | 5.69 | **19.22** | 40.57 | 47.37% |
| Single-hop | 50 | 0.63 | **33.91** | 75.65 | 44.82% |

Temporal 最接近 Oracle，表明时间戳和 Episode 设计能够保留部分时间信息。但 Temporal Oracle 本身只有 32.71 F1，日期归一化与短答案格式仍显著限制绝对分数。

Single-hop 的绝对 F1 最高，但与 Oracle 仍相差 41.74 个百分点，说明许多普通事实没有稳定进入最终 Top 5。

Multi-hop 与 Open-domain 同时受到检索覆盖、跨证据组合和回答模型推断能力的影响。

## 6. 分会话结果

| 会话 | Memory F1 | No-Memory F1 | Oracle F1 | Oracle 达成率 | Evidence Hit@5 |
|---|---:|---:|---:|---:|---:|
| conv-26 | 31.62 | 1.64 | 40.51 | 78.05% | 36.84% |
| conv-30 | 28.41 | 1.19 | 45.89 | 61.90% | 60.00% |
| conv-41 | 31.04 | 1.06 | 48.59 | 63.87% | 45.00% |
| conv-42 | 30.28 | 8.37 | 57.25 | 52.90% | 40.00% |
| conv-43 | 41.61 | 3.20 | 52.59 | 79.12% | 70.00% |
| conv-44 | 16.22 | 2.42 | 47.50 | 34.14% | 55.00% |
| conv-47 | 13.33 | 5.75 | 51.32 | 25.98% | 25.00% |
| conv-48 | 8.38 | 0.00 | 47.46 | 17.65% | 45.00% |
| conv-49 | 34.63 | 3.58 | 58.58 | 59.11% | 63.16% |
| conv-50 | 24.61 | 0.00 | 58.81 | 41.85% | 57.89% |

不同会话之间差异很大。`conv-43` 的 Memory F1 为 41.61，而 `conv-48` 只有 8.38，说明单独使用 `conv-26` 的 20 题结果无法代表整体性能。

## 7. Evidence 检索诊断

200 题中有 197 题包含可映射的 Gold Evidence：

```text
Evidence Hit@5    = 49.75%
Evidence Recall@5 = 40.98%
```

这意味着每道题虽然都返回了 5 条记忆，但约一半问题的最终 Top 5 没有包含任何标准证据。当前主要问题不是空召回，而是召回结果与问题不匹配。

诊断类型如下：

| 诊断类型 | 数量 | 占 200 题 |
|---|---:|---:|
| 成功或部分成功 | 73 | 36.5% |
| 检索失败 | 72 | 36.0% |
| Oracle 回答失败 | 41 | 20.5% |
| 已召回证据但回答失败 | 11 | 5.5% |
| 无 Gold Evidence、不可评分 | 3 | 1.5% |

按类别拆分：

| 类别 | 成功或部分成功 | 检索失败 | Oracle 回答失败 | 有证据但回答失败 | 不可评分 |
|---|---:|---:|---:|---:|---:|
| Multi-hop | 19 | 21 | 10 | 2 | 0 |
| Temporal | 28 | 14 | 12 | 2 | 0 |
| Open-domain | 9 | 13 | 14 | 3 | 3 |
| Single-hop | 17 | 24 | 5 | 4 | 0 |

Single-hop 有 24/50 题被诊断为检索失败。Open-domain 只有 9/42 题成功或部分成功，并有 14 题 Oracle 回答失败，说明该类别同时受检索和模型推断能力限制。

## 8. Query Route 与 Top 5 构成

200 次查询的路由分布：

| 路由 | 数量 | 占比 |
|---|---:|---:|
| Episode | 111 | 55.5% |
| Fact | 67 | 33.5% |
| Mixed | 22 | 11.0% |
| Evidence | 0 | 0% |

路由判断来源：

| 来源 | 数量 | 占比 |
|---|---:|---:|
| LLM Intent Gate | 175 | 87.5% |
| Rule Fallback | 25 | 12.5% |

最终 1000 条 Top 5 记忆由以下类型构成：

| 类型 | 数量 | 占比 |
|---|---:|---:|
| Episode | 416 | 41.6% |
| Conversation Evidence | 354 | 35.4% |
| Fact | 230 | 23.0% |

按路由统计诊断结果：

| 路由 | 查询数 | 成功或部分成功 | 检索失败 | Oracle 回答失败 | 有证据但回答失败 |
|---|---:|---:|---:|---:|---:|
| Episode | 111 | 49 | 31 | 26 | 4 |
| Fact | 67 | 20 | 28 | 12 | 6 |
| Mixed | 22 | 4 | 13 | 3 | 1 |

Mixed 路由的成功比例最低，22 题中有 13 题检索失败。当前没有任何查询被路由为纯 Evidence，可能使需要原始措辞或普通细节的问题过度依赖 Fact/Episode。

## 9. 记忆整合与工程有效性

本次完整处理：

```text
3011 个 Exchange
507 个记忆整合批次
198 条 Fact
227 条 Episode
3011 条 Conversation Evidence
```

工程有效性检查：

- 10 个会话全部完成；
- Memory、No-Memory、Oracle 各生成 200 份预测；
- 200 个问题 ID 唯一，无重复；
- 所有数据库 `last_failure = null`；
- 507 个整合批次的重试次数为 0；
- 所有向量索引状态为 synced；
- 200 题空召回次数为 0；
- 每题最终召回数量均为 5；
- LoCoMo-Refined 离线测试为 `7 passed`。

三种回答模式共记录 280244 个回答阶段 Token：

| 阶段 | Token |
|---|---:|
| Memory 回答 | 193176 |
| No-Memory 回答 | 27854 |
| Oracle 回答 | 59214 |

该数值只统计回答调用，不包括记忆整合、Intent Gate 和 Embedding 的 Token 或费用。

## 10. 效果不佳的主要原因

### 10.1 正确证据无法稳定进入最终 Top 5

Evidence Hit@5 只有 49.75%，72 题被直接诊断为检索失败。由于所有问题都有 5 条返回结果，问题不是数据库为空，而是检索结果被主题相近但不能回答问题的记忆占据。

这是当前影响 Memory F1 的最大直接原因。

### 10.2 冷 Conversation Evidence 缺少向量覆盖

数据库检查显示，每段会话只有 60 条 chat embedding，共 600 条；而 Conversation Evidence 总量为 3011 条。也就是说，约 20% 的 Evidence 有向量表示，大量较旧的冷 Evidence 主要依赖 FTS5/BM25。

当问题与历史对话使用不同措辞时，BM25 很难召回对应内容。例如关系状态、国家、收藏偏好和间接表达的兴趣问题，都容易被更高频的主题词结果覆盖。

长会话中的旧信息尤其容易退化。`conv-47` 有 355 个 Exchange，Hit@5 只有 25%，Memory F1 为 13.33。

### 10.3 结构化记忆覆盖仍然稀疏

3011 个 Exchange 最终只形成 198 条 Fact 和 227 条 Episode。Fact 的 0.8 阈值倾向于保留稳定、高重要度信息，但会过滤国家、关系状态、日常习惯、普通计划和收藏偏好等“重要度不高但可能被提问”的细节。

Episode 数组和独立阈值增强了经历类信息，但不能覆盖所有普通事实。没有被抽取为 Fact/Episode 的信息只能依赖 Conversation Evidence，而冷 Evidence 的语义召回又较弱，形成连续损失。

### 10.4 路由和配额可能挤掉直接证据

路由决定 Top 5 中 Fact、Episode、Evidence 的优先比例。如果路由判断不准确，某一类弱候选可能占据名额，将真正相关的原始 Evidence 挤出最终结果。

Mixed 路由 22 题中有 13 题检索失败，当前表现最差；纯 Evidence 路由没有被选择。说明当前类型分配不能稳定覆盖需要原话、普通细节或多条证据的问题。

当前产物只保存最终 Top 5，没有保存 RRF 融合后的完整 Top 20 候选及 Gold Evidence 排名。因此目前无法进一步区分：

```text
Gold Evidence 没有进入初始 Top 20
还是已经进入 Top 20，但被重排、去重或路由配额裁掉
```

这是现有诊断链路的重要信息缺口。

### 10.5 回答模型本身存在明显上限

Gold Evidence Oracle F1 只有 50.85，并有 41 题 Oracle F1 为 0。这说明即使标准证据已经直接提供，当前模型仍可能因为保守拒答、跨证据推断不足、相对时间换算失败或输出格式不匹配而失分。

此外还有 11 题已经在 Memory Top 5 中命中 Gold Evidence，但回答模型仍未答对。Open-domain 的 Oracle F1 只有 40.57，其中 15/42 题 Oracle F1 为 0，因此 Open-domain 的低分不能全部归因于记忆检索。

### 10.6 Token F1 对语义等价表达较敏感

Token F1 会惩罚日期格式和冗余措辞，例如：

```text
last Saturday vs 2023-05-20
last year vs 2022
Caroline is a transgender woman vs transgender woman
```

这些回答可能在人工语义判断下正确或部分正确，但 Token F1 会显著降低。Oracle 在部分题目上也可能低于 Memory，因此 Oracle 只能视为相同回答模型下的经验基线。

## 11. 结论

本次 200 题评测表明，当前记忆系统能够将 Token F1 从 No-Memory 的 2.72 提升至 26.01，提升 23.29 个百分点，并达到 Gold Evidence Oracle 的 51.15%。Temporal 类达到 Oracle 的 84.42%，说明 Episode 与时间戳设计具有实际收益。

同时，全量诊断暴露出以下主要边界：

1. 最终 Top 5 的标准证据命中率不足 50%；
2. 大量冷 Conversation Evidence 缺少向量覆盖；
3. Fact/Episode 对普通细节的结构化覆盖不足；
4. 路由与 Top 5 配额可能挤掉直接证据；
5. 当前回答模型即使获得 Gold Evidence，也只有 50.85 F1；
6. Token F1 会低估部分语义正确回答。

因此当前系统的主要问题不在事务、持久化或失败恢复，而在记忆内容覆盖、长尾证据召回、Top 20 到 Top 5 的筛选可诊断性，以及回答模型对证据的利用能力。

## 12. 原始产物

- 批次目录：`eval/locomo_refined/runs/qwen36-full-200-oracle-20260905-021445/`
- 机器可读汇总：`runs/qwen36-full-200-oracle-20260905-021445/batch-summary.json`
- 自动生成报告：`runs/qwen36-full-200-oracle-20260905-021445/report.md`
- 各会话结果：`runs/qwen36-full-200-oracle-20260905-021445/conv-*/summary.json`
- 各会话诊断：`runs/qwen36-full-200-oracle-20260905-021445/conv-*/diagnostics.json`
- Memory 预测：`runs/qwen36-full-200-oracle-20260905-021445/conv-*/predictions_current_memory.json`
- No-Memory 预测：`runs/qwen36-full-200-oracle-20260905-021445/conv-*/predictions_no_memory.json`
- Oracle 预测：`runs/qwen36-full-200-oracle-20260905-021445/conv-*/predictions_oracle_evidence.json`

