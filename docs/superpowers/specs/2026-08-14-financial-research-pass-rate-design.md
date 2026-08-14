# 金融研究判别器通过率汇总设计

## 目标

更新个人 Skill `evaluating-financial-research`，使其在保留每个任务 16 项二元判定的同时，为批量评测生成统一通过率。同步为当前 `minimax-financegym20` 生成汇总文件。

## 唯一定义

批量通过率按所有 rubric 中 `value=1` 的占比计算：

```text
passed_items = 所有任务中 value=1 的 rubric 数量
total_items = 任务数 × 16
overall_pass_rate = passed_items / total_items
```

不使用权重，不把“16 项全部为 1 的任务比例”作为默认通过率，也不把 16 项相加后转换为传统分数。

## 输出

逐任务 JSON 保持不变：每个文件仍恰好包含 `R1–R4`、`P1–P4`、`E1–E4`、`K1–K4`，每项包含 `value`、`reason` 和 `evidence`。

批量评测额外生成 `judgements/summary.json`：

```json
{
  "definition": "passed_items / total_items",
  "task_count": 20,
  "rubrics_per_task": 16,
  "passed_items": 179,
  "total_items": 320,
  "overall_pass_rate": 0.559375,
  "layers": {
    "result": {"passed_items": 43, "total_items": 80, "pass_rate": 0.5375},
    "process": {"passed_items": 22, "total_items": 80, "pass_rate": 0.275},
    "efficiency": {"passed_items": 77, "total_items": 80, "pass_rate": 0.9625},
    "risk": {"passed_items": 37, "total_items": 80, "pass_rate": 0.4625}
  }
}
```

比例字段保存为 `0–1` 小数；展示给用户时可以同时格式化为百分比。`summary.json` 不包含加权项、等级或排名。

## Skill 修改范围

- 更新 `SKILL.md` 的触发描述和批量工作流，允许且要求上述通过率汇总。
- 将原先“禁止百分比”的规则收窄为“禁止未定义的总分、加权分、等级和排名”。
- 在 `references/rubric.md` 中加入批量汇总公式与边界规则。
- 更新 `agents/openai.yaml` 默认提示词，使批量审查会生成汇总通过率。

## 边界与错误处理

- 汇总前必须验证每个任务文件恰好包含 16 个 rubric，且 `value` 是整数 `0` 或 `1`。
- 发现缺失任务、重复任务、额外 rubric、非二元值或无法解析的 JSON 时停止计算，不用不完整数据生成通过率。
- 层级通过率分别按每层 4 项计算；20 个任务时每层分母为 80。
- 修改 Skill 不改变已有的逐任务判定内容。

## 验证

- Skill 目录通过 `quick_validate.py`。
- 当前 20 个任务应产生 `passed_items=179`、`total_items=320`、`overall_pass_rate=0.559375`。
- 四层结果应分别为 `43/80`、`22/80`、`77/80`、`37/80`。
- 20 个逐任务 JSON 仍全部通过键名、字段和二元值检查。
