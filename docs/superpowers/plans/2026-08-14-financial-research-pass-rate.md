# Financial Research Pass Rate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the personal financial-research evaluator Skill to calculate rubric-level batch pass rates and add the verified summary for the current 20-task MiniMax FinanceGym run.

**Architecture:** Keep each task judgement as an independent 16-key binary JSON object. Add batch-only aggregation that counts `value == 1` across all task files and emits an unweighted `summary.json` with overall and four layer rates. Update the Skill instructions and UI prompt so future batch evaluations reproduce the same contract.

**Tech Stack:** Markdown/YAML Skill files, JSON evaluation artifacts, PowerShell validation, Python `quick_validate.py` with PyYAML from the project virtual environment.

## Global Constraints

- Overall pass rate is `passed_items / total_items`, where `total_items = task_count × 16`.
- Layer rates use the same formula over each four-rubric layer.
- Keep all per-task judgement JSON files unchanged.
- Do not introduce weights, grades, ranks, or “all 16 items must pass” as the default definition.
- Stop rather than summarize if any task JSON lacks exactly 16 IDs or contains a non-integer/non-binary value.

---

### Task 1: Update the Personal Skill Contract

**Files:**
- Modify: `C:/Users/Administrator/.codex/skills/evaluating-financial-research/SKILL.md`
- Modify: `C:/Users/Administrator/.codex/skills/evaluating-financial-research/references/rubric.md`
- Modify: `C:/Users/Administrator/.codex/skills/evaluating-financial-research/agents/openai.yaml`

**Interfaces:**
- Consumes: A directory of per-task judgement JSON objects with IDs `R1–K4` and binary integer `value` fields.
- Produces: Instructions for `overall_pass_rate`, layer pass rates, and the batch `summary.json` schema.

- [ ] **Step 1: Record the failing baseline**

Run:

```powershell
$skill = Get-Content 'C:\Users\Administrator\.codex\skills\evaluating-financial-research\SKILL.md' -Raw -Encoding utf8
[pscustomobject]@{
  ForbidsPercentages = $skill.Contains('不要计算总分、平均分、百分比、等级或排名')
  DefinesOverallPassRate = $skill.Contains('overall_pass_rate')
}
```

Expected: `ForbidsPercentages=True` and `DefinesOverallPassRate=False`, proving the current Skill rejects the requested behavior.

- [ ] **Step 2: Stage the exact Skill changes with `apply_patch`**

Create a workspace staging copy containing these contract changes:

```markdown
overall_pass_rate = passed_items / total_items
total_items = task_count × 16
```

The updated `SKILL.md` must:

- describe batch pass-rate summaries in frontmatter;
- keep each per-task output at exactly 16 binary items;
- require `summary.json` only for batch evaluations;
- prohibit weights, grades, ranks, and alternative implicit scoring;
- require validation before calculating the summary.

The updated `references/rubric.md` must define the overall and layer formulas, error cases, and numeric JSON schema from the approved design.

The updated `agents/openai.yaml` default prompt must request per-task binary judgements plus the unweighted batch pass-rate summary.

- [ ] **Step 3: Install the staged files into the personal Skill directory**

Copy only the three staged files to:

```text
C:\Users\Administrator\.codex\skills\evaluating-financial-research
```

Use an escalated, explicitly scoped copy because this location is outside the workspace write root.

- [ ] **Step 4: Validate the installed Skill**

Run:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe `
  'C:\Users\Administrator\.codex\skills\.system\skill-creator\scripts\quick_validate.py' `
  'C:\Users\Administrator\.codex\skills\evaluating-financial-research'
```

Expected: `Skill is valid!`

Also verify the installed text contains `overall_pass_rate`, `passed_items`, `total_items`, and no longer contains the blanket percentage prohibition.

---

### Task 2: Generate and Verify the Current Batch Summary

**Files:**
- Create: `eval/runs/minimax-financegym20/judgements/summary.json`
- Read: `eval/runs/minimax-financegym20/judgements/*.json`

**Interfaces:**
- Consumes: Exactly 20 per-task judgement files, each containing 16 ordered rubric objects.
- Produces: One JSON object with batch counts and overall/result/process/efficiency/risk pass rates.

- [ ] **Step 1: Verify the precondition and absence of a summary**

Run a read-only check that excludes `summary.json`, parses every per-task JSON file, and confirms:

```text
task_count = 20
rubrics_per_task = 16
all value fields are integer 0 or 1
```

Expected before implementation: the 20 judgement files validate and `summary.json` does not yet exist.

- [ ] **Step 2: Create the minimal summary JSON**

Write exactly this approved structure:

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

- [ ] **Step 3: Recompute and validate every number**

Parse only UUID-named judgement files and independently assert:

```text
sum(all value fields) == summary.passed_items == 179
20 × 16 == summary.total_items == 320
179 / 320 == summary.overall_pass_rate == 0.559375
R == 43 / 80
P == 22 / 80
E == 77 / 80
K == 37 / 80
```

Expected: all assertions pass and no per-task judgement file changes.

- [ ] **Step 4: Clean staging artifacts and inspect final paths**

Remove only the temporary workspace staging copy created in Task 1. Confirm the personal Skill contains its three intended files and the evaluation directory contains 20 per-task JSON files plus `summary.json`.

Generated evaluation artifacts and personal Codex configuration are intentionally not committed to the project repository.
