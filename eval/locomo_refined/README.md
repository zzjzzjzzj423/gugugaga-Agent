# Gugugaga LoCoMo-Refined smoke benchmark

The frozen 10-conversation, 200-question Memory/No-Memory/Gold-Evidence-Oracle
evaluation and failure analysis is documented in
[`REPORT_200_ORACLE.md`](REPORT_200_ORACLE.md).

This adapter replays MiniMem's LoCoMo-Refined data through Gugugaga's current
memory implementation. It intentionally uses the production Retrieval Gate and
SQLite FTS5/BM25 recall. If `GUGUGAGA_MEMORY_EMBEDDING_MODEL` is configured, it
also builds the production SQLite vector index and evaluates hybrid RRF recall;
it does not install MiniMem's embedding model.

## Scope

- one isolated SQLite database per selected conversation;
- chronological replay with original speaker names and session timestamps;
- an odd final message is retained as an explicitly counted partial exchange
  with an empty assistant side; no reply text is invented and sessions are
  never paired across their time boundary;
- normal consolidation every configured 6 exchanges;
- up to three immediate benchmark retries when strict consolidation output is
  invalid or a provider call fails (production backoff behavior is unchanged);
- one explicit final flush for the sub-threshold tail;
- Outbox vector indexing before QA when an embedding model is configured;
- semantic diversity filtering and same-Exchange chat expansion before prompt injection;
- 20 questions by default, sampled deterministically across four categories;
- Current Memory and No Memory answer variants;
- shortest-answer generation with one conservative inference for likely/would questions;
- local Token F1, BLEU-1, Gate and recall diagnostics.

This smoke test does not run the official LLM judge and does not compute
evidence-level Recall@K/MRR.

## Data

Download the two data files vendored by MiniMem:

```powershell
python eval/locomo_refined/download_data.py
```

The command creates:

```text
eval/locomo_refined/data/conversations.jsonl
eval/locomo_refined/data/questions.jsonl
```

Existing files are never overwritten.

Validate the real dataset and show the deterministic 20-question selection
without making any model calls:

```powershell
python eval/locomo_refined/run_smoke.py --validate-only
```

## Run

The runner uses the normal `SILICONFLOW_*` and `GUGUGAGA_MEMORY_*` settings from
the repository `.env` file.

```powershell
python eval/locomo_refined/run_smoke.py
```

Useful overrides:

```powershell
python eval/locomo_refined/run_smoke.py --sample-id <sample_id> --max-questions 20
python eval/locomo_refined/run_smoke.py --output-dir eval/locomo_refined/runs/manual
python eval/locomo_refined/run_smoke.py --max-consolidation-attempts 3
python eval/locomo_refined/run_smoke.py --model Qwen/Qwen3.6-35B-A3B --disable-thinking --temperature 0
```

Use `--disable-thinking` for dual-mode reasoning models when the benchmark
expects short answers and exact consolidation JSON. This prevents the model
from exhausting the output budget on hidden reasoning before emitting the
required response body.

For a three-run stability check, use a fresh output directory for every run and
then aggregate the summaries:

```powershell
$env:GUGUGAGA_MEMORY_CONSOLIDATION_MODEL = "Qwen/Qwen3.6-35B-A3B"
$repeatTag = Get-Date -Format "yyyyMMdd-HHmmss"
$repeatRoot = "eval/locomo_refined/runs/qwen36-t0-repeat-$repeatTag"
1..3 | ForEach-Object {
  python eval/locomo_refined/run_smoke.py `
    --sample-id conv-26 `
    --max-questions 20 `
    --questions-per-category 5 `
    --model "Qwen/Qwen3.6-35B-A3B" `
    --disable-thinking `
    --temperature 0 `
    --output-dir "$repeatRoot/run-$_"
  if ($LASTEXITCODE -ne 0) { throw "repeat run $_ failed" }
}
python eval/locomo_refined/summarize_repeats.py `
  "$repeatRoot/run-1" "$repeatRoot/run-2" "$repeatRoot/run-3" `
  --output "$repeatRoot/repeat-summary.json"
```

The aggregator rejects runs whose sample, question count, retrieval mode, or
recorded model/sampling configuration differs.

## Diagnose retrieval versus answering

Run post-hoc diagnostics against an existing run. This reuses the stored
retrieval results and database; it does not rebuild memory. It makes one extra
Oracle Evidence answer call per question:

```powershell
python eval/locomo_refined/diagnose_run.py `
  eval/locomo_refined/runs/qwen36-t0-repeat-20260904-183659/run-2
```

The command writes `diagnostics.json` and
`predictions_oracle_evidence.json` into the selected run directory. Evidence
Recall@K is measured by mapping LoCoMo evidence message IDs to the source turn
IDs already recorded in the rendered memory. A consolidated fact references
its whole source batch, so this source-level metric is intentionally generous.

Each run writes the isolated database, both prediction variants, per-question
diagnostics, and `summary.json` under `eval/locomo_refined/runs/<run_id>/`.
An existing evaluation database is never reused, preventing memory from a prior
run from contaminating the score.

If a run is interrupted, resume it explicitly. Existing turn IDs are validated
and skipped, so completed consolidation work is not repeated:

```powershell
python eval/locomo_refined/run_smoke.py --resume-output-dir eval/locomo_refined/runs/<run_id>
```
