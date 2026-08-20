# Hardening D Report

Base: `acf41442f24e3ea8dced7f8678b559a9e4fff0d3`

Commit message: `fix: preserve canonical citation identities`

## Result

Hardening D closes the two citation-identity findings from the hardened
whole-branch review without changing the user's accepted limitation for a
literal embedded `http(s)://` inside one URL path or query.

### Escape-aware citation structure

- Markdown and angle delimiters are structural only when preceded by an even
  number of consecutive backslashes.
- Markdown label delimiters and the URL-closing `)` are checked independently;
  angle `<` and `>` delimiters use the same parity rule.
- Escaped openers or closers leave the complete URL token to fail closed. They
  cannot turn an unfetched suffix into a match for a registered URL prefix.
- Even-backslash and ordinary unescaped Markdown/angle citations remain valid,
  including registered URL paths that themselves end in two backslashes.

### Strict query identity

- Query percent escapes must be complete hexadecimal triplets. Lone `%`,
  `%G0`, and `%0G` forms are rejected before decoding.
- Percent-decoded query components use strict UTF-8. Invalid octets such as
  `%FF` and `%FE` are rejected instead of both becoming the replacement
  character `%EF%BF%BD`.
- Valid UTF-8 escapes retain the existing deterministic query sorting and
  encoding behavior.
- Rejected fetch identities emit `source_rejected` with `invalid_url`, create
  no registry record, consume no distinct-source quota, and produce distinct
  bounded unmatched-citation digests during final-answer linkage.

## TDD Evidence

Initial focused RED command selected malformed/non-UTF-8 canonicalization,
linkage, escaped opener/closer, and even-backslash control cases. Result:
`8 failed, 2 passed, 107 deselected`. Failures showed replacement decoding and
registered-prefix matches through escaped delimiters.

The registration RED first encountered the known Windows temporary-directory
ACL error and was rerun with a fresh external basetemp and elevated access.
The valid behavioral RED result was `2 failed, 117 deselected`; both `%FF` and
`%FE` were registered as `https://example.com/report?q=%EF%BF%BD` with the same
`src_9e3084caca3ac490` identity.

Focused GREEN:

`python -m pytest -q tests/test_evidence_trace.py -k "canonical_url_rejects_malformed_or_non_utf8_query or canonical_url_keeps_valid_utf8_query_sorting or non_utf8_query_citations_remain_distinct_and_unmatched or escaped_citation_openers or escaped_citation_closers or even_backslashes_keep or non_utf8_query_fetch_is_rejected_without_consuming_registry_quota" --basetemp E:\AgentLearnProject\simple_cc\.simple_cc\pytest-sdd-hardening-d-green-20260820a`

Result: `12 passed, 107 deselected in 1.07s`.

Evidence regression:

`python -m pytest -q tests/test_evidence_trace.py --basetemp E:\AgentLearnProject\simple_cc\.simple_cc\pytest-sdd-hardening-d-evidence-20260820a`

Result: `119 passed in 3.14s`.

Affected regression:

`python -m pytest -q tests/test_evidence_trace.py tests/test_research_models.py tests/test_research_workflow.py tests/test_benchmark_worker.py --basetemp E:\AgentLearnProject\simple_cc\.simple_cc\pytest-sdd-hardening-d-affected-20260820a`

Result: `225 passed in 6.36s`.

Full regression:

`python -m pytest -q --deselect tests/test_source_audit.py::test_source_map_pins_baseline_and_classifies_every_top_level_source_block --basetemp E:\AgentLearnProject\simple_cc\.simple_cc\pytest-sdd-hardening-d-full-20260820a`

Result: `435 passed, 1 deselected in 13.44s`. The sole deselection is the
previously approved isolated-worktree sibling-path audit.

`python -m compileall -q simple_cc eval tests` and `git diff --check` also
passed.

## Changed Files

- `simple_cc/evidence.py`
- `tests/test_evidence_trace.py`
- `.superpowers/sdd/2026-08-19-routed-research-workflow/hardening-d-report.md`

No production file outside `simple_cc/evidence.py` was changed.

## Remaining Explicit Decision

Per user direction, the citation scanner still does not disambiguate a literal
embedded `http(s)://` within a single URL path or query from adjacent
citations. Hardening D does not alter that behavior.
