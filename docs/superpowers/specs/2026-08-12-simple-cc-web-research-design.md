# Simple CC Web Research Tools Design

**Date:** 2026-08-12  
**Status:** Approved for implementation

## Goal

Add minimal web-research capability to the existing `simple_cc` conversational
agent so it can act as a fair baseline in an internal FinanceGym-style
evaluation. The user continues to ask questions through the existing dialog;
the model may call search and page-reading tools from the normal agent loop.

This change adds retrieval capability only. It does not add financial skills,
specialized planning, report critique, multi-agent research orchestration, or a
rubric-aware prompt.

## Scope

The first version provides two tools:

- `web_search(query, max_results=5, cutoff=None)` returns compact search hits.
- `web_fetch(url, cutoff=None)` downloads one page, extracts readable text and
  publication-date evidence, and applies the cutoff policy.

The initial backend uses live public web search and page fetching. It is a
lightweight, non-strict PIT implementation: a page with an identified
publication date later than `cutoff` is rejected, while a page whose date
cannot be established is returned with an explicit `date_status=unknown`
warning. Formal internal benchmark runs may either reject unknown-date pages at
the runner layer or later replace this backend with a frozen local PIT corpus.

## Architecture

### Research module

Create `simple_cc/web_research.py` as the only module that knows about HTTP,
search-provider details, HTML parsing, publication-date extraction, and cutoff
comparison. It exposes synchronous handlers compatible with Simple CC's fixed
tool registry.

Search results use a stable JSON representation containing `title`, `url`, and
`snippet`. Fetched pages use a stable JSON representation containing `url`,
`title`, `published_at`, `date_status`, `content`, and `truncated`. Returning
JSON rather than prose keeps tool output deterministic and makes later
trajectory analysis possible.

The backend boundary is kept small so a local PIT implementation can replace
the live implementation without changing tool names or prompts.

### Tool registration

Modify `simple_cc/tools.py` to add both tool schemas to `TOOL_DEFINITIONS` and
both handlers to `TOOL_HANDLERS`. They participate in the existing fixed
registry, permission path, tool-call history, compaction, and dialog loop.

No separate UI or command is introduced. The tools appear in the same model
tool list used by an ordinary `python -m simple_cc` conversation.

### Prompt behavior

Modify `simple_cc/prompts.py` to list the new tools and provide narrow research
rules:

- Search before making claims that require current or external evidence.
- When the user supplies a cutoff date, pass it to every search and fetch call.
- Do not use evidence explicitly dated after the cutoff.
- Treat unknown publication dates as uncertain and disclose that limitation.
- Include source URLs in research answers.

These rules do not prescribe a finance workflow or force a particular report
outline, preserving `simple_cc` as the baseline harness.

## Cutoff Policy

`cutoff` is optional for normal dialog use and must be an ISO date
(`YYYY-MM-DD`) when supplied.

For `web_search`, the live provider cannot guarantee historical visibility.
The cutoff is carried in the request/result metadata and included in the search
query as a date constraint, but search results remain candidates rather than
trusted PIT evidence.

For `web_fetch`:

1. Extract candidate dates from structured metadata first (`article:published_time`,
   schema.org/JSON-LD, and `<time datetime>`), then conservative visible-page
   patterns.
2. If a reliable date is later than the cutoff, return a structured rejection
   and no article content.
3. If a date is on or before the cutoff, return the extracted content with
   `date_status=verified`.
4. If no reliable date is found, return content with `date_status=unknown` and
   an explicit warning. This preserves usability while making the limitation
   measurable.

This policy is deliberately labeled non-strict PIT. It must not be described
as equivalent to FinanceGym's frozen corpus.

## Safety and Limits

- Only `http` and `https` URLs are accepted.
- Loopback, link-local, private-network, and non-routable targets are rejected
  before fetching to reduce SSRF risk.
- Redirect targets are checked with the same policy.
- Response size, extracted-text size, redirect count, and request timeout are
  bounded.
- Binary or unsupported content types are rejected.
- User-agent identification is explicit.
- Tool errors are returned as structured JSON and do not crash the agent loop.

## Dependencies

Use small Python dependencies suitable for the existing package:

- `ddgs` for public web search.
- `httpx` for bounded HTTP requests and redirect inspection.
- `trafilatura` for main-text and metadata extraction.

Dependencies are recorded in both `pyproject.toml` and `requirements.txt` to
match the repository's existing installation paths.

## Testing

All automated tests are offline. They inject or monkeypatch search and HTTP
clients; no test contacts the public internet.

Coverage includes:

- Tool definitions and handler registration.
- Search result normalization and maximum-result bounds.
- Valid cutoff parsing and invalid-cutoff errors.
- Verified pre-cutoff page acceptance.
- Post-cutoff page rejection.
- Unknown-date warning behavior.
- HTML text extraction and output truncation.
- Rejection of localhost, private IPs, non-HTTP URLs, redirects to private
  addresses, and unsupported content types.
- Prompt exposure and instructions for cutoff propagation and citations.
- One agent-loop test demonstrating that the model can call a research tool
  from the normal dialog and receive its result.

## Evaluation Boundary

This feature makes `simple_cc` usable for retrieval-enabled smoke tests and
lightweight internal comparisons. A valid Harness comparison must still fix
the model, questions, prompt envelope, retrieval backend, cutoff policy, tool
budget, run count, and scoring procedure across all systems. Results produced
with the live backend must be labeled `internal preview / non-strict PIT` and
must not be compared directly with the FinanceGym leaderboard or the paper's
reported scores.

