# Simple CC PDF Fetch Tool Design

**Date:** 2026-08-13

## Goal

Add a focused `pdf_fetch` tool to Simple CC so the conversational agent can
read text-based financial-report PDFs from public HTTP or HTTPS URLs. The tool
must preserve page provenance, support bounded page-range reads, and keep the
existing `web_fetch` behavior unchanged.

## Scope

The first version supports:

- Public PDF URLs discovered through `web_search` or supplied by the user.
- PDFs with an extractable text layer.
- One-based, bounded page-range reads.
- Explicit page boundaries and page numbers in every successful result.
- Text extraction that preserves layout where practical.
- Separate table extraction for financial statements.
- Optional cutoff-date validation using PDF metadata when trustworthy metadata
  is present.

The first version does not support:

- Local PDF files.
- OCR for scanned or image-only PDFs.
- Password entry for encrypted PDFs.
- External PDF parsing services.
- Automatic document-wide indexing or vector retrieval.

## Architecture

Create a focused `simple_cc/pdf_research.py` module. It owns PDF-specific input
validation, bounded download orchestration, PDF parsing, metadata-date handling,
page-range validation, text and table extraction, and structured serialization.

The module reuses the public-URL validation and safe redirect/download behavior
already implemented for `web_fetch`. PDF handling must not introduce a second,
weaker network path. Shared download primitives may be extracted into a small
internal boundary if needed, provided the existing HTML fetch behavior and its
tests remain unchanged.

Register one new fixed tool:

```text
pdf_fetch(url, start_page=1, page_count=10, cutoff=None)
```

The normal agent loop, permission hooks, tool-result persistence, and context
compaction continue to handle the tool without a special execution branch.

## Tool Contract

### Inputs

- `url` is required and must be a public `http` or `https` URL.
- `start_page` is optional, one-based, and defaults to `1`.
- `page_count` is optional and defaults to `10`.
- `page_count` has a fixed upper bound to protect latency and context size. The
  implementation plan will use a single named constant and expose the same
  limit in the tool schema.
- `cutoff` is optional and must use `YYYY-MM-DD` format.

### Successful output

The handler returns deterministic JSON with this shape:

```json
{
  "ok": true,
  "operation": "pdf_fetch",
  "url": "https://example.com/report.pdf",
  "total_pages": 128,
  "start_page": 11,
  "end_page": 20,
  "has_more": true,
  "published_at": null,
  "date_status": "unknown",
  "warning": "PDF publication date could not be verified.",
  "pages": [
    {
      "page_number": 11,
      "content": "--- PAGE 11 START ---\n...\n--- PAGE 11 END ---"
    }
  ]
}
```

Rules:

- `url` is the final validated URL after redirects.
- `end_page` is the last page actually returned.
- `has_more` is true when pages remain after `end_page`.
- `pages` stays ordered and contains one item per requested page.
- Each `content` value starts with `--- PAGE N START ---` and ends with
  `--- PAGE N END ---`.
- Empty pages remain represented so page numbering never shifts.

## PDF Extraction

Use `pdfplumber` as the parsing dependency.

For each requested page:

1. Call `extract_text(layout=True)` to retain column alignment where possible.
2. Extract detected tables separately.
3. Render each non-empty table as tab-separated rows between stable markers:

   ```text
   --- TABLE 1 START ---
   Revenue\t100\t120
   --- TABLE 1 END ---
   ```

4. Append table sections after the page text and inside the page boundary.

The tool does not promise perfect semantic reconstruction of visually complex
tables. It preserves raw page provenance and a stable representation so the
model can reason about the evidence and cite the page number.

If the selected page range contains no extractable text or table cells, return
an `ocr_required` error. This signals that the document is scanned or otherwise
lacks a usable text layer. OCR is explicitly deferred.

## Download and Safety Policy

PDF downloads use the same safety posture as `web_fetch`:

- Accept only public HTTP and HTTPS URLs.
- Reject credentials in URLs, localhost, private IPs, non-routable IPs, and DNS
  results containing unsafe destinations.
- Validate every redirect target.
- Pin the connection to the validated public address and verify the peer.
- Enforce redirect, timeout, and response-byte limits.
- Return structured errors rather than raising through the agent loop.

The downloader accepts `application/pdf` and may also accept a generic content
type only when the body begins with a valid PDF signature. HTML error pages or
other non-PDF bodies are rejected as `not_pdf`.

The PDF byte limit may be larger than the HTML limit because financial reports
are commonly larger, but it must remain bounded through a named constant.

## Cutoff-Date Policy

PDF publication dates are often absent or unreliable. When `cutoff` is present:

- Parse explicit PDF metadata dates conservatively.
- If a single trustworthy date is confirmed and it is later than the cutoff,
  reject the document with `published_after_cutoff`.
- If metadata dates conflict, return `date_conflict`.
- If no trustworthy date is available, allow extraction but return
  `date_status: "unknown"` and an explicit warning.
- Never infer publication date from the live retrieval time.

This remains non-strict point-in-time research, consistent with the existing
web tools.

## Error Contract

Failures use the existing structured pattern:

```json
{
  "ok": false,
  "operation": "pdf_fetch",
  "error": {
    "code": "ocr_required",
    "message": "PDF has no extractable text in the requested page range"
  }
}
```

Required error categories are:

- `invalid_cutoff`
- `invalid_start_page`
- `invalid_page_count`
- `page_out_of_range`
- `unsafe_url`
- `dns_failed`
- `fetch_failed`
- `too_many_redirects`
- `response_too_large`
- `not_pdf`
- `invalid_pdf`
- `encrypted_pdf`
- `ocr_required`
- `date_conflict`
- `published_after_cutoff`

Messages should be actionable but must not include PDF bytes or sensitive
transport details.

## Agent Guidance

Update the system prompt and tool description so the model follows this flow:

1. Use `web_search` to find sources when the user has not supplied a PDF URL.
2. Call `pdf_fetch` for a PDF URL instead of `web_fetch`.
3. Start with the most relevant bounded page range.
4. Continue with later ranges when `has_more` is true and more evidence is
   needed.
5. Cite PDF evidence by URL and page number.
6. Preserve the same `cutoff` value across search and PDF reads.
7. Disclose unknown publication dates and the lack of OCR support.

## Compatibility

`web_fetch` remains an HTML and plain-text tool. Its schema, output contract,
and existing tests must not change as a side effect of adding PDF support.

The new dependency is added to both `pyproject.toml` and `requirements.txt`.
Tool registration remains fixed through `TOOL_DEFINITIONS` and
`TOOL_HANDLERS`, matching the current codebase.

## Testing

Tests remain offline and deterministic. Generate minimal fixtures in memory or
under pytest temporary directories rather than relying on live reports.

Coverage includes:

- Total pages, one-based page numbers, stable page boundaries, and continuation
  through `has_more`.
- Table markers and extracted cell values.
- Default and explicit page ranges.
- Invalid and out-of-range pagination inputs.
- Corrupt and encrypted PDFs.
- Image-only PDFs returning `ocr_required`.
- Non-PDF bodies and misleading content types.
- Existing private-network, redirect, timeout, and byte-limit protections.
- Cutoff rejection, conflicting metadata dates, and unknown-date warnings.
- Tool schema and handler registration.
- System-prompt guidance.
- One agent-loop test in which the model calls `pdf_fetch`, receives a paged
  result, and produces an answer with a page citation.
- The existing web-research tests and the complete test suite to catch
  regressions.

## Acceptance Criteria

The feature is complete when:

- A public, text-based financial-report PDF can be read through `pdf_fetch`.
- Results preserve correct page numbers and explicit page boundaries.
- Long reports can be traversed through bounded page-range calls.
- Extracted tables retain stable row and cell separation.
- Scanned PDFs fail clearly with `ocr_required`.
- Unsafe destinations and oversized or invalid downloads remain blocked.
- Existing `web_search` and `web_fetch` behavior remains intact.
- PDF-specific and full-project tests pass.
