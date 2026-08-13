from __future__ import annotations

import json
import re
from datetime import date, datetime
from io import BytesIO
from typing import Any
from urllib.parse import urljoin

from . import web_research
from .telemetry import capture_tool_artifact


PDF_PAGE_COUNT_DEFAULT = 10
PDF_PAGE_COUNT_MAX = 20
PDF_MAX_RESPONSE_BYTES = 25_000_000
PDF_ACCEPT = "application/pdf,application/octet-stream;q=0.8"
PDF_MAX_REDIRECTS = 5
PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/octet-stream",
    "binary/octet-stream",
    "",
}


class PDFInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _error(code: str, message: str, **details: Any) -> str:
    return _json(
        {
            "ok": False,
            "operation": "pdf_fetch",
            **details,
            "error": {"code": code, "message": message},
        }
    )


def _parse_pagination(start_page: Any, page_count: Any) -> tuple[int, int]:
    if isinstance(start_page, bool) or not isinstance(start_page, int):
        raise PDFInputError(
            "invalid_start_page", "start_page must be an integer"
        )
    if start_page < 1:
        raise PDFInputError(
            "invalid_start_page", "start_page must be at least 1"
        )
    if isinstance(page_count, bool) or not isinstance(page_count, int):
        raise PDFInputError(
            "invalid_page_count", "page_count must be an integer"
        )
    if page_count < 1 or page_count > PDF_PAGE_COUNT_MAX:
        raise PDFInputError(
            "invalid_page_count",
            f"page_count must be from 1 to {PDF_PAGE_COUNT_MAX}",
        )
    return start_page, page_count


def _fetch_pdf(url: str) -> tuple[str, bytes]:
    current = url
    for redirect_count in range(PDF_MAX_REDIRECTS + 1):
        addresses = sorted(web_research._validate_public_url(current))
        status, headers, body = web_research._fetch_once(
            current,
            addresses[0],
            max_response_bytes=PDF_MAX_RESPONSE_BYTES,
            accept=PDF_ACCEPT,
        )
        if status in {301, 302, 303, 307, 308}:
            location = headers.get("location")
            if not location:
                raise PDFInputError(
                    "fetch_failed", "redirect response has no location"
                )
            if redirect_count >= PDF_MAX_REDIRECTS:
                raise PDFInputError(
                    "too_many_redirects", "redirect limit exceeded"
                )
            current = urljoin(current, location)
            continue
        if status < 200 or status >= 300:
            raise PDFInputError("fetch_failed", f"HTTP status {status}")
        content_type = headers.get("content-type", "").split(";", 1)[0]
        content_type = content_type.strip().lower()
        if content_type not in PDF_CONTENT_TYPES:
            raise PDFInputError(
                "not_pdf", "response content type is not a supported PDF type"
            )
        if b"%PDF-" not in body[:1024]:
            raise PDFInputError(
                "not_pdf", "response body does not contain a PDF signature"
            )
        return current, body
    raise PDFInputError("fetch_failed", "request did not produce a response")


def _open_pdf(data: bytes):
    try:
        import pdfplumber
        from pdfminer.pdfdocument import PDFPasswordIncorrect

        return pdfplumber.open(BytesIO(data))
    except PDFPasswordIncorrect as error:
        raise PDFInputError(
            "encrypted_pdf", "PDF is encrypted and cannot be opened"
        ) from error
    except PDFInputError:
        raise
    except Exception as error:
        raise PDFInputError("invalid_pdf", "PDF could not be parsed") from error


def _normalize_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _extract_metadata_dates(
    metadata: dict[str, Any] | None,
) -> list[date]:
    explicit_keys = {
        "publicationdate",
        "published",
        "publication_date",
        "dc:date",
    }
    dates: list[date] = []
    for key, value in (metadata or {}).items():
        if str(key).lower() not in explicit_keys or not value:
            continue
        match = re.search(r"D:(\d{4})(\d{2})(\d{2})", str(value))
        if match:
            candidate = "-".join(match.groups())
        else:
            iso_match = re.search(
                r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", str(value)
            )
            if not iso_match:
                continue
            candidate = iso_match.group(1)
        try:
            parsed = datetime.strptime(candidate, "%Y-%m-%d").date()
        except ValueError:
            continue
        if parsed not in dates:
            dates.append(parsed)
    return dates


def _format_page(page: Any, page_number: int) -> tuple[str, bool]:
    text = (page.extract_text(layout=True) or "").strip()
    sections = [f"--- PAGE {page_number} START ---"]
    if text:
        sections.append(text)
    has_content = bool(text)
    table_number = 0
    for table in page.extract_tables() or []:
        rows = [
            "\t".join(_normalize_cell(cell) for cell in row or [])
            for row in table or []
        ]
        if not any(row.strip("\t ") for row in rows):
            continue
        table_number += 1
        has_content = True
        sections.extend(
            [
                f"--- TABLE {table_number} START ---",
                *rows,
                f"--- TABLE {table_number} END ---",
            ]
        )
    sections.append(f"--- PAGE {page_number} END ---")
    return "\n".join(sections), has_content


def pdf_fetch(
    url: str,
    start_page: int = 1,
    page_count: int = PDF_PAGE_COUNT_DEFAULT,
    cutoff: str | None = None,
) -> str:
    try:
        cutoff_date = web_research._parse_cutoff(cutoff)
        _parse_pagination(start_page, page_count)
        final_url, data = _fetch_pdf(url)
    except (PDFInputError, web_research.ResearchInputError) as error:
        return _error(error.code, str(error))
    except Exception:
        return _error("fetch_failed", "PDF download failed")

    pdf = None
    try:
        pdf = _open_pdf(data)
        total_pages = len(pdf.pages)
        publication_dates = _extract_metadata_dates(pdf.metadata)
        if len(publication_dates) > 1:
            return _error(
                "date_conflict",
                "PDF has conflicting explicit publication dates",
                url=final_url,
                cutoff=cutoff_date.isoformat() if cutoff_date else None,
                publication_dates=[
                    item.isoformat() for item in publication_dates
                ],
            )
        published_at = publication_dates[0] if publication_dates else None
        if (
            cutoff_date is not None
            and published_at is not None
            and published_at > cutoff_date
        ):
            return _error(
                "published_after_cutoff",
                "PDF publication date is later than cutoff",
                url=final_url,
                cutoff=cutoff_date.isoformat(),
                published_at=published_at.isoformat(),
            )
        if start_page > total_pages:
            return _error(
                "page_out_of_range",
                "start_page is greater than the PDF page count",
                url=final_url,
                total_pages=total_pages,
            )
        end_page = min(total_pages, start_page + page_count - 1)
        pages = []
        has_extractable_content = False
        for page_number in range(start_page, end_page + 1):
            content, page_has_content = _format_page(
                pdf.pages[page_number - 1], page_number
            )
            has_extractable_content = (
                has_extractable_content or page_has_content
            )
            pages.append(
                {"page_number": page_number, "content": content}
            )
        if not has_extractable_content:
            return _error(
                "ocr_required",
                "PDF has no extractable text in the requested page range",
                url=final_url,
                total_pages=total_pages,
            )
        capture_tool_artifact(
            data,
            media_type="application/pdf",
            source=final_url,
            suffix=".pdf",
        )
        capture_tool_artifact(
            pages,
            media_type="application/json",
            source=f"{final_url}#pages={start_page}-{end_page}",
            suffix=".json",
        )
        payload: dict[str, Any] = {
            "ok": True,
            "operation": "pdf_fetch",
            "url": final_url,
            "total_pages": total_pages,
            "start_page": start_page,
            "end_page": end_page,
            "has_more": end_page < total_pages,
            "published_at": (
                published_at.isoformat() if published_at else None
            ),
            "date_status": "verified" if published_at else "unknown",
            "cutoff": cutoff_date.isoformat() if cutoff_date else None,
            "pages": pages,
        }
        if published_at is None:
            payload["warning"] = (
                "PDF publication date could not be verified."
            )
        return _json(payload)
    except PDFInputError as error:
        return _error(error.code, str(error), url=final_url)
    except Exception:
        return _error("invalid_pdf", "PDF could not be parsed", url=final_url)
    finally:
        if pdf is not None:
            pdf.close()
