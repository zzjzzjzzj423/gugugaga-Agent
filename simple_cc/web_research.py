from __future__ import annotations

import json
import ipaddress
import http.client
import re
import socket
from datetime import date, datetime, timedelta
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlsplit

from .telemetry import capture_tool_artifact


SEARCH_LIMIT_DEFAULT = 5
SEARCH_LIMIT_MAX = 10
FETCH_TIMEOUT_S = 20.0
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 2_000_000
MAX_CONTENT_CHARS = 40_000
USER_AGENT = "SimpleCC-Research/0.1"
PIT_MODE = "non_strict_live_web"
PIT_WARNING = (
    "Live search cannot guarantee historical visibility; fetch and verify "
    "publication dates before using results as evidence."
)


class ResearchInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _error(operation: str, code: str, message: str) -> str:
    return _json(
        {
            "ok": False,
            "operation": operation,
            "error": {"code": code, "message": message},
        }
    )


def _parse_cutoff(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as error:
        raise ResearchInputError(
            "invalid_cutoff", "cutoff must use YYYY-MM-DD format"
        ) from error
    if parsed.isoformat() != value:
        raise ResearchInputError(
            "invalid_cutoff", "cutoff must use YYYY-MM-DD format"
        )
    return parsed


def _search_rows(query: str, limit: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    return list(
        DDGS(timeout=10).text(
            query,
            max_results=limit,
            backend="brave",
        )
    )


def _validate_public_url(url: str) -> set[str]:
    parsed = urlsplit(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ResearchInputError(
            "unsafe_url", "only public http and https URLs are allowed"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ResearchInputError("unsafe_url", "URL credentials are blocked")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ResearchInputError("unsafe_url", "local destinations are blocked")
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            rows = socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as error:
            raise ResearchInputError(
                "dns_failed", f"could not resolve destination: {error}"
            ) from error
        addresses = []
        for row in rows:
            address = row[4][0]
            parsed_address = ipaddress.ip_address(address.split("%")[0])
            if parsed_address not in addresses:
                addresses.append(parsed_address)
    if not addresses or any(not address.is_global for address in addresses):
        raise ResearchInputError(
            "unsafe_url", "private or non-routable destinations are blocked"
        )
    return {str(address) for address in addresses}


def _fetch_once(
    url: str,
    address: str,
    *,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    accept: str = "text/html,text/plain",
) -> tuple[int, dict[str, str], bytes]:
    parsed = urlsplit(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_class = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_class(parsed.hostname, port, timeout=FETCH_TIMEOUT_S)
    original_create_connection = socket.create_connection

    def connect_to_pinned_address(
        target: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
        **kwargs: Any,
    ) -> socket.socket:
        return original_create_connection(
            (address, target[1]),
            timeout,
            source_address,
            **kwargs,
        )

    connection._create_connection = connect_to_pinned_address
    request_target = parsed.path or "/"
    if parsed.query:
        request_target += f"?{parsed.query}"
    host_header = parsed.hostname
    if parsed.port is not None and parsed.port != (443 if parsed.scheme == "https" else 80):
        host_header += f":{parsed.port}"
    try:
        connection.request(
            "GET",
            request_target,
            headers={
                "Host": host_header,
                "User-Agent": USER_AGENT,
                "Accept": accept,
                "Connection": "close",
            },
        )
        peer_address = ipaddress.ip_address(
            str(connection.sock.getpeername()[0]).split("%")[0]
        )
        if not peer_address.is_global or str(peer_address) != address:
            raise ResearchInputError(
                "unsafe_url", "connected peer did not match pinned public address"
            )
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        body = response.read(max_response_bytes + 1)
        if len(body) > max_response_bytes:
            raise ResearchInputError(
                "response_too_large", "response exceeded byte limit"
            )
        return response.status, headers, body
    finally:
        connection.close()


def _fetch_html(url: str) -> tuple[str, str, str]:
    current = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        allowed_addresses = sorted(_validate_public_url(current))
        status, headers, body = _fetch_once(current, allowed_addresses[0])
        if status in {301, 302, 303, 307, 308}:
            location = headers.get("location")
            if not location:
                raise ResearchInputError(
                    "fetch_failed", "redirect response has no location"
                )
            if redirect_count >= MAX_REDIRECTS:
                raise ResearchInputError(
                    "too_many_redirects", "redirect limit exceeded"
                )
            current = urljoin(current, location)
            _validate_public_url(current)
            continue
        if status < 200 or status >= 300:
            raise ResearchInputError("fetch_failed", f"HTTP status {status}")
        content_type = headers.get("content-type", "").lower()
        if not (
            content_type.startswith("text/html")
            or content_type.startswith("text/plain")
        ):
            raise ResearchInputError(
                "unsupported_content_type",
                f"unsupported content type: {content_type or 'unknown'}",
            )
        encoding_match = re.search(r"charset=([^;\s]+)", content_type, re.I)
        encoding = encoding_match.group(1).strip('"\'') if encoding_match else "utf-8"
        return current, body.decode(encoding, errors="replace"), content_type
    raise ResearchInputError("fetch_failed", "request did not produce a response")


def _clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value, flags=re.DOTALL)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    return _clean_text(match.group(1)) if match else ""


def _parse_date_candidate(value: str | None) -> date | None:
    if not value:
        return None
    match = re.search(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", str(value))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _extract_publication_dates(html: str) -> list[date]:
    meta_patterns = [
        r"<meta[^>]+(?:property|name)=[\"'](?:article:published_time|datePublished)[\"'][^>]+content=[\"']([^\"']+)",
        r"<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+(?:property|name)=[\"'](?:article:published_time|datePublished)[\"']",
        r"[\"']datePublished[\"']\s*:\s*[\"']([^\"']+)[\"']",
    ]
    dates: list[date] = []
    for pattern in meta_patterns:
        for candidate in re.findall(pattern, html, re.I | re.S):
            parsed = _parse_date_candidate(candidate)
            if parsed is not None and parsed not in dates:
                dates.append(parsed)
    return dates


def _extract_content(html: str, content_type: str) -> str:
    if content_type.startswith("text/plain"):
        return re.sub(r"\s+", " ", html).strip()
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if extracted:
            return extracted.strip()
    except ImportError:
        pass
    body = re.search(r"<body[^>]*>(.*?)</body>", html, re.I | re.S)
    return _clean_text(body.group(1) if body else html)


def web_search(
    query: str,
    max_results: int = SEARCH_LIMIT_DEFAULT,
    cutoff: str | None = None,
) -> str:
    operation = "search"
    try:
        cutoff_date = _parse_cutoff(cutoff)
    except ResearchInputError as error:
        return _error(operation, error.code, str(error))

    normalized_query = str(query or "").strip()
    if not normalized_query:
        return _error(operation, "invalid_query", "query must not be empty")
    try:
        limit = min(max(int(max_results), 1), SEARCH_LIMIT_MAX)
    except (TypeError, ValueError):
        return _error(
            operation,
            "invalid_max_results",
            "max_results must be an integer from 1 to 10",
        )

    provider_query = normalized_query
    if cutoff_date is not None:
        provider_query += f" before:{(cutoff_date + timedelta(days=1)).isoformat()}"

    try:
        rows = _search_rows(provider_query, limit)
    except Exception as error:
        return _error(operation, "search_failed", f"{type(error).__name__}: {error}")

    results: list[dict[str, str]] = []
    for row in rows:
        url = str(row.get("href") or row.get("url") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": str(row.get("title") or "").strip(),
                "url": url,
                "snippet": str(
                    row.get("body") or row.get("snippet") or ""
                ).strip(),
            }
        )
        if len(results) >= limit:
            break

    return _json(
        {
            "ok": True,
            "operation": operation,
            "query": provider_query,
            "cutoff": cutoff_date.isoformat() if cutoff_date else None,
            "pit_mode": PIT_MODE,
            "warning": PIT_WARNING,
            "results": results,
        }
    )


def web_fetch(url: str, cutoff: str | None = None) -> str:
    operation = "fetch"
    try:
        cutoff_date = _parse_cutoff(cutoff)
        _validate_public_url(url)
        final_url, html, content_type = _fetch_html(url)
        _validate_public_url(final_url)
    except ResearchInputError as error:
        return _error(operation, error.code, str(error))
    except Exception as error:
        return _error(operation, "fetch_failed", f"{type(error).__name__}: {error}")

    publication_dates = _extract_publication_dates(html)
    if len(publication_dates) > 1:
        return _json(
            {
                "ok": False,
                "operation": operation,
                "url": final_url,
                "cutoff": cutoff_date.isoformat() if cutoff_date else None,
                "publication_dates": [
                    item.isoformat() for item in publication_dates
                ],
                "error": {
                    "code": "date_conflict",
                    "message": "page has conflicting explicit publication dates",
                },
            }
        )
    published_at = publication_dates[0] if publication_dates else None
    if (
        cutoff_date is not None
        and published_at is not None
        and published_at > cutoff_date
    ):
        return _json(
            {
                "ok": False,
                "operation": operation,
                "url": final_url,
                "cutoff": cutoff_date.isoformat(),
                "published_at": published_at.isoformat(),
                "error": {
                    "code": "post_cutoff",
                    "message": "page publication date is later than cutoff",
                },
            }
        )

    capture_tool_artifact(
        html,
        media_type=content_type.split(";", 1)[0],
        source=final_url,
        suffix=".txt" if content_type.startswith("text/plain") else ".html",
    )
    content = _extract_content(html, content_type)
    original_content_chars = len(content)
    truncated = len(content) > MAX_CONTENT_CHARS
    content = content[:MAX_CONTENT_CHARS]
    date_status = "verified" if published_at is not None else "unknown"
    payload: dict[str, Any] = {
        "ok": True,
        "operation": operation,
        "url": final_url,
        "title": _extract_title(html),
        "cutoff": cutoff_date.isoformat() if cutoff_date else None,
        "published_at": published_at.isoformat() if published_at else None,
        "date_status": date_status,
        "pit_mode": PIT_MODE,
        "content": content,
        "content_chars_original": original_content_chars,
        "content_chars_visible": len(content),
        "truncated": truncated,
    }
    if published_at is None:
        payload["warning"] = (
            "Publication date could not be verified; treat this page as "
            "uncertain evidence and disclose the limitation."
        )
    return _json(payload)
