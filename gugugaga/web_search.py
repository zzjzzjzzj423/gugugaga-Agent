from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_SEARCH_DEPTHS = {"basic", "advanced"}
_TOPICS = {"general", "news", "finance"}


def _error(code: str, message: str) -> str:
    return json.dumps(
        {"ok": False, "error_code": code, "message": message},
        ensure_ascii=False,
    )


def run_web_search(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    search_depth: str = "basic",
) -> str:
    """Search the public web through Tavily and return bounded source snippets."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return _error(
            "tavily_not_configured",
            "Tavily API Key is not configured. Open Settings in the Web Console.",
        )
    if not isinstance(query, str) or not query.strip():
        return _error("invalid_query", "query is required")
    clean_query = query.strip()
    if len(clean_query) > 400:
        return _error("invalid_query", "query must contain at most 400 characters")
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        return _error("invalid_max_results", "max_results must be an integer")
    if not 1 <= max_results <= 10:
        return _error("invalid_max_results", "max_results must be between 1 and 10")
    if topic not in _TOPICS:
        return _error("invalid_topic", "topic must be general, news, or finance")
    if search_depth not in _SEARCH_DEPTHS:
        return _error("invalid_search_depth", "search_depth must be basic or advanced")

    payload = json.dumps(
        {
            "query": clean_query,
            "max_results": max_results,
            "topic": topic,
            "search_depth": search_depth,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
    ).encode("utf-8")
    request = Request(
        TAVILY_SEARCH_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "gugugaga/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        code = {
            401: "tavily_auth_failed",
            429: "tavily_rate_limited",
        }.get(error.code, "tavily_request_failed")
        return _error(code, f"Tavily request failed with HTTP {error.code}")
    except (URLError, TimeoutError, socket.timeout):
        return _error("tavily_unavailable", "Tavily could not be reached")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error("tavily_invalid_response", "Tavily returned an invalid response")

    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        return _error("tavily_invalid_response", "Tavily returned an invalid response")
    results: list[dict[str, Any]] = []
    for item in value["results"][:max_results]:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            continue
        content = str(item.get("content") or "").strip()
        results.append(
            {
                "title": str(item.get("title") or url)[:300],
                "url": url[:2000],
                "content": content[:2000],
                "score": item.get("score")
                if isinstance(item.get("score"), (int, float))
                else None,
                "published_date": str(item.get("published_date"))[:100]
                if item.get("published_date")
                else None,
            }
        )
    return json.dumps(
        {
            "ok": True,
            "query": str(value.get("query") or clean_query),
            "results": results,
            "response_time": value.get("response_time"),
            "request_id": value.get("request_id"),
        },
        ensure_ascii=False,
    )
