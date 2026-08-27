from __future__ import annotations

import json

from gugugaga import web_search


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_tavily_web_search_sends_bounded_request_and_returns_sources(monkeypatch):
    captured = {}
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret-value")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "query": "latest gugugaga",
                "results": [
                    {
                        "title": "Official result",
                        "url": "https://example.com/source",
                        "content": "A current source snippet.",
                        "score": 0.91,
                    }
                ],
                "response_time": 0.4,
                "request_id": "request-1",
            }
        )

    monkeypatch.setattr(web_search, "urlopen", fake_urlopen)
    result = json.loads(
        web_search.run_web_search(
            "latest gugugaga", max_results=3, topic="news", search_depth="basic"
        )
    )

    assert captured == {
        "url": "https://api.tavily.com/search",
        "authorization": "Bearer tvly-secret-value",
        "payload": {
            "query": "latest gugugaga",
            "max_results": 3,
            "topic": "news",
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        },
        "timeout": 30,
    }
    assert result["ok"] is True
    assert result["results"][0]["url"] == "https://example.com/source"
    assert "tvly-secret-value" not in json.dumps(result)


def test_tavily_web_search_fails_closed_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = json.loads(web_search.run_web_search("current news"))

    assert result == {
        "ok": False,
        "error_code": "tavily_not_configured",
        "message": "Tavily API Key is not configured. Open Settings in the Web Console.",
    }


def test_tavily_web_search_rejects_unbounded_inputs(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    assert json.loads(web_search.run_web_search("x" * 401))["error_code"] == "invalid_query"
    assert json.loads(web_search.run_web_search("query", max_results=11))[
        "error_code"
    ] == "invalid_max_results"
