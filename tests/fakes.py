from simple_cc.models import ModelResponse


class ScriptedProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.requests = []

    def queue(self, response: ModelResponse):
        self.responses.append(response)

    def complete(self, system, messages, tools, max_tokens=8192):
        self.requests.append({
            "system": system,
            "messages": [dict(message) for message in messages],
            "tools": list(tools),
            "max_tokens": max_tokens,
        })
        if not self.responses:
            raise AssertionError("ScriptedProvider has no queued response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

