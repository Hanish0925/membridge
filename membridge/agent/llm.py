"""A minimal chat client for OpenAI-compatible endpoints.

Written against `urllib` rather than a vendor SDK for two reasons, both about
where this runs. A Lambda zip that already carries a 90MB ONNX model has no room
for an SDK and its transitive dependencies; and the only thing MemBridge needs
from a chat API is one POST with tool definitions, which is not worth a
dependency.

Groq and Gemini both expose OpenAI-compatible chat endpoints, so switching
providers is a base URL, a model name and a different key -- see PROVIDERS. The
agent deliberately does not care which is in use: the interesting claim in this
project is about the memory layer, and a memory layer whose behaviour depends on
which model is reading it would be a bad one.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

#: Base URL and a sensible default model per provider. Both speak the same
#: request shape, which is the only reason a single client works for both.
PROVIDERS: dict[str, tuple[str, str, str]] = {
    # name: (base url, default model, env var holding the key)
    "groq": (
        "https://api.groq.com/openai/v1/chat/completions",
        "llama-3.3-70b-versatile",
        "GROQ_API_KEY",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    ),
}

DEFAULT_PROVIDER = "groq"
DEFAULT_TIMEOUT = 30.0


class LLMUnavailable(RuntimeError):
    """Raised when the model cannot be reached or is not configured.

    Its own type so the agent can degrade honestly: a memory layer that
    retrieved the right records and could not reach a model has not failed at
    the thing this project is about, and should say so rather than reporting an
    empty answer.
    """


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    """One assistant turn: prose, tool calls, or both."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: The raw message, kept so it can be appended to the next request verbatim.
    #: Reconstructing it loses provider-specific fields and breaks the loop in
    #: ways that only show up on the second tool call.
    raw: dict[str, Any] = field(default_factory=dict)


class ChatClient:
    """One POST, with tools. Nothing else."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.provider = provider or os.environ.get("MEMBRIDGE_LLM_PROVIDER", DEFAULT_PROVIDER)
        if self.provider not in PROVIDERS:
            raise LLMUnavailable(
                f"unknown provider {self.provider!r}; known: {sorted(PROVIDERS)}"
            )
        url, default_model, key_env = PROVIDERS[self.provider]
        self.url = url
        self.model = model or os.environ.get("MEMBRIDGE_LLM_MODEL") or default_model
        self.api_key = api_key or os.environ.get(key_env)
        self.timeout = timeout
        if not self.api_key:
            raise LLMUnavailable(
                f"no API key for {self.provider}: set {key_env}"
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # Zero by default because the demo is a claim about what was
            # retrieved. A model that phrases the same memories differently on
            # each run makes it harder to see that the memories are the same.
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            raise LLMUnavailable(f"{self.provider} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"cannot reach {self.provider}: {exc.reason}") from exc

        message = body["choices"][0]["message"]
        calls = [
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                # Arguments arrive as a JSON *string*. A model can emit
                # malformed JSON here, and the loop must not die on it -- an
                # empty dict makes the tool report its own bad input back to
                # the model, which usually recovers on the next turn.
                arguments=_loads_or_empty(call["function"].get("arguments", "")),
            )
            for call in message.get("tool_calls") or []
        ]
        return Reply(content=message.get("content"), tool_calls=calls, raw=message)


def _loads_or_empty(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
