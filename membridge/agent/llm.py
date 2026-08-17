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
import random
import time
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
        # An alias rather than a pinned version, which is the opposite of what
        # this project does everywhere else -- and for a reason it learned the
        # hard way. `gemini-2.5-flash` was the pinned default here until it
        # started answering 404 "no longer available to new users": a pin is
        # only reproducible while the vendor still serves it, and a demo that
        # 404s for whoever clones this next is worse than one whose model
        # drifts. The claim being demonstrated is about the memory layer, which
        # does not depend on which model reads it.
        "gemini-flash-latest",
        "GEMINI_API_KEY",
    ),
}

DEFAULT_PROVIDER = "gemini"
DEFAULT_TIMEOUT = 30.0

#: Statuses worth sending the same request again. All of them mean "not now"
#: rather than "not ever": 429 is a rate limit, and 5xx here is the vendor
#: shedding load -- Gemini's free tier answers `503 UNAVAILABLE, "Spikes in
#: demand are usually temporary"`, which is a retry instruction written in
#: prose.
#:
#: What is deliberately absent matters more than what is present. **401 and 403
#: are never retried**, and this project has a specific reason to be sure of
#: that: Groq returns `401 invalid_api_key` for `POST /chat/completions` from an
#: AWS IP while serving `GET /models` with the same key perfectly well. That is
#: a permanent refusal wearing a credentials error's clothes, and retrying it
#: would spend the whole budget arriving at the same answer three times.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

#: Attempts *after* the first, so 2 means at most three requests.
DEFAULT_RETRIES = 2

#: First backoff in seconds; doubled per attempt, with jitter. The jitter is not
#: decoration -- a demo where several people click at once retries in lockstep
#: without it, which is how a transient 503 becomes a sustained one.
DEFAULT_BACKOFF = 0.75

#: Sent on every request, and not optional. Groq sits behind Cloudflare, which
#: rejects urllib's default `Python-urllib/3.11` with **403 error code 1010** --
#: a bot-protection response that looks exactly like a rejected API key and is
#: not one. Any identifiable UA gets through. Worth knowing before debugging a
#: perfectly valid key for an hour.
USER_AGENT = "membridge/0.1 (+https://github.com/membridge)"


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
        retries: int = DEFAULT_RETRIES,
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
        self.retries = max(0, retries)
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
        deadline: float | None = None,
    ) -> Reply:
        """One completion, retried while the failure looks temporary.

        `deadline` is an absolute `time.monotonic()` value and is a wall clock
        over the whole call, retries included -- not a per-request timeout. The
        distinction is what makes this safe to call in a loop: the caller runs
        inside a Lambda with a hard 60s limit, and a function killed by that
        limit returns no body at all, so the honest "the model was unreachable"
        response would never reach whoever is watching. Being late is a worse
        failure than being unavailable, because only one of them can explain
        itself.
        """
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

        data = json.dumps(payload).encode()
        body = self._send_with_retries(data, deadline)

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

    # --- sending ----------------------------------------------------------

    def _send_with_retries(
        self, data: bytes, deadline: float | None
    ) -> dict[str, Any]:
        attempt = 0
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise LLMUnavailable(
                    f"{self.provider} ran out of time after {attempt} attempt(s)"
                )

            # The socket timeout is clamped to what is left, so a single slow
            # request cannot overrun the budget the caller set.
            timeout = self.timeout if remaining is None else min(self.timeout, remaining)
            try:
                return self._send_once(data, timeout)
            except _Transient as exc:
                attempt += 1
                if attempt > self.retries:
                    raise LLMUnavailable(
                        f"{exc.detail} (gave up after {attempt} attempt(s))"
                    ) from exc.cause

                delay = exc.retry_after
                if delay is None:
                    delay = DEFAULT_BACKOFF * (2 ** (attempt - 1))
                    delay += random.uniform(0, delay / 2)

                if deadline is not None:
                    # Sleeping past the deadline only to fail on the next
                    # iteration wastes the time that was left for the fallback
                    # search. Fail now, while the answer can still be useful.
                    left = deadline - time.monotonic()
                    if delay >= left:
                        raise LLMUnavailable(
                            f"{exc.detail} (no time left to retry)"
                        ) from exc.cause

                time.sleep(delay)

    def _send_once(self, data: bytes, timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            message = f"{self.provider} returned {exc.code}: {detail}"
            if exc.code in RETRY_STATUSES:
                raise _Transient(message, _retry_after(exc), exc) from exc
            raise LLMUnavailable(message) from exc
        except urllib.error.URLError as exc:
            # Includes socket timeouts. Treated as temporary because the far
            # side never answered, so nothing is known about whether it would.
            raise _Transient(
                f"cannot reach {self.provider}: {exc.reason}", None, exc
            ) from exc


class _Transient(Exception):
    """Internal: a failure worth attempting again. Never escapes `chat`."""

    def __init__(
        self, detail: str, retry_after: float | None, cause: Exception
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after
        self.cause = cause


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Seconds the server asked us to wait, when it said so and meant seconds.

    `Retry-After` may also be an HTTP date. That form is not parsed here: it is
    rare from these providers, and a misparsed date would produce a nonsense
    delay rather than an obviously absent one. Falling through to the computed
    backoff is the safer wrong answer.
    """
    raw = (exc.headers.get("Retry-After") or "").strip() if exc.headers else ""
    try:
        seconds = float(raw)
    except ValueError:
        return None
    # A server asking for longer than any demo would wait is refusing, not
    # scheduling; let the caller's deadline logic end it instead.
    return seconds if 0 <= seconds <= 30 else None


def _loads_or_empty(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
