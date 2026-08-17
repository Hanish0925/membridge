"""What happens when the model fails and the memory layer does not.

No network and no database. Every failure here is arranged deliberately, for
the same reason `test_fidelity.py` damages bundles by hand: the behaviour worth
testing is the one that only shows up when a vendor is having a bad afternoon,
and waiting for that to happen naturally is not a test strategy.

The distinction these tests exist to protect is the project's whole claim in
miniature. A memory layer that retrieved the right records while the model was
unreachable has not failed at what MemBridge demonstrates -- so `/ask` must say
which half broke rather than returning one undifferentiated error. Before this,
a Gemini 503 produced a page that read exactly like a broken database.
"""

from __future__ import annotations

import email.message
import io
import time
import urllib.error
from datetime import datetime, timezone
from typing import Any

import pytest

from membridge.adapters.cockroach import MemoryHit
from membridge.agent import llm as llm_module
from membridge.agent.llm import ChatClient, LLMUnavailable
from membridge.agent.memory_agent import MemoryAgent
from membridge.cmm import Actor, MemoryRecord, Provenance, Scope

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)

ANSWERED = {"choices": [{"message": {"content": "an answer", "tool_calls": []}}]}


# --- doubles ---------------------------------------------------------------


class _Body(io.BytesIO):
    """A urlopen result: a context manager that can be read once."""

    def __enter__(self) -> "_Body":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _http_error(code: int, detail: str = "{}", retry_after: str | None = None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://example.invalid", code, "err", headers, io.BytesIO(detail.encode())
    )


def _client(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any], **kw: Any):
    """A ChatClient whose transport returns `outcomes` in order.

    An outcome is either an exception to raise or a dict to return, so a test
    reads as the sequence the provider produced.
    """
    calls: list[int] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
        calls.append(1)
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        import json

        return _Body(json.dumps(outcome).encode())

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    return ChatClient(provider="gemini", api_key="test-key", **kw), calls


def _record(content: str) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        scope=Scope(user_id="john_001"),
        attribution=Actor.USER,
        created_at=NOW,
        updated_at=NOW,
        provenance=Provenance(
            source_system="mem0", source_id="src-1", exported_at=NOW, adapter="tests"
        ),
    )


class _Reader:
    def __init__(self, contents: list[str] | None = None, fails: bool = False) -> None:
        self.contents = contents if contents is not None else ["a remembered fact"]
        self.fails = fails
        self.queries: list[str] = []

    def search(self, query: Any, **kw: Any) -> list[MemoryHit]:
        if self.fails:
            raise RuntimeError("connection to CockroachDB is gone")
        return [
            MemoryHit(record=_record(text), distance=0.9, similarity=0.55)
            for text in self.contents
        ]


class _Encoder:
    def embed(self, text: str) -> str:
        return f"vector-for:{text}"


class _DeadLLM:
    """Unreachable from the first call."""

    def chat(self, *a: Any, **kw: Any) -> Any:
        raise LLMUnavailable("gemini returned 503: high demand")


# --- the retry layer -------------------------------------------------------


def test_a_transient_503_is_retried_and_the_next_attempt_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _client(
        monkeypatch, [_http_error(503, "overloaded"), ANSWERED]
    )
    reply = client.chat([{"role": "user", "content": "hi"}])

    assert reply.content == "an answer"
    assert len(calls) == 2, "the 503 should not have been the final word"


def test_a_401_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Groq-from-Lambda finding, pinned.

    Groq answers `GET /models` and refuses `POST /chat/completions` with
    `401 invalid_api_key` from an AWS IP using the same key. That is a
    permanent refusal that looks like a credentials problem, and retrying it
    would spend the entire budget arriving at the same answer three times --
    with the retries hiding, rather than revealing, that the cause is fixed.
    """
    client, calls = _client(monkeypatch, [_http_error(401, "invalid_api_key")])

    with pytest.raises(LLMUnavailable, match="401"):
        client.chat([{"role": "user", "content": "hi"}])

    assert len(calls) == 1


def test_retries_are_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls = _client(monkeypatch, [_http_error(503)], retries=2)

    with pytest.raises(LLMUnavailable, match="gave up after 3"):
        client.chat([{"role": "user", "content": "hi"}])

    assert len(calls) == 3, "retries=2 means the first attempt plus two more"


def test_a_retry_is_not_attempted_past_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sleeping into the caller's timeout is worse than failing early.

    The budget exists so a degraded answer can still be assembled and sent.
    Burning what is left of it on a backoff that will be cut short anyway
    trades a useful answer for no answer at all.
    """
    slept: list[float] = []
    client, calls = _client(monkeypatch, [_http_error(503)])
    # After `_client`, which installs its own no-op sleep.
    monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))

    with pytest.raises(LLMUnavailable, match="no time left"):
        client.chat(
            [{"role": "user", "content": "hi"}],
            deadline=time.monotonic() + 0.05,
        )

    assert slept == [], "it should have failed rather than slept past the deadline"
    assert len(calls) == 1


def test_a_server_asking_for_an_absurd_wait_is_refusing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Retry-After: 3600` is a refusal, not a schedule, so it is not honoured."""
    slept: list[float] = []
    client, _ = _client(monkeypatch, [_http_error(429, "{}", "3600"), ANSWERED])
    monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))

    client.chat([{"role": "user", "content": "hi"}])

    assert slept and slept[0] < 30


# --- degrading -------------------------------------------------------------


def test_the_agent_answers_from_memory_when_the_model_is_unreachable() -> None:
    agent = MemoryAgent(_Reader(["John is allergic to shellfish."]), _Encoder(), _DeadLLM())
    answer = agent.ask("what can I not eat?", user_id="john_001")

    assert answer.degraded is True
    assert "503" in (answer.failure or "")
    # The point of the whole exercise: the memory is in the answer.
    assert "shellfish" in answer.text
    assert [h.record.content for h in answer.memories_used] == [
        "John is allergic to shellfish."
    ]


def test_a_fallback_search_is_marked_as_not_the_models_query() -> None:
    """Recall is a tool, so a model that dies first never chose a query.

    MemBridge searching on the raw question is a weaker search than the one the
    model was asked to write -- `recall_memory` tells it to phrase the query as
    the remembered fact might be written. Presenting the two identically would
    claim the model participated when it never ran.
    """
    agent = MemoryAgent(_Reader(), _Encoder(), _DeadLLM())
    answer = agent.ask("what can I not eat?", user_id="john_001")

    assert len(answer.recalled) == 1
    assert answer.recalled[0].fallback is True
    assert answer.recalled[0].query == "what can I not eat?"


def test_recalls_the_model_did_make_are_kept_when_it_dies_mid_loop() -> None:
    """A model that searched and then died leaves real evidence behind."""

    class _DiesAfterSearching:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, *a: Any, **kw: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                from membridge.agent.llm import Reply, ToolCall

                return Reply(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="recall_memory",
                            arguments={"query": "foods to avoid"},
                        )
                    ],
                    raw={"role": "assistant", "tool_calls": []},
                )
            raise LLMUnavailable("gemini returned 503: high demand")

    agent = MemoryAgent(_Reader(), _Encoder(), _DiesAfterSearching())
    answer = agent.ask("what can I not eat?", user_id="john_001")

    assert answer.degraded is True
    assert len(answer.recalled) == 1
    assert answer.recalled[0].query == "foods to avoid"
    assert answer.recalled[0].fallback is False, "the model chose this query"


def test_a_store_failure_is_not_degraded_away() -> None:
    """Degrading is only honest while the memory layer is the part that worked.

    If the store is unreachable too, MemBridge has nothing to say and must say
    so loudly. Returning a calm "the model is unavailable" here would report a
    total failure as a partial one.
    """
    agent = MemoryAgent(_Reader(fails=True), _Encoder(), _DeadLLM())

    with pytest.raises(RuntimeError, match="CockroachDB"):
        agent.ask("what can I not eat?", user_id="john_001")


def test_a_degraded_answer_never_claims_to_be_a_model_answer() -> None:
    agent = MemoryAgent(_Reader(), _Encoder(), _DeadLLM())
    text = agent.ask("anything?", user_id="john_001").text

    assert text.startswith("The language model is unreachable")


def test_memory_returning_nothing_is_reported_as_that() -> None:
    """An empty store and an unreachable model are different failures."""
    agent = MemoryAgent(_Reader(contents=[]), _Encoder(), _DeadLLM())
    answer = agent.ask("anything?", user_id="john_001")

    assert answer.degraded is True
    assert "returned nothing" in answer.text
