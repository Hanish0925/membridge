"""An agent whose entire memory is a CockroachDB table.

The point of this module is narrow and worth stating plainly, because it is easy
to build something that looks like it and is not: the agent has **no
conversational context that outlives a request**. It holds no history, no
summary buffer, no vector cache. Every fact it uses about the user was retrieved
from CockroachDB during the turn it was used in, by a scoped L2 query against
`memory_record_embedding_idx`.

That constraint is what makes the demo evidence rather than decoration. If
memory were partly in a Python object, "CockroachDB is the memory layer" would
be unfalsifiable in exactly the way "we migrated your memories" is unfalsifiable
without a fidelity score. Restart the process and the agent knows precisely as
much as it did before, because the store is where knowing happens.

The recall tool is exposed to the model rather than run unconditionally, so the
agent decides *when* to remember and *what to look for* -- and the transcript
records those decisions, which is what `Answer.recalled` is for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from membridge.adapters.cockroach import CockroachReader, MemoryHit
from membridge.agent.llm import ChatClient, LLMUnavailable, Reply
from membridge.embed import MiniLMEncoder

#: Kept deliberately short. A long persona would let the model answer from its
#: own priors and still sound grounded, which is the one failure mode that would
#: make this demo dishonest.
SYSTEM_PROMPT = """You are an assistant with a persistent memory of this user.

You do not remember anything between conversations on your own. Everything you
know about this user lives in an external memory store, and `recall_memory` is
the only way to reach it. Call it before answering anything that depends on who
the user is, what they like, or what happened previously — including when you
think you already know.

Ground every claim about the user in a memory you retrieved this turn. If recall
returns nothing relevant, say you have no memory of it. Do not guess, and do not
fill gaps with what a typical user might want. Saying "I don't have that in
memory" is a correct answer and a useful one.

Answer in two or three sentences unless asked for more."""

#: Opens a degraded answer. Deliberately says which half failed, because the
#: whole claim of this project is that they are separable: a memory layer that
#: retrieved the right records while the model was unreachable has not failed at
#: the thing being demonstrated, and an error page would report that it had.
DEGRADED_PREFIX = (
    "The language model is unreachable, so this is the memory layer's own "
    "output rather than a phrased answer."
)

RECALL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": (
            "Search this user's long-term memory for facts relevant to a query. "
            "Returns memories ranked by semantic similarity. Call it more than "
            "once with different phrasings if the first search misses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to look for, phrased as the remembered fact might "
                        "be written rather than as the user's question."
                    ),
                },
                "limit": {
                    # Both types, deliberately. Groq validates tool arguments
                    # against this schema *server-side* and returns a hard 400
                    # when they disagree -- and llama routinely emits numeric
                    # arguments as strings (`"limit": "5"`). Declaring only
                    # `integer` turns a well-known model quirk into a failed
                    # request that never reaches `_run_tool`, which coerces it
                    # perfectly well. Better to accept what the model actually
                    # sends than to be right about JSON types and get a 400.
                    "type": ["integer", "string"],
                    "description": "How many memories to return (default 5, max 20).",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass
class Recall:
    """One recall_memory call and what the store returned.

    Kept even when it returns nothing: a search that found no memory is the
    evidence for an "I don't have that" answer, and dropping it would leave the
    transcript looking like the model simply declined to check.
    """

    query: str
    hits: list[MemoryHit]
    #: True when MemBridge ran this search itself because the model never got
    #: the chance to. Recorded rather than blended in, because it is a weaker
    #: search and saying so is the difference between degrading and pretending:
    #: `recall_memory` asks the model to phrase the query *as the remembered
    #: fact might be written*, and a fallback can only use the raw question.
    fallback: bool = False


@dataclass
class Answer:
    text: str
    recalled: list[Recall] = field(default_factory=list)
    turns: int = 0
    #: The model failed and memory did not. The answer is still real; what is
    #: missing is the phrasing, not the retrieval.
    degraded: bool = False
    #: Why, in the provider's own words. Kept on the answer so a caller can show
    #: it rather than having to catch an exception to find out.
    failure: str | None = None

    @property
    def memories_used(self) -> list[MemoryHit]:
        """Every distinct memory retrieved this turn, closest first."""
        best: dict[str, MemoryHit] = {}
        for recall in self.recalled:
            for hit in recall.hits:
                key = hit.record.fingerprint()
                if key not in best or hit.distance < best[key].distance:
                    best[key] = hit
        return sorted(best.values(), key=lambda hit: hit.distance)


class MemoryAgent:
    """Question in, grounded answer out, with the memory reads on the record."""

    def __init__(
        self,
        reader: CockroachReader,
        encoder: MiniLMEncoder,
        llm: ChatClient,
        *,
        max_turns: int = 4,
        budget_seconds: float | None = None,
    ) -> None:
        self.reader = reader
        self.encoder = encoder
        self.llm = llm
        #: The model can search more than once, but not forever. Hitting this
        #: means the model kept searching instead of answering, which is worth
        #: surfacing rather than hiding behind a truncated loop.
        self.max_turns = max_turns
        #: Wall clock for the whole of `ask`, or None for no limit. Worth
        #: setting wherever the caller is itself on a clock: `max_turns` alone
        #: bounds the number of model calls, not their duration, so four turns
        #: against a slow provider can outlast a Lambda's 60s and be killed
        #: mid-answer -- which returns nothing at all, degraded or otherwise.
        self.budget_seconds = budget_seconds

    def recall(self, user_id: str, query: str, limit: int = 5) -> Recall:
        """One scoped vector search. The only route to anything user-specific."""
        embedding = self.encoder.embed(query)
        hits = self.reader.search(
            embedding, user_id=user_id, limit=max(1, min(limit, 20))
        )
        return Recall(query=query, hits=hits)

    def ask(self, question: str, *, user_id: str) -> Answer:
        """Ask, and get an answer even when the model cannot be reached.

        `LLMUnavailable` is caught rather than raised: it is not an error about
        this request so much as a fact about the provider, and the memories are
        worth returning without it. A failure of the *store* still propagates,
        because that one is a real failure of what MemBridge claims to do.
        """
        deadline = (
            None if self.budget_seconds is None
            else time.monotonic() + self.budget_seconds
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        recalled: list[Recall] = []

        for turn in range(1, self.max_turns + 1):
            try:
                reply = self.llm.chat(messages, tools=[RECALL_TOOL], deadline=deadline)
            except LLMUnavailable as exc:
                return self._degrade(question, user_id, recalled, str(exc), turn)

            if not reply.tool_calls:
                return Answer(
                    text=reply.content or "",
                    recalled=recalled,
                    turns=turn,
                )

            messages.append(reply.raw)
            for call in reply.tool_calls:
                messages.append(self._run_tool(call, user_id, recalled))

            if deadline is not None and time.monotonic() >= deadline:
                # The recalls already happened, so there is something to show.
                # Another model call would only be killed by the caller's own
                # timeout, taking the retrieved memories down with it.
                return self._degrade(
                    question,
                    user_id,
                    recalled,
                    f"ran out of time after {self.budget_seconds:.0f}s",
                    turn,
                )

        # Out of turns with the model still searching. Answering from what was
        # retrieved would paper over that, so say what happened.
        return Answer(
            text=(
                "I searched memory several times without settling on an answer. "
                f"Retrieved {sum(len(r.hits) for r in recalled)} memories across "
                f"{len(recalled)} searches."
            ),
            recalled=recalled,
            turns=self.max_turns,
        )

    def _degrade(
        self,
        question: str,
        user_id: str,
        recalled: list[Recall],
        failure: str,
        turn: int,
    ) -> Answer:
        """Answer from memory alone, and say that is what happened.

        The interesting case is `recalled` being empty. Recall is a *tool*, so
        the model chooses when to search and what to search for -- if it died on
        the first call it never chose anything, and there is nothing retrieved
        to fall back to. So MemBridge searches on the raw question itself and
        flags it, rather than reporting an empty answer that would read as "your
        memory store had nothing" when the store was never asked.

        A failure of `recall` here is deliberately not caught. If the model and
        the store are both unreachable then MemBridge genuinely has nothing to
        say, and inventing a calm answer for that case is the exact dishonesty
        this module exists to avoid.
        """
        if not recalled:
            recalled = [
                Recall(
                    query=question,
                    hits=self.recall(user_id, question).hits,
                    fallback=True,
                )
            ]

        answer = Answer(
            text="",
            recalled=recalled,
            turns=turn,
            degraded=True,
            failure=failure,
        )
        answer.text = _degraded_text(answer)
        return answer

    def _run_tool(
        self, call: Any, user_id: str, recalled: list[Recall]
    ) -> dict[str, Any]:
        if call.name != "recall_memory":
            return _tool_result(call.id, f"no tool named {call.name!r}")

        query = call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            # Reported back to the model rather than raised: it usually fixes
            # its own malformed call on the next turn, and a crashed request is
            # a worse demo than a recovered one.
            return _tool_result(call.id, "recall_memory needs a non-empty 'query'")

        recall = self.recall(user_id, query, int(call.arguments.get("limit") or 5))
        recalled.append(recall)

        if not recall.hits:
            return _tool_result(call.id, "No memories found for that query.")

        lines = []
        for index, hit in enumerate(recall.hits, start=1):
            # Similarity is None when the store's vectors are not known
            # unit-length; the model is shown L2 distance in that case rather
            # than a cosine converted on a false premise.
            score = (
                f"similarity {hit.similarity:.3f}"
                if hit.similarity is not None
                else f"L2 distance {hit.distance:.3f}"
            )
            when = hit.record.created_at.date().isoformat()
            lines.append(f"{index}. [{score}, remembered {when}] {hit.record.content}")
        return _tool_result(call.id, "\n".join(lines))


def _tool_result(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _degraded_text(answer: Answer) -> str:
    """The memories as prose, with no claim to be more than a list of them."""
    memories = answer.memories_used
    if not memories:
        return (
            f"{DEGRADED_PREFIX} Memory was searched and returned nothing for "
            "this question."
        )

    searched = (
        "Searched on your question directly, since the model never chose a query."
        if all(recall.fallback for recall in answer.recalled)
        else f"The model ran {len(answer.recalled)} search(es) before it became "
        "unreachable."
    )
    lines = [f"{DEGRADED_PREFIX} {searched} What memory holds:", ""]
    lines += [f"- {hit.record.content}" for hit in memories]
    return "\n".join(lines)


__all__ = [
    "Answer",
    "DEGRADED_PREFIX",
    "LLMUnavailable",
    "MemoryAgent",
    "Recall",
    "RECALL_TOOL",
    "SYSTEM_PROMPT",
]
