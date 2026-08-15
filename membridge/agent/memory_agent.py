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
                    "type": "integer",
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


@dataclass
class Answer:
    text: str
    recalled: list[Recall] = field(default_factory=list)
    turns: int = 0

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
    ) -> None:
        self.reader = reader
        self.encoder = encoder
        self.llm = llm
        #: The model can search more than once, but not forever. Hitting this
        #: means the model kept searching instead of answering, which is worth
        #: surfacing rather than hiding behind a truncated loop.
        self.max_turns = max_turns

    def recall(self, user_id: str, query: str, limit: int = 5) -> Recall:
        """One scoped vector search. The only route to anything user-specific."""
        embedding = self.encoder.embed(query)
        hits = self.reader.search(
            embedding, user_id=user_id, limit=max(1, min(limit, 20))
        )
        return Recall(query=query, hits=hits)

    def ask(self, question: str, *, user_id: str) -> Answer:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        recalled: list[Recall] = []

        for turn in range(1, self.max_turns + 1):
            reply = self.llm.chat(messages, tools=[RECALL_TOOL])

            if not reply.tool_calls:
                return Answer(
                    text=reply.content or "",
                    recalled=recalled,
                    turns=turn,
                )

            messages.append(reply.raw)
            for call in reply.tool_calls:
                messages.append(self._run_tool(call, user_id, recalled))

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


__all__ = [
    "Answer",
    "LLMUnavailable",
    "MemoryAgent",
    "Recall",
    "RECALL_TOOL",
    "SYSTEM_PROMPT",
]
