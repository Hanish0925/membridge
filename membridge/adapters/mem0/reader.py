"""Read a Mem0 store into a CMM bundle.

Three things about Mem0's read API shape this module, all of them discovered by
reading its source rather than its docs (see CLAUDE.md):

  1. `get_all()` is paginated by `top_k`, which **defaults to 20**. A migration
     that calls it plainly gets its first twenty memories and no indication that
     anything was left behind. `read_bundle` refuses to guess: it escalates
     `top_k` until the result stops filling it, so a short read is provably short.

  2. `get_all(show_expired=False)` is the default and it *silently drops*
     expired memories. For a migration that is precisely backwards — an expired
     memory is still data the source holds, and whether the target should keep
     it is the caller's decision, not Mem0's. We always read with
     `show_expired=True` and carry expiry through CMM's `expires_at`.

  3. `get_all()` never returns vectors, and this is structural, not incidental:
     Mem0's Qdrant backend hardcodes `with_vectors=False` in its `list()`. The
     only way to migrate embeddings is to reach past Mem0 into the vector store
     and fetch them by id — which is also the only way to see `text_lemmatized`,
     a payload field Mem0's own API never surfaces.

The mapping itself is deliberately a plain function (`record_to_cmm`) so it can
be tested against the recorded fixture without a live Mem0, Qdrant or Ollama.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Iterable, Sequence

from membridge.adapters.mem0.config import EMBEDDING_DIM, EMBEDDING_MODEL
from membridge.cmm import (
    Actor,
    Embedding,
    MemoryBundle,
    MemoryRecord,
    Provenance,
    Scope,
    is_unit_length,
)

SOURCE_SYSTEM = "mem0"
ADAPTER_NAME = "membridge.adapters.mem0.reader"

#: Where `read_bundle` starts before escalating. Comfortably above Mem0's own
#: default of 20 so the common case is a single round trip.
INITIAL_TOP_K = 512

#: Refuse to escalate past this. A store this size wants a streaming migration,
#: not a bundle held in memory, and silently trying would just fail later and
#: less clearly.
MAX_TOP_K = 1_048_576

#: Fields Mem0 promotes out of the Qdrant payload onto the record, which CMM
#: maps to typed fields. Anything *not* here and not core is left in `metadata`
#: by Mem0 itself.
_SCOPE_FIELDS = ("user_id", "agent_id", "run_id")

#: Vendor fields preserved verbatim under `extensions["mem0"]`. Present here
#: because CMM has no cross-system meaning for them, not because they are
#: unimportant — `role` and `actor_id` are Mem0's multi-actor attribution, which
#: we have no ground-truth dump for yet.
_EXTENSION_FIELDS = ("role", "actor_id", "text_lemmatized", "expiration_date")


class Mem0ReadError(RuntimeError):
    """Raised when a read cannot be shown to be complete or faithful."""


def _parse_timestamp(value: Any, *, field: str, record_id: str) -> datetime:
    if value is None:
        raise Mem0ReadError(
            f"mem0 record {record_id} has no {field}; CMM requires it, and "
            "inventing one would fabricate ordering the source never asserted"
        )
    if isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        # Mem0 writes UTC ISO strings with an offset. A naive one means the
        # payload was written by something else, and guessing its zone is how
        # timestamps quietly shift by hours across a migration.
        raise Mem0ReadError(
            f"mem0 record {record_id} has a naive {field} ({value!r}); refusing "
            "to assume a timezone"
        )
    return parsed


def _parse_expiration(value: Any) -> datetime | None:
    """Mem0 stores expiry as a bare date; CMM wants an instant.

    Mem0 treats a memory as expired once that date is *past* (`< today`), so the
    equivalent instant is the end of that day. Midnight would expire memories a
    day early. The original string is kept in `extensions` regardless.
    """
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, time.max, tzinfo=timezone.utc)
    try:
        return datetime.combine(date.fromisoformat(str(value)), time.max, tzinfo=timezone.utc)
    except ValueError as exc:
        raise Mem0ReadError(f"unparseable expiration_date {value!r}") from exc


def _attribution(raw: dict[str, Any]) -> Actor:
    """Mem0's `attributed_to`, falling back to `role`.

    Both are absent in stores written before Mem0 2.x's multi-actor support, and
    CMM says so explicitly rather than assuming the user said it.
    """
    value = raw.get("attributed_to") or raw.get("role")
    if not value:
        return Actor.UNKNOWN
    try:
        return Actor(str(value).lower())
    except ValueError:
        # An attribution we don't model is still information; keep it in
        # extensions (see record_to_cmm) rather than mapping it to the wrong one.
        return Actor.UNKNOWN


def record_to_cmm(
    raw: dict[str, Any],
    *,
    exported_at: datetime,
    source_version: str | None = None,
    embedding: Embedding | None = None,
    payload: dict[str, Any] | None = None,
) -> MemoryRecord:
    """Map one Mem0 record onto CMM.

    `payload` is the raw Qdrant payload when available; it carries fields Mem0's
    API withholds (`text_lemmatized`) and is merged in for extension purposes
    only — the typed fields always come from Mem0's own view of the record.
    """
    record_id = str(raw.get("id", "<unknown>"))
    merged: dict[str, Any] = {**(payload or {}), **raw}

    scope = Scope(
        user_id=merged.get("user_id"),
        agent_id=merged.get("agent_id"),
        session_id=merged.get("run_id"),
    )

    extensions: dict[str, Any] = {
        key: merged[key] for key in _EXTENSION_FIELDS if merged.get(key) is not None
    }
    # An attribution string Mem0 has and CMM doesn't model must not vanish.
    raw_attribution = merged.get("attributed_to") or merged.get("role")
    if raw_attribution and _attribution(merged) is Actor.UNKNOWN:
        extensions["attributed_to_raw"] = raw_attribution

    created_at = _parse_timestamp(merged.get("created_at"), field="created_at", record_id=record_id)
    updated_at_raw = merged.get("updated_at")
    updated_at = (
        created_at
        if updated_at_raw is None
        else _parse_timestamp(updated_at_raw, field="updated_at", record_id=record_id)
    )

    return MemoryRecord(
        content=merged.get("memory") or merged.get("data") or "",
        scope=scope,
        attribution=_attribution(merged),
        created_at=created_at,
        updated_at=updated_at,
        expires_at=_parse_expiration(merged.get("expiration_date")),
        metadata=merged.get("metadata"),
        embedding=embedding,
        extensions={SOURCE_SYSTEM: extensions} if extensions else {},
        provenance=Provenance(
            source_system=SOURCE_SYSTEM,
            source_id=record_id,
            source_version=source_version,
            source_hash=merged.get("hash"),
            exported_at=exported_at,
            adapter=ADAPTER_NAME,
        ),
    )


class Mem0Reader:
    """Reads a live Mem0 instance into CMM bundles."""

    def __init__(
        self,
        memory: Any,
        *,
        source_version: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.memory = memory
        self.source_version = source_version or self._detect_version()
        # Qdrant records dimensionality but not *which* embedder produced a
        # vector, so this has to come from config. Getting it wrong silently
        # poisons every downstream similarity number.
        self.embedding_model = embedding_model or self._detect_embedding_model()
        self.embedding_dim = embedding_dim or self._detect_embedding_dim()

    # -- introspection -----------------------------------------------------

    @staticmethod
    def _detect_version() -> str | None:
        try:
            from importlib.metadata import version

            return f"mem0ai {version('mem0ai')}"
        except Exception:  # pragma: no cover - best-effort provenance
            return None

    def _detect_embedding_model(self) -> str:
        config = getattr(getattr(self.memory, "embedding_model", None), "config", None)
        return getattr(config, "model", None) or EMBEDDING_MODEL

    def _detect_embedding_dim(self) -> int:
        config = getattr(getattr(self.memory, "embedding_model", None), "config", None)
        return getattr(config, "embedding_dims", None) or EMBEDDING_DIM

    @property
    def collection_name(self) -> str:
        return getattr(self.memory, "collection_name", None) or getattr(
            self.memory.vector_store, "collection_name"
        )

    # -- reading -----------------------------------------------------------

    def read_records(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Every record matching `filters`, with truncation ruled out.

        Mem0 offers no cursor and no total count, only `top_k`. So completeness
        is established by escalation: ask for `n`, and if exactly `n` come back
        we cannot distinguish "that's all of them" from "there are more", so ask
        for more. This is only sound because we read with `show_expired=True` —
        with server-side filtering in play, a short result would not imply the
        end of the data.
        """
        top_k = INITIAL_TOP_K
        while True:
            raw = self.memory.get_all(filters=filters, top_k=top_k, show_expired=True)
            records = raw["results"] if isinstance(raw, dict) and "results" in raw else raw

            if len(records) < top_k:
                return list(records)

            if top_k >= MAX_TOP_K:
                raise Mem0ReadError(
                    f"mem0 returned {len(records)} records at top_k={top_k}; refusing "
                    "to escalate further. A store this large needs a streaming "
                    "migration rather than a single in-memory bundle."
                )
            top_k *= 2

    def read_vector_store(self, ids: Sequence[str]) -> dict[str, tuple[list[float] | None, dict[str, Any]]]:
        """Fetch vectors and full payloads straight from Qdrant, by id.

        Necessary because Mem0's Qdrant backend hardcodes `with_vectors=False`
        in `list()`, so there is no route to the embeddings through Mem0's API.
        """
        if not ids:
            return {}

        client = self.memory.vector_store.client
        try:
            points = client.retrieve(
                collection_name=self.collection_name,
                ids=list(ids),
                with_vectors=True,
                with_payload=True,
            )
        except Exception as exc:
            raise Mem0ReadError(
                f"could not read vectors from collection {self.collection_name!r}: {exc!r}"
            ) from exc

        out: dict[str, tuple[list[float] | None, dict[str, Any]]] = {}
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):  # named vectors
                vector = next(iter(vector.values()), None)
            out[str(point.id)] = (vector, dict(point.payload or {}))
        return out

    def read_bundle(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_embeddings: bool = True,
    ) -> MemoryBundle:
        """Read one scope out of Mem0 as a CMM bundle."""
        filters = {
            key: value
            for key, value in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id))
            if value is not None
        }
        if not filters:
            raise Mem0ReadError(
                "at least one of user_id, agent_id, run_id is required; mem0 has no "
                "unscoped read and a bundle of everything is not a thing it offers"
            )

        exported_at = datetime.now(timezone.utc)
        raw_records = self.read_records(filters)

        extras: dict[str, tuple[list[float] | None, dict[str, Any]]] = {}
        if include_embeddings:
            extras = self.read_vector_store([str(r["id"]) for r in raw_records if r.get("id")])
            missing = {str(r.get("id")) for r in raw_records} - set(extras)
            if missing:
                raise Mem0ReadError(
                    f"{len(missing)} record(s) present in mem0 but absent from the "
                    f"vector store (e.g. {sorted(missing)[0]}); the store is "
                    "inconsistent and a migration from it would be too"
                )

        records = []
        for raw in raw_records:
            vector, payload = extras.get(str(raw.get("id")), (None, {}))
            embedding = None
            if vector is not None:
                components = list(vector)
                embedding = Embedding(
                    vector=components,
                    model=self.embedding_model,
                    dim=len(components),
                    # Measured, not assumed. A target may accelerate only some
                    # distance metrics — CockroachDB indexes L2 and not cosine —
                    # and L2 ranks the same as cosine only for unit-length
                    # vectors. Leaving this None would hand the target a choice
                    # between a wrong answer and a slow one, with no way to tell
                    # which it was making.
                    normalized=is_unit_length(components),
                )
            records.append(
                record_to_cmm(
                    raw,
                    exported_at=exported_at,
                    source_version=self.source_version,
                    embedding=embedding,
                    payload=payload,
                )
            )

        return MemoryBundle(
            source_system=SOURCE_SYSTEM,
            exported_at=exported_at,
            records=records,
        )


def bundle_from_dump(
    raw: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    source_version: str | None = None,
    exported_at: datetime | None = None,
) -> MemoryBundle:
    """Build a bundle from an already-captured `get_all()` dump.

    The offline path: lets the recorded fixture exercise the same mapping the
    live reader uses, with no Mem0, Qdrant or Ollama in the loop. Carries no
    embeddings, because a `get_all()` dump has none to carry.
    """
    records = raw["results"] if isinstance(raw, dict) and "results" in raw else list(raw)
    stamp = exported_at or datetime.now(timezone.utc)
    return MemoryBundle(
        source_system=SOURCE_SYSTEM,
        exported_at=stamp,
        records=[
            record_to_cmm(r, exported_at=stamp, source_version=source_version) for r in records
        ],
    )
