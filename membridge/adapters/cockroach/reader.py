"""Read a CMM bundle back out of CockroachDB — the return trip.

This is the half that makes fidelity scoring possible. A writer alone can only
report that it wrote without raising; whether what arrived is what left is a
question only a reader can answer, and only if it refuses to paper over
differences on the way back.

So this module is deliberately suspicious of its own storage:

  * `content_fingerprint` is recomputed from the content and compared against the
    stored column. The schema documents that column as a denormalized index key
    the reader must verify rather than trust — this is where that happens.
  * Provenance is returned as stored. A record read out of Cockroach still says
    it came from Mem0, because provenance describes a record's *origin*, not the
    last system to hold it. Overwriting it would erase the lineage after exactly
    one hop, which is the failure this project is named after.
  * Nothing is defaulted. A row that cannot produce a valid `MemoryRecord` raises
    instead of yielding a partial one, because a partial record scores as a
    successful migration of less data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator
from uuid import UUID

from membridge.adapters.cockroach.types import (
    decode_vector,
    encode_vector,
    is_unit_length,
)
from membridge.cmm import (
    Actor,
    Embedding,
    MemoryBundle,
    MemoryRecord,
    Provenance,
    Scope,
)

ADAPTER_NAME = "membridge.adapters.cockroach.reader"

_BUNDLE_SELECT = """
SELECT id, cmm_version, source_system, exported_at, embedding_model, embedding_dim
FROM memory_bundle
WHERE id = %s
"""

# Spelled once and reused by every read path, because `_row_to_record` unpacks
# positionally: a SELECT that drifts from this order does not fail, it silently
# maps content into the fingerprint slot.
_RECORD_COLUMNS = """
    id, cmm_version, content, content_fingerprint,
    scope_user_id, scope_agent_id, scope_session_id, scope_app_id,
    attribution, created_at, updated_at, expires_at,
    metadata, extensions,
    embedding, embedding_model, embedding_dim, embedding_normalized,
    prov_source_system, prov_source_id, prov_source_version, prov_source_hash,
    prov_exported_at, prov_adapter
"""

_RECORD_SELECT = f"""
SELECT {_RECORD_COLUMNS}
FROM memory_record
WHERE bundle_id = %s
ORDER BY created_at, id
"""

# The scoped-L2 search the vector index exists to serve.
#
# Two details are load-bearing and neither is obvious from the query text:
#
#   * `scope_user_id = %s` is not merely a filter, it is what makes the index
#     usable at all. `memory_record_embedding_idx` is prefix-partitioned on that
#     column, and CockroachDB falls back to a FULL SCAN when a query does not
#     match the prefix. Dropping the scope predicate to "search everything"
#     would still return correct rows, just by reading the whole table.
#   * `<->` (L2), never `<=>` (cosine). Cockroach accelerates only L2. This
#     ranks identically to cosine *provided the vectors are unit length*, which
#     `search` verifies rather than assumes -- see the guard there.
_SIMILARITY_SEARCH = f"""
SELECT {_RECORD_COLUMNS},
    embedding <-> %(query)s::vector AS distance
FROM memory_record
WHERE scope_user_id = %(user_id)s
  AND embedding IS NOT NULL
  {{expiry_predicate}}
ORDER BY embedding <-> %(query)s::vector
LIMIT %(limit)s
"""

# Distinct embedding spaces present under one scope. A query vector compared
# against a space it does not belong to yields a number that looks like a score
# and means nothing -- the same confusion `Embedding` carries a model name to
# prevent.
_SCOPE_SPACES = """
SELECT DISTINCT embedding_model, embedding_dim, embedding_normalized
FROM memory_record
WHERE scope_user_id = %s AND embedding IS NOT NULL
"""

_BUNDLE_LIST = """
SELECT b.id, b.source_system, b.exported_at, b.imported_at, count(r.id) AS records
FROM memory_bundle b
LEFT JOIN memory_record r ON r.bundle_id = b.id
GROUP BY b.id, b.source_system, b.exported_at, b.imported_at
ORDER BY b.imported_at DESC
"""


class CockroachReadError(RuntimeError):
    """Raised when stored rows cannot be returned to CMM faithfully."""


class EmbeddingSpaceMismatch(CockroachReadError):
    """Raised when a query vector does not live in the space it is searched against.

    Its own type because it is the one search failure a caller might reasonably
    want to catch and recover from -- by re-embedding the query with the right
    model -- rather than a sign the store is corrupt.
    """


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved memory and how close it was.

    `distance` is L2, as returned by CockroachDB. `similarity` is the cosine
    equivalent, and is present only when the space is known unit-length: for
    unit vectors ||a-b||^2 = 2 - 2*cos(a,b), so cosine is recoverable exactly
    from L2. When it is None the ranking is still L2 and still correct as a
    ranking -- there is simply no honest cosine number to report, and inventing
    one is how a meaningless score ends up in a report looking authoritative.
    """

    record: MemoryRecord
    distance: float
    similarity: float | None


class CockroachReader:
    """Reads CMM bundles back out of CockroachDB."""

    def __init__(self, conn: Any, *, verify_fingerprints: bool = True) -> None:
        self.conn = conn
        #: Off only for diagnosing a store already known to be inconsistent —
        #: the whole reason the column exists is that it can drift from the
        #: content it describes, and a reader that does not check is a reader
        #: that launders that drift into a clean-looking bundle.
        self.verify_fingerprints = verify_fingerprints

    # -- listing -----------------------------------------------------------

    def list_bundles(self) -> list[dict[str, Any]]:
        """Every bundle in the target, newest import first."""
        with self.conn.cursor() as cur:
            cur.execute(_BUNDLE_LIST)
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "source_system": row[1],
                "exported_at": row[2],
                "imported_at": row[3],
                "records": row[4],
            }
            for row in rows
        ]

    # -- reading -----------------------------------------------------------

    def read_bundle(self, bundle_id: UUID | str) -> MemoryBundle:
        """Read one bundle back into CMM."""
        with self.conn.cursor() as cur:
            cur.execute(_BUNDLE_SELECT, (str(bundle_id),))
            header = cur.fetchone()
            if header is None:
                raise CockroachReadError(f"no bundle {bundle_id} in this target")

            cur.execute(_RECORD_SELECT, (str(bundle_id),))
            rows = cur.fetchall()

        bundle = MemoryBundle(
            cmm_version=header[1],
            source_system=header[2],
            exported_at=header[3],
            records=[self._row_to_record(row) for row in rows],
        )

        # The bundle-level invariant, re-checked on the way out. The schema
        # enforces it structurally via the composite FK, so a failure here means
        # the two disagree — worth knowing loudly.
        declared = (header[4], header[5]) if header[4] is not None else None
        if bundle.embedding_space != declared:
            raise CockroachReadError(
                f"bundle {bundle_id} declares embedding space {declared} but its "
                f"records are in {bundle.embedding_space}"
            )

        return bundle

    def iter_records(self, bundle_id: UUID | str) -> Iterator[MemoryRecord]:
        """Records one at a time, for stores too large to bundle in memory.

        The Mem0 reader's open question about streaming applies on this side too,
        and a relational source is the one place it is actually easy to answer.
        """
        with self.conn.cursor() as cur:
            cur.execute(_RECORD_SELECT, (str(bundle_id),))
            for row in cur:
                yield self._row_to_record(row)

    # -- retrieval ---------------------------------------------------------

    def scope_embedding_spaces(
        self, user_id: str
    ) -> list[tuple[str, int, bool | None]]:
        """Every distinct (model, dim, normalized) held under one scope."""
        with self.conn.cursor() as cur:
            cur.execute(_SCOPE_SPACES, (user_id,))
            return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    def search(
        self,
        query: Embedding,
        *,
        user_id: str,
        limit: int = 5,
        include_expired: bool = False,
    ) -> list[MemoryHit]:
        """Nearest memories to `query` within one user's scope.

        This is the read an agent actually makes -- and the only path that uses
        `memory_record_embedding_idx`, which until now was a declared index no
        code ever asked a question of.

        Three refusals, each protecting a number that would otherwise look fine:

        1. **Mixed embedding spaces raise.** If the scope holds vectors from more
           than one embedder, there is no single space to search and the
           distances would not be mutually comparable. `MemoryBundle` raises on
           this at export time for the same reason; a target that has accepted
           two bundles can reach the state a single bundle cannot.
        2. **A query from the wrong space raises** (`EmbeddingSpaceMismatch`).
           Cosine or L2 between vectors from different embedders is a number
           with no meaning that still sorts, so the failure is silent by
           default: plausible-looking results in a plausible-looking order.
        3. **Unnormalized vectors get no cosine number.** L2 ranks identically
           to cosine only for unit-length vectors. Where that is unproven --
           including where `embedding_normalized` is NULL, which means "no
           adapter checked" and is not the same as false -- the hit carries
           `similarity=None` rather than a converted value that would be wrong.

        `include_expired` defaults to False, which looks like the `show_expired`
        default this project criticises Mem0 for and is the opposite decision.
        Mem0 hides expired records on the *migration* path, where the store's
        own contents are the subject and dropping them loses data silently. This
        is the *retrieval* path, feeding an agent that is about to act: serving a
        memory the user expired is the "confidently wrong" failure that expiry
        exists to prevent. The migration path in `read_bundle` still returns
        every row.
        """
        spaces = self.scope_embedding_spaces(user_id)
        if not spaces:
            return []
        if len(spaces) > 1:
            raise CockroachReadError(
                f"scope user_id={user_id!r} holds {len(spaces)} embedding spaces "
                f"({', '.join(f'{m}/{d}' for m, d, _ in spaces)}). Distances across "
                "spaces are not comparable, so there is no correct result to return."
            )

        model, dim, normalized = spaces[0]
        if (query.model, query.dim) != (model, dim):
            raise EmbeddingSpaceMismatch(
                f"query vector is {query.model}/{query.dim} but this scope stores "
                f"{model}/{dim}. Re-embed the query with {model} — comparing across "
                "embedders produces a score that sorts but means nothing."
            )

        # Both sides must be unit length for the L2->cosine identity to hold. The
        # stored side is the column (NULL = unchecked, not true); the query side
        # is measured here, since a caller can hand over any vector.
        unit_space = normalized is True and is_unit_length(query.vector)

        sql = _SIMILARITY_SEARCH.format(
            expiry_predicate=(
                "" if include_expired else "AND (expires_at IS NULL OR expires_at >= now())"
            )
        )
        with self.conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "query": encode_vector(query.vector),
                    "user_id": user_id,
                    "limit": limit,
                },
            )
            rows = cur.fetchall()

        hits = []
        for row in rows:
            distance = float(row[-1])
            # cos = 1 - d^2/2 for unit vectors. Clamped because float32 storage
            # can put an identical vector a hair past 1.0.
            similarity = max(-1.0, min(1.0, 1.0 - (distance * distance) / 2.0))
            hits.append(
                MemoryHit(
                    record=self._row_to_record(row[:-1]),
                    distance=distance,
                    similarity=similarity if unit_space else None,
                )
            )
        return hits

    # -- mapping -----------------------------------------------------------

    def _row_to_record(self, row: tuple[Any, ...]) -> MemoryRecord:
        (
            record_id, cmm_version, content, stored_fingerprint,
            user_id, agent_id, session_id, app_id,
            attribution, created_at, updated_at, expires_at,
            metadata, extensions,
            embedding, embedding_model, embedding_dim, embedding_normalized,
            prov_system, prov_id, prov_version, prov_hash, prov_exported_at, prov_adapter,
        ) = row

        vector = decode_vector(embedding)
        embedding_obj = None
        if vector is not None:
            embedding_obj = Embedding(
                vector=vector,
                model=embedding_model,
                dim=embedding_dim,
                normalized=embedding_normalized,
            )

        record = MemoryRecord(
            cmm_version=cmm_version,
            id=record_id if isinstance(record_id, UUID) else UUID(str(record_id)),
            content=content,
            scope=Scope(
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                app_id=app_id,
            ),
            attribution=Actor(attribution),
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            metadata=metadata,
            embedding=embedding_obj,
            extensions=extensions or {},
            provenance=Provenance(
                source_system=prov_system,
                source_id=prov_id,
                source_version=prov_version,
                source_hash=prov_hash,
                exported_at=prov_exported_at,
                adapter=prov_adapter,
            ),
        )

        if self.verify_fingerprints:
            recomputed = record.fingerprint()
            if recomputed != stored_fingerprint:
                raise CockroachReadError(
                    f"record {record_id}: stored fingerprint {stored_fingerprint[:12]}… does "
                    f"not match content (recomputed {recomputed[:12]}…). The content or the "
                    "column was modified after the write; alignment against the source would "
                    "be meaningless."
                )

        return record
