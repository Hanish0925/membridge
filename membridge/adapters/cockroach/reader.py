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

from typing import Any, Iterator
from uuid import UUID

from membridge.adapters.cockroach.types import decode_vector
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

_RECORD_SELECT = """
SELECT
    id, cmm_version, content, content_fingerprint,
    scope_user_id, scope_agent_id, scope_session_id, scope_app_id,
    attribution, created_at, updated_at, expires_at,
    metadata, extensions,
    embedding, embedding_model, embedding_dim, embedding_normalized,
    prov_source_system, prov_source_id, prov_source_version, prov_source_hash,
    prov_exported_at, prov_adapter
FROM memory_record
WHERE bundle_id = %s
ORDER BY created_at, id
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
