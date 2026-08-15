"""Write a CMM bundle into CockroachDB.

The schema (`sql/schema.sql`) already refuses to store anything CMM would have
rejected, so this module's job is narrower than it looks: get the values across
faithfully and let the database enforce the invariants. Where it does more than
that, it is for one of three reasons, each marked below —

  * to fail *before* writing rather than halfway through,
  * to populate `content_fingerprint`, which nothing else can, or
  * to record something the target needs and the source did not measure.

A bundle is written in a single transaction. Migration is the unit of work here,
not the record: a half-written bundle would be scored against its source and
reported as partial loss, which is a lie about the migration rather than a
finding about the data.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from membridge.adapters.cockroach.config import SCHEMA_EMBEDDING_DIM
from membridge.adapters.cockroach.types import encode_vector, is_unit_length
from membridge.cmm import MemoryBundle, MemoryRecord

ADAPTER_NAME = "membridge.adapters.cockroach.writer"

_BUNDLE_INSERT = """
INSERT INTO memory_bundle (
    id, cmm_version, source_system, exported_at, embedding_model, embedding_dim
) VALUES (%s, %s, %s, %s, %s, %s)
"""

_RECORD_INSERT = """
INSERT INTO memory_record (
    id, bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, scope_agent_id, scope_session_id, scope_app_id,
    attribution, created_at, updated_at, expires_at,
    metadata, extensions,
    embedding, embedding_model, embedding_dim, embedding_normalized,
    prov_source_system, prov_source_id, prov_source_version, prov_source_hash,
    prov_exported_at, prov_adapter
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s::cmm_actor, %s, %s, %s,
    %s, %s,
    %s::vector, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s
)
"""


class CockroachWriteError(RuntimeError):
    """Raised when a bundle cannot be written faithfully."""


class CockroachWriter:
    """Writes CMM bundles into a CockroachDB instance holding `sql/schema.sql`."""

    def __init__(self, conn: Any, *, measure_normalization: bool = True) -> None:
        self.conn = conn
        #: The source may not have checked whether its vectors are unit-length —
        #: the Mem0 reader does now, but an older bundle or another adapter may
        #: leave `Embedding.normalized` as None. Measuring here is not derived
        #: data smuggled into storage: it is the one fact this specific target
        #: needs (its only accelerated distance is L2) that CMM permits to be
        #: unknown. Set False to store the source's own answer verbatim, None
        #: included.
        self.measure_normalization = measure_normalization

    # -- pre-flight --------------------------------------------------------

    def _check_writable(self, bundle: MemoryBundle) -> tuple[str | None, int | None]:
        """Everything that should stop a write before it starts.

        `bundle.embedding_space` is the important one and it is called for its
        exception as much as its value: it raises when records disagree about
        their embedder, and finding that out after inserting half the bundle
        would leave a partial migration behind a plausible-looking error.
        """
        space = bundle.embedding_space  # raises on mixed embedders

        if space is not None:
            model, dim = space
            if dim != SCHEMA_EMBEDDING_DIM:
                raise CockroachWriteError(
                    f"bundle embeds at {dim} dimensions with {model!r}, but the schema's "
                    f"VECTOR column is declared VECTOR({SCHEMA_EMBEDDING_DIM}). Migrating "
                    "this bundle needs a schema built for its embedding space; storing it "
                    "in this one would silently truncate or fail per-record."
                )
            return model, dim

        return None, None

    # -- writing -----------------------------------------------------------

    def write_bundle(self, bundle: MemoryBundle, *, bundle_id: UUID | None = None) -> UUID:
        """Write one bundle, atomically. Returns the id of the bundle row.

        Note the fingerprint call: `MemoryRecord.fingerprint()` is computed, never
        stored, in CMM — but the target needs it as an indexed column, and SQL
        cannot compute it (it is defined over NFC-normalized text and SQL has no
        NFC). So this is the only place it can come from, and the reader
        recomputes rather than trusting what it finds. See sql/schema.sql.
        """
        model, dim = self._check_writable(bundle)
        new_id = bundle_id or uuid4()

        # `bundle.fingerprints()` raises on duplicate content within the bundle.
        # The schema's UNIQUE constraint would catch it too, but this way the
        # error names both offending records instead of one index.
        bundle.fingerprints()

        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute(
                    _BUNDLE_INSERT,
                    (
                        str(new_id),
                        bundle.cmm_version,
                        bundle.source_system,
                        bundle.exported_at,
                        model,
                        dim,
                    ),
                )

                # One statement per record rather than executemany. Cockroach's
                # own guidance is that large batch inserts of VECTOR values
                # degrade badly, and a bundle is by definition mostly vectors.
                for record in bundle.records:
                    cur.execute(_RECORD_INSERT, self._record_params(record, new_id))

        return new_id

    def _record_params(self, record: MemoryRecord, bundle_id: UUID) -> tuple[Any, ...]:
        from psycopg.types.json import Jsonb

        embedding = record.embedding
        normalized: bool | None = None
        if embedding is not None:
            normalized = embedding.normalized
            if normalized is None and self.measure_normalization:
                normalized = is_unit_length(embedding.vector)

        return (
            str(record.id),
            str(bundle_id),
            record.cmm_version,
            record.content,
            record.fingerprint(),
            record.scope.user_id,
            record.scope.agent_id,
            record.scope.session_id,
            record.scope.app_id,
            record.attribution.value,
            record.created_at,
            record.updated_at,
            record.expires_at,
            Jsonb(record.metadata),
            Jsonb(record.extensions),
            encode_vector(embedding.vector) if embedding is not None else None,
            embedding.model if embedding is not None else None,
            embedding.dim if embedding is not None else None,
            normalized,
            record.provenance.source_system,
            record.provenance.source_id,
            record.provenance.source_version,
            record.provenance.source_hash,
            record.provenance.exported_at,
            record.provenance.adapter,
        )
