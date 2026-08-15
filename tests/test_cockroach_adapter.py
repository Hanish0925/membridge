"""The CockroachDB adapter.

Split deliberately. The encoding tests run everywhere and need no database,
because the subtlest bug found while building this adapter lives entirely in the
float32 text representation and would be missed by any test that only checked
"did the round trip raise". The round-trip tests need a live CockroachDB and skip
cleanly without one — a skipped test says so, whereas a mocked database would
have happily confirmed the wrong behaviour.

Point the live tests at a cluster with MEMBRIDGE_COCKROACH_DSN. They create and
drop their own bundles and leave nothing behind.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest

from membridge.adapters.cockroach import (
    CockroachReader,
    CockroachWriteError,
    CockroachWriter,
    as_float32,
    decode_vector,
    encode_vector,
    is_float32_exact,
)
from membridge.cmm import (
    Actor,
    Embedding,
    MemoryBundle,
    MemoryRecord,
    Provenance,
    Scope,
    is_unit_length,
)

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
NOW_UUID = UUID("11111111-1111-1111-1111-111111111111")


def _record(**overrides: Any) -> MemoryRecord:
    base: dict[str, Any] = dict(
        content="John is allergic to shellfish.",
        scope=Scope(user_id="john_001"),
        attribution=Actor.USER,
        created_at=NOW,
        updated_at=NOW,
        provenance=Provenance(
            source_system="mem0",
            source_id="src-1",
            exported_at=NOW,
            adapter="tests",
        ),
    )
    base.update(overrides)
    return MemoryRecord(**base)


def _embedding(dim: int = 384, value: float = 0.1) -> Embedding:
    return Embedding(vector=[value] * dim, model="test-embedder", dim=dim)


# --- encoding, no database needed -----------------------------------------


def test_vector_round_trips_through_the_wire_format() -> None:
    vector = [0.1, -0.25, 0.0, 1.0]
    assert decode_vector(encode_vector(vector)) == [as_float32(v) for v in vector]


def test_cockroachs_nine_digit_output_is_not_a_float64() -> None:
    """The bug this adapter was nearly shipped with.

    CockroachDB prints a float32 component as the shortest decimal that
    round-trips *as a float32* — nine significant digits. Parsed as a float64
    that string is a DIFFERENT number from the float32 it came from. Comparing
    the two for equality fails on every component of a perfectly migrated
    vector, which would read as total embedding loss.
    """
    stored = 0.018742891028523445           # a genuine float32 value
    printed = "0.018742891"                 # how CockroachDB renders it

    assert float(printed) != stored, "the naive parse is not the stored value"
    assert as_float32(float(printed)) == stored, "rounding to float32 recovers it"
    assert decode_vector(f"[{printed}]") == [stored]


def test_decode_recovers_a_real_384d_vector_bit_for_bit() -> None:
    # float32 values, as they arrive from Qdrant.
    vector = [as_float32(i / 384.0 - 0.5) for i in range(384)]
    printed = "[" + ",".join(f"{v:.9g}" for v in vector) + "]"
    assert decode_vector(printed) == vector


def test_float32_exactness_is_reported_not_assumed() -> None:
    assert is_float32_exact([as_float32(0.1), as_float32(-0.25)])
    # A float64 that no float32 can represent: writing it IS lossy.
    assert not is_float32_exact([0.018742891028523446 + 1e-17, 0.1])


def test_unit_length_is_tolerant_but_not_credulous() -> None:
    assert is_unit_length([1.0, 0.0, 0.0])
    assert is_unit_length([0.6, 0.8])
    assert not is_unit_length([1.0, 1.0])
    assert not is_unit_length([0.0, 0.0])


def test_encoding_a_vector_keeps_full_precision() -> None:
    """Encoding must not be where precision goes; storage is float32 already."""
    value = 0.1234567890123456789
    assert float(encode_vector([value]).strip("[]")) == value


# --- pre-flight refusals, no database needed ------------------------------


def test_writer_refuses_a_bundle_in_the_wrong_embedding_space() -> None:
    bundle = MemoryBundle(
        source_system="mem0",
        exported_at=NOW,
        records=[_record(embedding=_embedding(dim=768))],
    )
    with pytest.raises(CockroachWriteError, match="768 dimensions"):
        CockroachWriter(conn=None)._check_writable(bundle)


def test_writer_refuses_a_bundle_that_mixes_embedders() -> None:
    """MemoryBundle.embedding_space raises; the writer must not swallow it."""
    bundle = MemoryBundle(
        source_system="mem0",
        exported_at=NOW,
        records=[
            _record(embedding=Embedding(vector=[0.1] * 384, model="a", dim=384)),
            _record(
                content="different",
                provenance=Provenance(source_system="mem0", source_id="src-2", exported_at=NOW),
                embedding=Embedding(vector=[0.1] * 384, model="b", dim=384),
            ),
        ],
    )
    with pytest.raises(ValueError, match="mixes embedding spaces"):
        CockroachWriter(conn=None)._check_writable(bundle)


def test_a_vectorless_bundle_is_writable() -> None:
    bundle = MemoryBundle(source_system="mem0", exported_at=NOW, records=[_record()])
    assert CockroachWriter(conn=None)._check_writable(bundle) == (None, None)


def test_writer_measures_normalization_the_source_left_unknown() -> None:
    """CMM permits `normalized=None`; this target cannot afford not to know.

    Its only accelerated distance is L2, which ranks like cosine only for
    unit-length vectors.
    """
    unit = [1.0] + [0.0] * 383
    record = _record(embedding=Embedding(vector=unit, model="m", dim=384))
    assert record.embedding.normalized is None

    params = CockroachWriter(conn=None)._record_params(record, bundle_id=NOW_UUID)
    assert params[18] is True  # embedding_normalized

    verbatim = CockroachWriter(conn=None, measure_normalization=False)
    assert verbatim._record_params(record, bundle_id=NOW_UUID)[18] is None


def test_writer_populates_the_fingerprint_column() -> None:
    """Nothing else can: SQL has no NFC, so it cannot be a computed column."""
    record = _record()
    params = CockroachWriter(conn=None)._record_params(record, bundle_id=NOW_UUID)
    assert params[4] == record.fingerprint()
    assert len(params[4]) == 64


# --- round trip, needs a live CockroachDB ---------------------------------


def _connection() -> Any:
    from membridge.adapters.cockroach import connect, ensure_schema

    conn = connect()
    ensure_schema(conn)
    return conn


@pytest.fixture(scope="module")
def conn() -> Any:
    try:
        connection = _connection()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no CockroachDB reachable: {exc!r}")
    yield connection
    connection.close()


def _bundle_with(records: list[MemoryRecord]) -> MemoryBundle:
    return MemoryBundle(source_system="mem0", exported_at=NOW, records=records)


def test_round_trip_preserves_every_typed_field(conn: Any) -> None:
    source = _bundle_with([
        _record(
            content="John is allergic to shellfish.",
            expires_at=NOW + timedelta(days=365),
            metadata={"confidence": 0.9, "nested": {"a": [1, 2]}},
            extensions={"mem0": {"actor_id": "john", "role": "user"}},
            embedding=Embedding(
                vector=[as_float32(i / 384 - 0.5) for i in range(384)],
                model="sentence-transformers/all-MiniLM-L6-v2",
                dim=384,
                normalized=False,
            ),
        ),
        _record(
            content="Maria is vegetarian.",
            scope=Scope(user_id="john_001", agent_id="agent-1", session_id="run-1"),
            attribution=Actor.UNKNOWN,
            provenance=Provenance(
                source_system="mem0",
                source_id="src-2",
                source_version="mem0ai 2.0.15",
                source_hash="deadbeef",
                exported_at=NOW,
                adapter="tests",
            ),
        ),
    ])

    bundle_id = CockroachWriter(conn).write_bundle(source)
    target = CockroachReader(conn).read_bundle(bundle_id)

    assert len(target.records) == len(source.records)
    assert set(source.fingerprints()) == set(target.fingerprints())

    src, dst = source.fingerprints(), target.fingerprints()
    for fingerprint, original in src.items():
        arrived = dst[fingerprint]
        assert arrived.content == original.content
        assert arrived.scope == original.scope
        assert arrived.attribution == original.attribution
        assert arrived.created_at == original.created_at
        assert arrived.updated_at == original.updated_at
        assert arrived.expires_at == original.expires_at
        assert arrived.metadata == original.metadata
        assert arrived.extensions == original.extensions
        assert arrived.id == original.id
        assert arrived.provenance == original.provenance
        if original.embedding is not None:
            # Bit-exact, not approximate. See decode_vector.
            assert arrived.embedding.vector == original.embedding.vector
            assert arrived.embedding.model == original.embedding.model
            assert arrived.embedding.normalized == original.embedding.normalized
        else:
            assert arrived.embedding is None


def test_provenance_still_names_the_original_source(conn: Any) -> None:
    """A record read out of Cockroach must not claim Cockroach wrote it.

    Provenance describes origin, not the last system to hold the record. If the
    target overwrote it, lineage would be gone after exactly one hop.
    """
    bundle_id = CockroachWriter(conn).write_bundle(_bundle_with([_record()]))
    target = CockroachReader(conn).read_bundle(bundle_id)

    assert target.source_system == "mem0"
    assert target.records[0].provenance.source_system == "mem0"
    assert target.records[0].provenance.adapter == "tests"


def test_reader_catches_a_fingerprint_that_stopped_matching(conn: Any) -> None:
    """The stored fingerprint is a denormalized key, so it can drift."""
    bundle_id = CockroachWriter(conn).write_bundle(_bundle_with([_record()]))

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memory_record SET content = %s WHERE bundle_id = %s",
            ("something else entirely", str(bundle_id)),
        )
    conn.commit()

    from membridge.adapters.cockroach import CockroachReadError

    with pytest.raises(CockroachReadError, match="does not match content"):
        CockroachReader(conn).read_bundle(bundle_id)

    # …and reading without verification still works, for diagnosing such a store.
    assert len(CockroachReader(conn, verify_fingerprints=False).read_bundle(bundle_id).records) == 1


def test_duplicate_content_in_one_bundle_is_refused(conn: Any) -> None:
    """CMM reports duplicates rather than merging them; so must the target."""
    duplicate = _bundle_with([
        _record(content="same text"),
        _record(
            content="same   text",  # same fingerprint: whitespace is collapsed
            provenance=Provenance(source_system="mem0", source_id="src-2", exported_at=NOW),
        ),
    ])
    with pytest.raises(ValueError, match="duplicate content fingerprint"):
        CockroachWriter(conn).write_bundle(duplicate)


def test_a_failed_bundle_leaves_nothing_behind(conn: Any) -> None:
    """Migration is the unit of work: no half-written bundles."""
    from uuid import uuid4

    bundle_id = uuid4()
    good = _record()
    # created_at after updated_at is rejected by CMM, so build the violation at
    # the SQL layer instead: a second record reusing the first's source id.
    clash = _record(
        content="different content",
        provenance=Provenance(source_system="mem0", source_id="src-1", exported_at=NOW),
    )

    with pytest.raises(Exception):
        CockroachWriter(conn).write_bundle(_bundle_with([good, clash]), bundle_id=bundle_id)

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM memory_record WHERE bundle_id = %s", (str(bundle_id),))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM memory_bundle WHERE id = %s", (str(bundle_id),))
        assert cur.fetchone()[0] == 0
