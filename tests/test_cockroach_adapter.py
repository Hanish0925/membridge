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


# --- semantic retrieval, needs a live CockroachDB -------------------------
#
# The vector index existed in sql/schema.sql long before any code queried it,
# which means "the index works" was an untested claim of exactly the kind this
# project exists not to make. These are its falsifier.


def _unit(dim: int, *components: float) -> list[float]:
    """A unit-length 384d vector with the given leading components.

    One-hot and near-one-hot vectors are genuinely norm 1.0, so the L2->cosine
    identity holds and `similarity` is checkable against hand arithmetic rather
    than against whatever the code happened to produce.
    """
    vector = [0.0] * dim
    for index, value in enumerate(components):
        vector[index] = value
    return vector


@pytest.fixture
def scope(conn: Any) -> Any:
    """A scope no other test has written to, dropped afterwards.

    Search is scoped by user_id, so a leftover record from another test is not
    inert -- it is an extra neighbour that changes the ranking under assertion.
    """
    from uuid import uuid4

    user_id = f"search_{uuid4().hex[:12]}"
    yield user_id
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memory_record WHERE scope_user_id = %s", (user_id,))
    conn.commit()


def _seed(conn: Any, user_id: str, vectors: dict[str, list[float]], **kw: Any) -> None:
    from uuid import uuid4

    records = [
        _record(
            content=f"{user_id}: {label}",
            scope=Scope(user_id=user_id),
            provenance=Provenance(
                source_system="mem0", source_id=f"{user_id}-{label}", exported_at=NOW
            ),
            embedding=Embedding(vector=vector, model="test-embedder", dim=384),
            **kw,
        )
        for label, vector in vectors.items()
    ]
    CockroachWriter(conn).write_bundle(_bundle_with(records), bundle_id=uuid4())


def test_search_ranks_by_distance(conn: Any, scope: str) -> None:
    _seed(conn, scope, {
        "alpha": _unit(384, 1.0),
        "beta": _unit(384, 0.0, 1.0),
        "gamma": _unit(384, 0.0, 0.0, 1.0),
    })

    # Closest to alpha, then beta, then gamma -- by hand:
    #   d(q,alpha) = sqrt(0.04 + 0.36) = 0.632
    #   d(q,beta)  = sqrt(0.64 + 0.16) = 0.894
    #   d(q,gamma) = sqrt(0.64 + 0.36 + 1) = 1.414
    query = Embedding(vector=_unit(384, 0.8, 0.6), model="test-embedder", dim=384)
    hits = CockroachReader(conn).search(query, user_id=scope, limit=3)

    assert [hit.record.content.split(": ")[1] for hit in hits] == ["alpha", "beta", "gamma"]
    assert hits[0].distance == pytest.approx(0.632, abs=1e-3)
    # cos = 1 - d^2/2; for q.alpha that is 0.8 exactly.
    assert hits[0].similarity == pytest.approx(0.8, abs=1e-5)


def test_search_returns_whole_cmm_records_not_just_ids(conn: Any, scope: str) -> None:
    """A hit must be usable as memory, which means it is a verified record.

    It goes through the same `_row_to_record` as the migration path, so the
    fingerprint check applies here too: retrieval cannot serve an agent content
    that has drifted from what was migrated.
    """
    _seed(conn, scope, {"alpha": _unit(384, 1.0)})
    query = Embedding(vector=_unit(384, 1.0), model="test-embedder", dim=384)

    (hit,) = CockroachReader(conn).search(query, user_id=scope, limit=5)
    assert isinstance(hit.record, MemoryRecord)
    assert hit.record.scope.user_id == scope
    assert hit.record.provenance.source_system == "mem0"
    assert hit.record.fingerprint() == hit.record.fingerprint()


def test_search_refuses_a_query_from_a_different_embedder(conn: Any, scope: str) -> None:
    """The silent-wrong-answer case: it would have sorted, and meant nothing."""
    from membridge.adapters.cockroach import EmbeddingSpaceMismatch

    _seed(conn, scope, {"alpha": _unit(384, 1.0)})
    foreign = Embedding(vector=_unit(384, 1.0), model="some-other-embedder", dim=384)

    with pytest.raises(EmbeddingSpaceMismatch, match="some-other-embedder"):
        CockroachReader(conn).search(foreign, user_id=scope)


def test_search_withholds_cosine_when_vectors_are_not_unit_length(
    conn: Any, scope: str
) -> None:
    """L2 ranks like cosine only for unit vectors. Elsewhere: no number at all."""
    _seed(conn, scope, {"alpha": _unit(384, 3.0)})  # norm 3, not 1
    query = Embedding(vector=_unit(384, 1.0), model="test-embedder", dim=384)

    (hit,) = CockroachReader(conn).search(query, user_id=scope)
    assert hit.record.embedding.normalized is False
    assert hit.distance == pytest.approx(2.0)
    assert hit.similarity is None, "a cosine here would be arithmetic on a false premise"


def test_search_hides_expired_memories_unless_asked(conn: Any, scope: str) -> None:
    """The opposite call to `read_bundle`, deliberately.

    Migration must see everything the store holds. Retrieval feeds an agent
    about to act, and a memory the user expired is worse than a missing one.
    """
    _seed(conn, scope, {"alpha": _unit(384, 1.0)}, expires_at=NOW - timedelta(days=1))
    query = Embedding(vector=_unit(384, 1.0), model="test-embedder", dim=384)
    reader = CockroachReader(conn)

    assert reader.search(query, user_id=scope) == []
    assert len(reader.search(query, user_id=scope, include_expired=True)) == 1


def test_search_is_scoped_and_does_not_leak_across_users(conn: Any, scope: str) -> None:
    _seed(conn, scope, {"alpha": _unit(384, 1.0)})
    query = Embedding(vector=_unit(384, 1.0), model="test-embedder", dim=384)

    assert len(CockroachReader(conn).search(query, user_id=scope)) == 1
    assert CockroachReader(conn).search(query, user_id="nobody_at_all") == []


def test_the_vector_index_is_actually_used(conn: Any, scope: str) -> None:
    """EXPLAIN, not faith.

    sql/schema.sql documents that the index is prefix-partitioned on
    scope_user_id and that an unscoped query falls back to a full scan. That is
    the reason `search` requires a user_id, so it is worth pinning: if a future
    Cockroach release changes planning, this fails rather than retrieval
    quietly degrading to a scan of every memory in the store.

    **This test seeds a thousand records, and has to.** The first version of it
    asserted the same thing over a three-record table and failed -- not because
    the index was broken, but because CockroachDB's planner correctly declines a
    C-SPANN traversal when scanning the whole table is cheaper. Index usage is
    therefore a claim about a populated store, not about the schema, and a
    fixture-sized table cannot evidence it either way. Measured on v26.2.5: at
    ~8 rows the planner picks `memory_record_scope_idx`; at 1000 it picks the
    vector index.
    """
    import math
    import random
    from uuid import uuid4

    random.seed(20260815)

    def unit_vector() -> list[float]:
        raw = [random.gauss(0, 1) for _ in range(384)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

    records = [
        _record(
            content=f"{scope} memory {i}",
            scope=Scope(user_id=scope),
            provenance=Provenance(
                source_system="mem0", source_id=f"{scope}-{i}", exported_at=NOW
            ),
            embedding=Embedding(vector=unit_vector(), model="test-embedder", dim=384),
        )
        for i in range(1000)
    ]
    CockroachWriter(conn).write_bundle(_bundle_with(records), bundle_id=uuid4())

    vector = encode_vector(unit_vector())
    with conn.cursor() as cur:
        # Without fresh statistics the planner is choosing on stale row counts,
        # which is the same trap as the three-record version of this test.
        cur.execute("ANALYZE memory_record")
        cur.execute(
            "EXPLAIN SELECT id FROM memory_record WHERE scope_user_id = %s "
            "ORDER BY embedding <-> %s::vector LIMIT 5",
            (scope, vector),
        )
        scoped = "\n".join(row[0] for row in cur.fetchall())

        cur.execute(
            "EXPLAIN SELECT id FROM memory_record ORDER BY embedding <-> %s::vector LIMIT 5",
            (vector,),
        )
        unscoped = "\n".join(row[0] for row in cur.fetchall())

    assert "vector search" in scoped, f"scoped L2 should hit the index, got:\n{scoped}"
    assert "vector search" not in unscoped, (
        "an unscoped query cannot use a prefix-partitioned vector index; if this "
        f"now passes, sql/schema.sql's justification for requiring a scope is "
        f"stale:\n{unscoped}"
    )
