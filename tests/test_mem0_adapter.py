"""The Mem0 adapter, tested offline against the recorded fixture.

No live Mem0, Qdrant or Ollama is involved. The mapping runs against
`tests/fixtures/mem0_raw_dump.json`; the parts that need a live store (pagination
escalation, the vector-store reach-around) run against fakes that reproduce the
behaviours found by reading Mem0's source — chiefly that `get_all` truncates at
`top_k` with no signal, and that its Qdrant backend never returns vectors.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from membridge.adapters.mem0 import (
    ADAPTER_NAME,
    Mem0ReadError,
    Mem0Reader,
    bundle_from_dump,
    record_to_cmm,
)
from membridge.cmm import Actor

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DUMP_PATH = FIXTURE_DIR / "mem0_raw_dump.json"
META_PATH = FIXTURE_DIR / "mem0_raw_dump.meta.json"


@pytest.fixture(scope="module")
def dump() -> dict[str, Any]:
    return json.loads(DUMP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def meta() -> dict[str, Any]:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


# --- the recorded fixture round-trips -------------------------------------


def test_fixture_becomes_a_valid_bundle(dump: dict[str, Any], meta: dict[str, Any]) -> None:
    bundle = bundle_from_dump(dump, source_version=f"mem0ai {meta['sdk_versions']['mem0ai']}")

    assert len(bundle.records) == meta["records_extracted"]
    assert bundle.embedding_space is None  # a get_all() dump carries no vectors
    assert len(bundle.fingerprints()) == len(bundle.records)

    for record, raw in zip(bundle.records, dump["results"]):
        assert record.content == raw["memory"]
        assert record.scope.user_id == raw["user_id"]
        assert record.attribution is Actor(raw["attributed_to"])
        assert record.provenance.source_id == raw["id"]
        assert record.provenance.source_hash == raw["hash"]
        assert record.provenance.adapter == ADAPTER_NAME
        assert record.metadata == {}       # mem0 emits null throughout this dump
        assert record.expires_at is None   # nothing in this dump expires


def test_adapter_agrees_with_the_independent_mapping(dump: dict[str, Any]) -> None:
    """Double-entry check against the mapping in test_cmm_schema.py.

    That module spells out mem0 -> CMM by hand so the field-accounting test fails
    on Mem0's changes rather than on adapter refactors. Two mappings are only
    worth keeping if they agree, so: assert they do.
    """
    from tests.test_cmm_schema import to_cmm

    stamp = datetime.now(timezone.utc)
    for raw in dump["results"]:
        mine = record_to_cmm(raw, exported_at=stamp, source_version="mem0ai 2.0.15")
        theirs = to_cmm(raw, source_version="mem0ai 2.0.15")

        assert mine.content == theirs.content
        assert mine.scope == theirs.scope
        assert mine.attribution == theirs.attribution
        assert mine.created_at == theirs.created_at
        assert mine.updated_at == theirs.updated_at
        assert mine.metadata == theirs.metadata
        assert mine.provenance.source_id == theirs.provenance.source_id
        assert mine.provenance.source_hash == theirs.provenance.source_hash
        assert mine.fingerprint() == theirs.fingerprint()


# --- fields the fixture does not contain ----------------------------------


def _raw(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "77608922-acfc-4e24-a091-7536e3be47c2",
        "memory": "User's name is John.",
        "hash": "bf0a7f08964abb09c980e6927909a0bd",
        "metadata": None,
        "created_at": "2026-08-04T10:39:26.534018+00:00",
        "updated_at": "2026-08-04T10:39:26.534018+00:00",
        "user_id": "john_001",
        "attributed_to": "user",
    }
    base.update(overrides)
    return base


def _map(**overrides: Any):
    return record_to_cmm(_raw(**overrides), exported_at=datetime.now(timezone.utc))


def test_agent_and_run_ids_reach_scope() -> None:
    record = _map(agent_id="agent_7", run_id="run_42")
    assert record.scope.agent_id == "agent_7"
    assert record.scope.session_id == "run_42"  # mem0 calls this run_id


def test_expiration_date_becomes_expires_at() -> None:
    record = _map(expiration_date="2026-08-20")
    assert record.expires_at is not None
    assert record.expires_at.date().isoformat() == "2026-08-20"
    # Source-exact form is kept too: mem0's granularity is a day, CMM's an instant.
    assert record.extensions["mem0"]["expiration_date"] == "2026-08-20"


def test_expiry_lands_at_end_of_day_not_midnight() -> None:
    """Mem0 expires a memory once the date is *past*, so midnight is a day early."""
    record = _map(expiration_date="2026-08-20")
    assert not record.is_expired(now=datetime(2026, 8, 20, 23, 0, tzinfo=timezone.utc))
    assert record.is_expired(now=datetime(2026, 8, 21, 0, 30, tzinfo=timezone.utc))


def test_text_lemmatized_is_preserved_from_the_qdrant_payload() -> None:
    record = record_to_cmm(
        _raw(),
        exported_at=datetime.now(timezone.utc),
        payload={"text_lemmatized": "user name be john", "data": "User's name is John."},
    )
    assert record.extensions["mem0"]["text_lemmatized"] == "user name be john"


def test_multi_actor_fields_are_preserved_not_guessed() -> None:
    record = _map(attributed_to=None, role="tool", actor_id="calendar_api")
    assert record.attribution is Actor.TOOL
    assert record.extensions["mem0"]["actor_id"] == "calendar_api"


def test_unmodelled_attribution_is_kept_rather_than_mapped_wrongly() -> None:
    record = _map(attributed_to="supervisor")
    assert record.attribution is Actor.UNKNOWN
    assert record.extensions["mem0"]["attributed_to_raw"] == "supervisor"


def test_missing_attribution_is_unknown_not_user() -> None:
    assert _map(attributed_to=None).attribution is Actor.UNKNOWN


def test_naive_timestamps_are_refused_rather_than_assumed_utc() -> None:
    with pytest.raises(Mem0ReadError, match="naive"):
        _map(created_at="2026-08-04T10:39:26.534018")


def test_missing_created_at_is_refused() -> None:
    with pytest.raises(Mem0ReadError, match="no created_at"):
        _map(created_at=None)


def test_missing_updated_at_falls_back_to_created_at() -> None:
    record = _map(updated_at=None)
    assert record.updated_at == record.created_at


# --- fakes standing in for a live Mem0 ------------------------------------


class FakePoint:
    def __init__(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class FakeQdrantClient:
    def __init__(self, points: dict[str, FakePoint]) -> None:
        self.points = points

    def retrieve(self, *, collection_name: str, ids: list[str], **_: Any) -> list[FakePoint]:
        return [self.points[i] for i in ids if i in self.points]


class FakeMemory:
    """Reproduces the two Mem0 behaviours that matter for a complete read.

    `get_all` truncates at `top_k` and says nothing about it; the vector store is
    reachable only around Mem0, never through it.
    """

    def __init__(self, records: list[dict[str, Any]], *, vectors: bool = True) -> None:
        self.records = records
        self.collection_name = "membridge_mem0"
        self.calls: list[int] = []
        points = {
            str(r["id"]): FakePoint(
                str(r["id"]),
                [0.01] * 384,
                {"text_lemmatized": f"lemma {i}", "data": r["memory"]},
            )
            for i, r in enumerate(records)
        }
        self.vector_store = type(
            "VS", (), {"client": FakeQdrantClient(points if vectors else {})}
        )()
        self.embedding_model = type(
            "EM",
            (),
            {"config": type("C", (), {"model": "all-MiniLM-L6-v2", "embedding_dims": 384})()},
        )()

    def get_all(self, *, filters: dict[str, Any], top_k: int = 20, show_expired: bool = False):
        self.calls.append(top_k)
        assert show_expired, "a migration must not let mem0 hide expired records"
        matching = [
            r for r in self.records if all(r.get(k) == v for k, v in filters.items())
        ]
        return {"results": matching[:top_k]}


def _many(n: int) -> list[dict[str, Any]]:
    return [_raw(id=f"{i:08d}-0000-0000-0000-000000000000", memory=f"Fact {i}.") for i in range(n)]


def test_read_escalates_past_mem0s_silent_truncation() -> None:
    """Mem0's `get_all` defaults to top_k=20 and gives no signal when it truncates."""
    memory = FakeMemory(_many(700))
    bundle = Mem0Reader(memory).read_bundle(user_id="john_001")

    assert len(bundle.records) == 700
    assert memory.calls == [512, 1024]  # asked again because 512 came back full


def test_a_single_full_read_still_gets_verified() -> None:
    memory = FakeMemory(_many(512))
    bundle = Mem0Reader(memory).read_bundle(user_id="john_001")
    assert len(bundle.records) == 512
    assert memory.calls == [512, 1024], "an exactly-full page is indistinguishable from a truncated one"


def test_embeddings_come_from_the_vector_store() -> None:
    bundle = Mem0Reader(FakeMemory(_many(3))).read_bundle(user_id="john_001")

    assert bundle.embedding_space == ("all-MiniLM-L6-v2", 384)
    for record in bundle.records:
        assert record.embedding is not None
        assert len(record.embedding.vector) == 384
        # The reach-around also recovers the payload field mem0's API withholds.
        assert "text_lemmatized" in record.extensions["mem0"]


def test_records_missing_from_the_vector_store_are_an_error() -> None:
    memory = FakeMemory(_many(3), vectors=False)
    with pytest.raises(Mem0ReadError, match="absent from the vector store"):
        Mem0Reader(memory).read_bundle(user_id="john_001")


def test_embeddings_can_be_skipped() -> None:
    bundle = Mem0Reader(FakeMemory(_many(3), vectors=False)).read_bundle(
        user_id="john_001", include_embeddings=False
    )
    assert bundle.embedding_space is None
    assert len(bundle.records) == 3


def test_unscoped_read_is_refused() -> None:
    with pytest.raises(Mem0ReadError, match="at least one"):
        Mem0Reader(FakeMemory([])).read_bundle()


def test_scope_filters_are_passed_through() -> None:
    records = _many(2)
    records[0]["agent_id"] = "agent_7"
    bundle = Mem0Reader(FakeMemory(records)).read_bundle(user_id="john_001", agent_id="agent_7")
    assert len(bundle.records) == 1
    assert bundle.records[0].scope.agent_id == "agent_7"


def test_reader_detects_provenance_from_the_live_instance() -> None:
    reader = Mem0Reader(FakeMemory(_many(1)))
    assert reader.embedding_model == "all-MiniLM-L6-v2"
    assert reader.embedding_dim == 384
    assert reader.source_version is not None and reader.source_version.startswith("mem0ai ")


def test_expired_records_are_carried_not_dropped() -> None:
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    memory = FakeMemory([_raw(expiration_date=yesterday)])
    bundle = Mem0Reader(memory).read_bundle(user_id="john_001")

    assert len(bundle.records) == 1, "an expired memory is still data the source holds"
    assert bundle.records[0].is_expired()
