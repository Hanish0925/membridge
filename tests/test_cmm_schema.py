"""Does CMM actually hold everything Mem0 hands us?

The load-bearing test here is `test_no_mem0_field_is_unaccounted_for`. MemBridge's
claim is that CMM is a lossless intermediate; the only way that claim stays true
as Mem0 changes is if adding a field to its output breaks a test. So the mapping
below is exhaustive by construction — every key in the ground-truth dump must be
named, and naming it means saying where in CMM it lands.

Ground truth comes from `tests/fixtures/mem0_raw_dump.json`, produced by
`scripts/dump_mem0.py` against mem0ai 2.0.15. Regenerate both together.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from membridge.cmm import (
    Actor,
    Embedding,
    MemoryBundle,
    MemoryRecord,
    Provenance,
    Scope,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DUMP_PATH = FIXTURE_DIR / "mem0_raw_dump.json"
META_PATH = FIXTURE_DIR / "mem0_raw_dump.meta.json"

#: Every field Mem0 2.0.15's `get_all()` can return, and where CMM puts it.
#:
#: Wider than what the fixture happens to contain: the demo conversation only
#: exercises `user_id` and `attributed_to`, but Mem0's `_get_all_from_vector_store`
#: promotes a larger set out of the Qdrant payload. Those were read off its source
#: rather than waiting for a fixture to surface them, because discovering
#: `expiration_date` during a real migration means having already dropped it.
MEM0_GET_ALL_FIELDS: dict[str, str] = {
    "id": "provenance.source_id",
    "memory": "content",
    "hash": "provenance.source_hash",
    "metadata": "metadata",
    "created_at": "created_at",
    "updated_at": "updated_at",
    "user_id": "scope.user_id",
    "agent_id": "scope.agent_id",
    "run_id": "scope.session_id",
    "attributed_to": "attribution",
    "role": "attribution (fallback)",
    "actor_id": "extensions['mem0']['actor_id']",
    "expiration_date": "expires_at (+ extensions['mem0'], source-exact)",
}

#: The subset the current fixture actually contains. Pinned separately so the
#: fixture drifting from the dump script is its own failure, distinct from Mem0
#: growing a field we have nowhere to put.
FIXTURE_FIELDS: set[str] = {
    "id",
    "memory",
    "hash",
    "metadata",
    "created_at",
    "updated_at",
    "user_id",
    "attributed_to",
}

#: Mem0 withholds vectors from `get_all()`, so an adapter has to read the Qdrant
#: payload directly. These are the keys it finds there (recorded in the dump's
#: meta file by the same script), and where each one goes.
MEM0_QDRANT_PAYLOAD_FIELDS: dict[str, str] = {
    "data": "content",  # same text as `memory` above, under a different name
    "hash": "provenance.source_hash",
    "created_at": "created_at",
    "updated_at": "updated_at",
    "user_id": "scope.user_id",
    "attributed_to": "attribution",
    "text_lemmatized": "extensions['mem0']['text_lemmatized']",
    # Surfaced by the multi-actor dump, not the Phase 0 one. Mem0 writes these
    # into the payload only on the `infer=False` path, so a fixture built the
    # default way can never show them.
    "agent_id": "scope.agent_id",
    "run_id": "scope.session_id",
    "role": "attribution (fallback)",
    "actor_id": "extensions['mem0']['actor_id']",
    "expiration_date": "expires_at (+ extensions['mem0'], source-exact)",
}


@pytest.fixture(scope="module")
def dump() -> list[dict[str, Any]]:
    raw = json.loads(DUMP_PATH.read_text(encoding="utf-8"))
    return raw["results"] if isinstance(raw, dict) and "results" in raw else raw


@pytest.fixture(scope="module")
def meta() -> dict[str, Any]:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def to_cmm(record: dict[str, Any], *, source_version: str) -> MemoryRecord:
    """The explicit mem0 -> CMM mapping this test file asserts against.

    Kept here rather than imported from an adapter on purpose: the accounting
    test should fail when Mem0's shape changes, not when the adapter changes.
    """
    return MemoryRecord(
        content=record["memory"],
        scope=Scope(user_id=record["user_id"]),
        attribution=Actor(record["attributed_to"]),
        created_at=datetime.fromisoformat(record["created_at"]),
        updated_at=datetime.fromisoformat(record["updated_at"]),
        metadata=record["metadata"],
        provenance=Provenance(
            source_system="mem0",
            source_id=record["id"],
            source_version=source_version,
            source_hash=record["hash"],
            exported_at=datetime.now(timezone.utc),
            adapter="tests.test_cmm_schema.to_cmm",
        ),
    )


# --- the accounting tests -------------------------------------------------


def test_no_mem0_field_is_unaccounted_for(dump: list[dict[str, Any]]) -> None:
    observed: set[str] = set()
    for record in dump:
        observed.update(record.keys())

    unmapped = observed - set(MEM0_GET_ALL_FIELDS)
    assert not unmapped, (
        f"Mem0 returns field(s) {sorted(unmapped)} that CMM has no home for. "
        "Either map them to a typed field or preserve them under "
        "extensions['mem0'], then add them to MEM0_GET_ALL_FIELDS."
    )

    assert observed == FIXTURE_FIELDS, (
        f"the fixture's field set changed (now {sorted(observed)}); regenerate it "
        "with scripts/dump_mem0.py and update FIXTURE_FIELDS deliberately."
    )
    assert FIXTURE_FIELDS <= set(MEM0_GET_ALL_FIELDS)


def test_no_qdrant_payload_field_is_unaccounted_for(meta: dict[str, Any]) -> None:
    probe = meta["embeddings"]["vector_store_probe"]
    if not probe.get("reachable"):
        pytest.skip("vector store was not reachable when the fixture was generated")

    observed = set(probe["payload_keys"])
    unmapped = observed - set(MEM0_QDRANT_PAYLOAD_FIELDS)
    assert not unmapped, (
        f"Qdrant payload carries key(s) {sorted(unmapped)} with no CMM destination."
    )


def test_get_all_still_withholds_embeddings(meta: dict[str, Any]) -> None:
    """A regression guard on the finding that shapes the adapter's design.

    If Mem0 ever starts returning vectors from `get_all()`, the mem0 adapter can
    stop reaching around it into Qdrant — and we want to be told, not to keep the
    workaround forever out of habit.
    """
    assert meta["embeddings"]["returned_by_get_all"] is False


# --- the schema holds the real data ---------------------------------------


def test_every_dump_record_round_trips(dump: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    version = f"mem0ai {meta['sdk_versions']['mem0ai']}"
    records = [to_cmm(r, source_version=version) for r in dump]

    assert len(records) == meta["records_extracted"]

    for cmm, source in zip(records, dump):
        assert cmm.content == source["memory"]
        assert cmm.provenance.source_id == source["id"]
        assert cmm.scope.user_id == source["user_id"]
        assert cmm.attribution is Actor(source["attributed_to"])
        # Mem0 emits null metadata on every record in the ground-truth dump.
        assert cmm.metadata == {}

    bundle = MemoryBundle(
        source_system="mem0",
        exported_at=datetime.now(timezone.utc),
        records=records,
    )
    assert len(bundle.fingerprints()) == len(records)
    assert bundle.embedding_space is None  # this dump carries no vectors

    # A bundle must survive JSON, since that is how it moves between adapters.
    reloaded = MemoryBundle.model_validate_json(bundle.model_dump_json())
    assert reloaded.fingerprints().keys() == bundle.fingerprints().keys()


# --- invariants -----------------------------------------------------------


def _record(**overrides: Any) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    kwargs: dict[str, Any] = {
        "content": "User's name is John.",
        "scope": Scope(user_id="john_001"),
        "created_at": now,
        "updated_at": now,
        "provenance": Provenance(
            source_system="mem0", source_id="abc", exported_at=now
        ),
    }
    kwargs.update(overrides)
    return MemoryRecord(**kwargs)


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _record(created_at=datetime(2026, 8, 4, 10, 39))


def test_timestamps_are_normalized_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    stamp = datetime(2026, 8, 4, 16, 9, tzinfo=ist)
    record = _record(created_at=stamp, updated_at=stamp)
    assert record.created_at.tzinfo is timezone.utc
    assert record.created_at.hour == 10


def test_updated_before_created_is_rejected() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="precedes"):
        _record(created_at=now, updated_at=now - timedelta(seconds=1))


def test_unscoped_memory_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        Scope()


def test_attribution_defaults_to_unknown_not_user() -> None:
    assert _record().attribution is Actor.UNKNOWN


def test_null_metadata_becomes_empty_dict() -> None:
    assert _record(metadata=None).metadata == {}


def test_empty_content_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        _record(content="   ")


def test_unknown_fields_are_refused_rather_than_carried_silently() -> None:
    """Unmapped vendor data belongs in `extensions`, not loose on the record."""
    with pytest.raises(ValidationError):
        _record(text_lemmatized="user name be john")


def test_extensions_preserve_vendor_fields_verbatim() -> None:
    record = _record(extensions={"mem0": {"text_lemmatized": "user name be john"}})
    reloaded = MemoryRecord.model_validate_json(record.model_dump_json())
    assert reloaded.extensions["mem0"]["text_lemmatized"] == "user name be john"


# --- fingerprints ---------------------------------------------------------


def test_fingerprint_ignores_everything_a_migration_may_change() -> None:
    now = datetime.now(timezone.utc)
    later = now + timedelta(days=1)
    a = _record(created_at=now, updated_at=now)
    b = _record(
        created_at=later,
        updated_at=later,
        scope=Scope(user_id="someone_else"),
        provenance=Provenance(
            source_system="cockroach", source_id="zzz", exported_at=later
        ),
    )
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_normalizes_whitespace_and_unicode() -> None:
    spaced = _record(content="  User's   name\nis John.  ")
    assert spaced.fingerprint() == _record(content="User's name is John.").fingerprint()

    # 'é' as one codepoint vs 'e' + combining accent must not read as data loss.
    composed = _record(content="Café booking")
    decomposed = _record(content="Café booking")
    assert composed.fingerprint() == decomposed.fingerprint()


def test_fingerprint_distinguishes_different_content() -> None:
    assert _record().fingerprint() != _record(content="User's name is Jane.").fingerprint()


# --- embeddings -----------------------------------------------------------


def test_embedding_dim_must_match_vector_length() -> None:
    with pytest.raises(ValidationError, match="declared dim"):
        Embedding(vector=[0.1, 0.2], model="all-MiniLM-L6-v2", dim=384)


def test_bundle_refuses_to_mix_embedding_spaces() -> None:
    now = datetime.now(timezone.utc)
    mini = Embedding(vector=[0.0] * 384, model="all-MiniLM-L6-v2", dim=384)
    other = Embedding(vector=[0.0] * 768, model="all-mpnet-base-v2", dim=768)
    bundle = MemoryBundle(
        source_system="mem0",
        exported_at=now,
        records=[
            _record(content="a", embedding=mini),
            _record(content="b", embedding=other),
        ],
    )
    with pytest.raises(ValueError, match="not comparable"):
        _ = bundle.embedding_space


def test_bundle_reports_its_single_embedding_space() -> None:
    mini = Embedding(vector=[0.0] * 384, model="all-MiniLM-L6-v2", dim=384)
    bundle = MemoryBundle(
        source_system="mem0",
        exported_at=datetime.now(timezone.utc),
        records=[_record(embedding=mini)],
    )
    assert bundle.embedding_space == ("all-MiniLM-L6-v2", 384)


def test_bundle_rejects_duplicate_ids() -> None:
    record = _record()
    with pytest.raises(ValidationError, match="duplicate record id"):
        MemoryBundle(
            source_system="mem0",
            exported_at=datetime.now(timezone.utc),
            records=[record, record],
        )


def test_bundle_reports_duplicate_content_rather_than_deduplicating() -> None:
    bundle = MemoryBundle(
        source_system="mem0",
        exported_at=datetime.now(timezone.utc),
        records=[_record(), _record()],  # distinct ids, identical content
    )
    with pytest.raises(ValueError, match="duplicate content fingerprint"):
        bundle.fingerprints()
