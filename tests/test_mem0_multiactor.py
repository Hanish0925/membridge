"""The multi-actor fixture: ground truth for the mappings that had none.

`tests/test_mem0_adapter.py` already covers `agent_id`, `run_id`, `role`,
`actor_id` and `expiration_date` — but only against records this repo builds by
hand (`_raw(**overrides)`). A hand-built record proves the adapter is
self-consistent; it cannot prove Mem0 ever emits that shape. Until this fixture
existed, those five mappings rested entirely on reading Mem0's source, which is
exactly the sort of unfalsifiable claim MemBridge is written to avoid making.

Ground truth comes from `tests/fixtures/mem0_multiactor_dump.json`, produced by
`scripts/dump_mem0_multiactor.py` against a real Mem0 2.0.15 + Qdrant store.

The fixture is generated with `infer=False`, which matters twice over. It is the
only Mem0 path that writes `role` and `actor_id` at all, and because it never
invokes the extraction LLM, this fixture is deterministic — unlike the Phase 0
dump, regenerating it cannot change the content.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from membridge.adapters.mem0 import bundle_from_dump, record_to_cmm
from membridge.cmm import Actor

from tests.test_cmm_schema import MEM0_GET_ALL_FIELDS, MEM0_QDRANT_PAYLOAD_FIELDS

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DUMP_PATH = FIXTURE_DIR / "mem0_multiactor_dump.json"
META_PATH = FIXTURE_DIR / "mem0_multiactor_dump.meta.json"


@pytest.fixture(scope="module")
def dump() -> dict[str, Any]:
    return json.loads(DUMP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records(dump: dict[str, Any]) -> list[dict[str, Any]]:
    return dump["results"] if isinstance(dump, dict) and "results" in dump else dump


@pytest.fixture(scope="module")
def meta() -> dict[str, Any]:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def _by_content(records: list[dict[str, Any]], needle: str) -> dict[str, Any]:
    matches = [r for r in records if needle in r["memory"]]
    assert len(matches) == 1, f"expected exactly one record containing {needle!r}"
    return matches[0]


# --- accounting -----------------------------------------------------------


def test_no_field_in_the_multiactor_dump_is_unaccounted_for(
    records: list[dict[str, Any]],
) -> None:
    observed: set[str] = set()
    for record in records:
        observed.update(record.keys())

    unmapped = observed - set(MEM0_GET_ALL_FIELDS)
    assert not unmapped, (
        f"the multi-actor dump carries field(s) {sorted(unmapped)} with no CMM "
        "destination; map them or preserve them under extensions['mem0']."
    )


def test_the_multiactor_dump_reaches_fields_phase_zero_could_not(
    records: list[dict[str, Any]],
) -> None:
    """The point of this fixture, stated as an assertion.

    If a future regeneration stops covering these, the fixture has silently
    lost its reason to exist and the five mappings go back to being unproven.
    """
    observed: set[str] = set()
    for record in records:
        observed.update(record.keys())

    assert {"agent_id", "run_id", "role", "actor_id", "expiration_date"} <= observed


def test_qdrant_payload_keys_are_accounted_for(meta: dict[str, Any]) -> None:
    unmapped = set(meta["qdrant_payload_keys"]) - set(MEM0_QDRANT_PAYLOAD_FIELDS)
    assert not unmapped, (
        f"Qdrant payload carries key(s) {sorted(unmapped)} with no CMM destination."
    )


# --- the two attribution paths are mutually exclusive ---------------------


def test_infer_false_writes_role_and_actor_id_but_never_attributed_to(
    records: list[dict[str, Any]],
) -> None:
    """Mem0's two attribution paths cannot both be exercised by one store.

    `infer=True` sets `attributed_to` from the extraction LLM and never writes
    `role`/`actor_id`; `infer=False` does the reverse. The adapter's
    `attributed_to or role` fallback therefore spans two paths that never
    co-occur — which is only safe to rely on while this stays true of Mem0.
    """
    for record in records:
        assert record.get("role"), "infer=False must set role on every record"
        assert "attributed_to" not in record, (
            "Mem0 now writes attributed_to on the infer=False path; the adapter's "
            "attribution fallback assumes these two paths are exclusive."
        )


def test_the_phase_zero_fixture_is_the_other_path() -> None:
    """Guards the claim above from the other side."""
    phase0 = json.loads((FIXTURE_DIR / "mem0_raw_dump.json").read_text(encoding="utf-8"))
    rows = phase0["results"] if isinstance(phase0, dict) else phase0

    for row in rows:
        assert row.get("attributed_to"), "the Phase 0 dump is the attributed_to path"
        assert "role" not in row
        assert "actor_id" not in row


# --- the mappings, now against real records -------------------------------


def test_scope_carries_all_three_identifiers(
    records: list[dict[str, Any]], meta: dict[str, Any]
) -> None:
    bundle = bundle_from_dump({"results": records})
    scope = meta["scope"]

    for record in bundle.records:
        assert record.scope.user_id == scope["user_id"]
        assert record.scope.agent_id == scope["agent_id"]
        # Mem0's run_id is CMM's session_id: same concept, and CMM does not
        # rename it back on the way out.
        assert record.scope.session_id == scope["run_id"]


def test_actor_id_is_preserved_verbatim_not_mapped_to_an_actor(
    records: list[dict[str, Any]],
) -> None:
    """`actor_id` is a free-form name, so it cannot become a CMM `Actor`.

    Mem0 populates it from a message's `name` key — undocumented, and the only
    route to it. Two distinct humans in one thread is what makes this visible:
    `role` says "user" for both, so `role` alone cannot tell them apart.
    """
    stamp = datetime.now(timezone.utc)
    seen: set[str] = set()

    for raw in records:
        mapped = record_to_cmm(raw, exported_at=stamp)
        actor_id = mapped.extensions["mem0"]["actor_id"]
        assert actor_id == raw["actor_id"]
        seen.add(actor_id)

    assert {"john", "maria", "support_bot"} <= seen, (
        "the fixture must keep more than one human actor, or it cannot show that "
        "attribution survives per-record rather than per-bundle"
    )


def test_role_becomes_the_attribution_when_attributed_to_is_absent(
    records: list[dict[str, Any]],
) -> None:
    stamp = datetime.now(timezone.utc)

    for raw in records:
        mapped = record_to_cmm(raw, exported_at=stamp)
        assert mapped.attribution is Actor(raw["role"])
        # role is a typed mapping *and* kept verbatim: the fallback is lossy in
        # principle (Mem0 may add roles CMM has no Actor for), so the source
        # string has to survive regardless.
        assert mapped.extensions["mem0"]["role"] == raw["role"]

    assert {r["role"] for r in records} >= {"user", "assistant"}


def test_expiry_maps_to_end_of_day_and_keeps_the_source_string(
    records: list[dict[str, Any]], meta: dict[str, Any]
) -> None:
    stamp = datetime.now(timezone.utc)
    expiring = [r for r in records if r.get("expiration_date")]
    assert len(expiring) == 2, "fixture should carry one expired and one live record"

    for raw in expiring:
        mapped = record_to_cmm(raw, exported_at=stamp)
        source_date = date.fromisoformat(raw["expiration_date"])

        assert mapped.expires_at is not None
        assert mapped.expires_at.date() == source_date
        # End of day, not midnight: Mem0 expires on `< today`, so midnight would
        # expire every memory a full day early.
        assert (mapped.expires_at.hour, mapped.expires_at.minute) == (23, 59)
        assert mapped.expires_at.tzinfo is not None
        # Mem0's granularity is a day and CMM's is an instant, so the exact
        # source string is kept rather than reconstructed from the instant.
        assert mapped.extensions["mem0"]["expiration_date"] == raw["expiration_date"]


def test_the_expired_record_is_expired_and_the_other_is_not(
    records: list[dict[str, Any]], meta: dict[str, Any]
) -> None:
    stamp = datetime.now(timezone.utc)
    dates = meta["expiration_dates"]

    expired = record_to_cmm(
        _by_content(records, "Rowdy Burgers"), exported_at=stamp
    )
    live = record_to_cmm(_by_content(records, "window seats"), exported_at=stamp)

    assert expired.extensions["mem0"]["expiration_date"] == dates["expired"]
    assert live.extensions["mem0"]["expiration_date"] == dates["future"]

    assert expired.is_expired() is True
    assert live.is_expired() is False


def test_records_without_expiry_do_not_acquire_one(records: list[dict[str, Any]]) -> None:
    stamp = datetime.now(timezone.utc)
    for raw in (r for r in records if not r.get("expiration_date")):
        assert record_to_cmm(raw, exported_at=stamp).expires_at is None


# --- what Mem0's default read would have cost -----------------------------


def test_mem0s_default_read_would_have_dropped_a_record(meta: dict[str, Any]) -> None:
    """The recorded cost of `show_expired=False`, in this exact store.

    Both reads were taken at dump time precisely so the difference is evidence
    rather than an assertion about Mem0's docs. A migration using Mem0's default
    would have silently left this record behind.
    """
    assert meta["records_hidden_by_default_expiry"] == 1
    assert meta["records_visible_by_default"] == meta["records_extracted"] - 1


def test_the_bundle_is_valid_and_every_record_is_distinct(
    records: list[dict[str, Any]], meta: dict[str, Any]
) -> None:
    bundle = bundle_from_dump({"results": records})

    assert len(bundle.records) == meta["records_extracted"]
    assert len(bundle.fingerprints()) == len(bundle.records)
    assert bundle.embedding_space is None  # a get_all() dump carries no vectors


def test_infer_false_stores_messages_verbatim(
    records: list[dict[str, Any]], meta: dict[str, Any]
) -> None:
    """One record per message, and the text is the message text.

    Worth pinning: it is what makes this fixture deterministic, and it is the
    sharpest available contrast with the Phase 0 dump, where an LLM turned four
    messages into three facts of its own wording.
    """
    assert meta["llm_invoked"] is False
    assert meta["records_extracted"] == meta["messages_ingested"]
