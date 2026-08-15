"""Ground truth for the Mem0 fields the first dump could not reach.

`scripts/dump_mem0.py` establishes what Mem0 does with a single-user inferred
conversation. It cannot exercise five fields the adapter nevertheless maps —
`agent_id`, `run_id`, `role`, `actor_id`, `expiration_date` — so those mappings
were written from reading Mem0's source alone. In a project whose whole claim is
that migration fidelity should be falsifiable, that is the one place a mapping
was asserted rather than demonstrated. This script demonstrates it.

The reason a second script is needed rather than a second conversation is that
Mem0 has **two mutually exclusive attribution paths**, and no single `add()` call
can produce both (see `_add_to_vector_store` in mem0/memory/main.py):

  * `infer=True` (the default, and what dump_mem0.py uses) runs the extraction
    LLM and sets `attributed_to` from the model's own output. It never writes
    `role` and never writes `actor_id`.
  * `infer=False` stores each message verbatim, setting `role` from the message
    and `actor_id` from the message's `name` key. It never writes
    `attributed_to`.

So the adapter's `attributed_to or role` fallback spans two paths that cannot
co-occur, and only this script covers the second one. That is worth knowing
before a migration reports "attribution preserved" on a store written the other
way.

A useful side effect: `infer=False` never calls the LLM, so unlike the Phase 0
dump this fixture is fully deterministic — the memory text is the message text.
No Ollama, no extraction nondeterminism, no model pinning required to reproduce.

Writes its own store (`./data/qdrant_multiactor`, own collection) so that
regenerating this fixture cannot disturb the Phase 0 ground truth.

Usage:
    uv run python scripts/dump_mem0_multiactor.py
    uv run python scripts/dump_mem0_multiactor.py --keep
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from membridge.adapters.mem0 import EMBEDDING_DIM, EMBEDDING_MODEL, build_config

REPO_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = REPO_ROOT / "data" / "qdrant_multiactor"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "mem0_multiactor_dump.json"
META_PATH = REPO_ROOT / "tests" / "fixtures" / "mem0_multiactor_dump.meta.json"

#: A separate collection from the Phase 0 dump's, so the two ground truths are
#: independent artifacts and neither regeneration can corrupt the other.
COLLECTION = "membridge_mem0_multiactor"

#: All three scope identifiers at once. CMM maps user_id/agent_id to Scope's own
#: fields and run_id to `session_id`; the Phase 0 fixture only ever carried
#: user_id, so the other two columns have never been shown to survive a read.
USER_ID = "john_001"
AGENT_ID = "support_agent_v2"
RUN_ID = "thread_8891"

#: Two distinct human actors plus an assistant, each carrying `name`. `name` is
#: the *only* way to populate `actor_id`, and Mem0 documents neither that fact
#: nor the field. Two humans is the point: a single-actor thread cannot show
#: whether attribution is per-record or per-bundle.
CONVERSATION: list[dict[str, str]] = [
    {
        "role": "user",
        "name": "john",
        "content": "My name is John and I am allergic to shellfish.",
    },
    {
        "role": "assistant",
        "name": "support_bot",
        "content": "Noted, John. I have flagged shellfish on your account.",
    },
    {
        "role": "user",
        "name": "maria",
        "content": "This is Maria, John's partner. I am vegetarian.",
    },
]

#: Added separately because `expiration_date` is a parameter of the whole `add()`
#: call, not of a message, so a mixed-expiry store takes more than one call.
EXPIRING_MESSAGE = {
    "role": "user",
    "name": "john",
    "content": "I have a dinner reservation at Rowdy Burgers this Friday.",
}

TRACKED_PACKAGES = [
    "mem0ai",
    "sentence-transformers",
    "qdrant-client",
    "pydantic",
]


def sdk_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def payload_keys(memory: Any) -> list[str]:
    """The union of Qdrant payload keys actually written.

    Read straight from the vector store rather than from `get_all()`, because
    the whole reason this project reaches past Mem0 is that its read API
    withholds payload fields (`text_lemmatized`) and every vector.
    """
    client = memory.vector_store.client
    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=100,
        with_vectors=False,
        with_payload=True,
    )
    keys: set[str] = set()
    for point in points:
        keys.update((point.payload or {}).keys())
    return sorted(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="do not wipe ./data/qdrant_multiactor before ingesting",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if not args.keep and QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
        print(f"[reset] removed {QDRANT_PATH}")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    from mem0 import Memory

    config = build_config(QDRANT_PATH, collection=COLLECTION)
    memory = Memory.from_config(config)

    print(f"[config] embedder : {EMBEDDING_MODEL} ({EMBEDDING_DIM}d, local)")
    print(f"[config] store    : qdrant @ {QDRANT_PATH} / {COLLECTION}")
    print("[config] infer    : False (no LLM; message text is stored verbatim)")

    started = datetime.now(timezone.utc)

    # Expiry dates are computed relative to today so the fixture keeps meaning
    # one expired and one live record forever. Mem0 expires on `< today`, so
    # yesterday is expired and tomorrow is not; recording literal dates would
    # mean this fixture silently changes meaning the day after it was written.
    today = date.today()
    expired_on = today - timedelta(days=1)
    expires_on = today + timedelta(days=365)

    # infer=False is what routes us down the role/actor_id branch. It is also
    # why no `widen_ollama_context` call appears here: the LLM is never invoked.
    added = memory.add(
        CONVERSATION,
        user_id=USER_ID,
        agent_id=AGENT_ID,
        run_id=RUN_ID,
        infer=False,
    )
    print(f"[ingest] base      : {json.dumps(added, default=str)[:300]}")

    added_expired = memory.add(
        [EXPIRING_MESSAGE],
        user_id=USER_ID,
        agent_id=AGENT_ID,
        run_id=RUN_ID,
        infer=False,
        expiration_date=expired_on.isoformat(),
    )
    print(f"[ingest] expired   : {json.dumps(added_expired, default=str)[:200]}")

    added_future = memory.add(
        [{**EXPIRING_MESSAGE, "content": "I prefer window seats when flying."}],
        user_id=USER_ID,
        agent_id=AGENT_ID,
        run_id=RUN_ID,
        infer=False,
        expiration_date=expires_on.isoformat(),
    )
    print(f"[ingest] future    : {json.dumps(added_future, default=str)[:200]}")

    filters = {"user_id": USER_ID, "agent_id": AGENT_ID, "run_id": RUN_ID}

    # Both reads are recorded on purpose. The difference between them *is* the
    # finding: Mem0's default silently drops data the store still holds, which
    # for a migration is exactly backwards.
    raw_default = memory.get_all(filters=filters)
    raw = memory.get_all(filters=filters, show_expired=True)

    def results_of(payload: Any) -> list[dict]:
        return payload["results"] if isinstance(payload, dict) and "results" in payload else payload

    records = results_of(raw)
    records_default = results_of(raw_default)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(raw, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )

    all_fields: set[str] = set()
    for rec in records:
        all_fields.update(rec.keys())

    meta = {
        "generated_at": started.isoformat(),
        "generator": "scripts/dump_mem0_multiactor.py",
        "purpose": (
            "ground truth for agent_id, run_id, role, actor_id and expiration_date, "
            "none of which the Phase 0 single-user inferred dump can reach"
        ),
        "scope": {"user_id": USER_ID, "agent_id": AGENT_ID, "run_id": RUN_ID},
        "infer": False,
        "llm_invoked": False,
        "messages_ingested": len(CONVERSATION) + 2,
        "records_extracted": len(records),
        "records_visible_by_default": len(records_default),
        "records_hidden_by_default_expiry": len(records) - len(records_default),
        "expiration_dates": {
            "expired": expired_on.isoformat(),
            "future": expires_on.isoformat(),
            "relative_to": today.isoformat(),
        },
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "collection": COLLECTION,
        "sdk_versions": sdk_versions(),
        "record_fields": sorted(all_fields),
        "qdrant_payload_keys": payload_keys(memory),
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # ---- summary -----------------------------------------------------------
    print()
    print("=" * 72)
    print(f"records (show_expired=True)  : {len(records)}")
    print(f"records (mem0 default)       : {len(records_default)}"
          f"   <- {len(records) - len(records_default)} silently dropped")
    print("=" * 72)

    for i, rec in enumerate(records):
        print(f"\n--- record {i} ---")
        for key in sorted(rec):
            val = rec[key]
            shown = val if not isinstance(val, (dict, list)) else json.dumps(val, default=str)
            print(f"  {key:<18}: {shown}")

    print(f"\nunion of record fields : {sorted(all_fields)}")
    print(f"qdrant payload keys    : {meta['qdrant_payload_keys']}")
    print(f"\nwrote {FIXTURE_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {META_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
