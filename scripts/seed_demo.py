"""Seed the demo corpus into a CockroachDB target.

Two bundles, written for two different reasons.

**The demo bundle** is ~60 memories for one user, hand-written to be coherent
with the memories migrated out of Mem0 by `dump_mem0_multiactor.py` — same
`user_id`, same persona. That is the point: after a migration the agent should be
unable to tell which of its memories arrived from Mem0 and which were always
here, because a memory layer that treats migrated records as second-class has not
really migrated them. The topics are deliberately spread across food, travel,
work, health, family and money so that a top-5 retrieval for one topic is not
crowded out by near-duplicates of another.

**The corpus bundle** is bulk. It exists for the query planner, not for the demo:
CockroachDB only chooses `memory_record_embedding_idx` once statistics say the
table is worth an index scan, so a store holding five records answers every
search with a full scan and proves nothing about the vector index. It is written
under *other* users, so it inflates the table without polluting the demo user's
recall.

Everything is embedded with the same MiniLM/384 encoder the Mem0 records were,
because `CockroachReader.search` refuses a scope holding two embedding spaces —
correctly, since distances across embedders sort without meaning anything.

    uv run python scripts/seed_demo.py --replace

Costs a write to whatever `MEMBRIDGE_COCKROACH_DSN` points at. It never creates
the schema; run `membridge schema --apply` first.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from membridge.adapters.cockroach import (  # noqa: E402
    CockroachWriter,
    connect,
    schema_is_present,
)
from membridge.cmm import (  # noqa: E402
    Actor,
    Embedding,
    MemoryBundle,
    MemoryRecord,
    Provenance,
    Scope,
)
from membridge.embed import MODEL_DIM, MODEL_NAME, default_encoder  # noqa: E402

#: The user the Mem0 multi-actor thread was scoped to. Shared on purpose — see
#: the module docstring.
DEMO_USER = "john_001"

#: Tags the rows this script owns, so `--replace` can remove exactly them and
#: nothing else. A seed script that clears the whole table would eventually be
#: run against a target holding a real migration.
SEED_SOURCE = "membridge-demo"

#: One MiniLM forward pass per batch; 64 keeps peak memory small without making
#: the ONNX call overhead dominate.
BATCH = 64


# --- the demo user ---------------------------------------------------------
#
# (content, attribution, age in days, expires in N days or None)
#
# Expiries are relative so the fixture cannot quietly change meaning the day
# after it is written — the same reason `dump_mem0_multiactor.py` computes its
# dates rather than hard-coding them. Two are already lapsed at write time,
# which is what makes `include_expired` demonstrable rather than merely
# implemented.
CURATED: list[tuple[str, Actor, int, int | None]] = [
    # -- food, and the allergy the Mem0 thread established
    ("I keep kosher salt and good olive oil at home and cook most weeknights.", Actor.USER, 240, None),
    ("I cannot eat shrimp, crab, lobster or anything cooked in the same fryer.", Actor.USER, 200, None),
    ("Maria does not eat meat or fish, but she does eat eggs and dairy.", Actor.USER, 180, None),
    ("Our favourite restaurant for a night out is Casa Lupita on Bell Street.", Actor.USER, 150, None),
    ("I drink coffee black, and never after four in the afternoon.", Actor.USER, 300, None),
    ("I am trying to cut down on red meat to twice a week.", Actor.USER, 90, None),
    ("Maria is mildly lactose intolerant despite eating dairy; she takes an enzyme.", Actor.USER, 88, None),
    ("The bakery on Fifth closes at two on Sundays, so order the day before.", Actor.ASSISTANT, 60, None),
    ("I have a standing brunch with my sister on the first Sunday of the month.", Actor.USER, 130, None),
    ("I do not like coriander at all; it tastes like soap to me.", Actor.USER, 210, None),
    # -- travel, extending the window-seat preference
    ("I fly out of Newark rather than JFK because the drive is half as long.", Actor.USER, 170, None),
    ("My passport expires in March 2029 and my Global Entry in 2028.", Actor.USER, 160, None),
    ("I get motion sick on small propeller aircraft and avoid regional hops.", Actor.USER, 155, None),
    ("I pack carry-on only for anything under five nights.", Actor.USER, 140, None),
    ("Maria and I are planning two weeks in Portugal next spring.", Actor.USER, 45, None),
    ("I have a hotel booked in Lisbon for the second week of the Portugal trip.", Actor.USER, 40, 240),
    ("My airline status resets at the end of the calendar year.", Actor.ASSISTANT, 100, 137),
    ("I would rather take an overnight train than a 6am flight.", Actor.USER, 120, None),
    # -- work
    ("I work as a data engineer at Halloway Logistics, mostly on the pipeline team.", Actor.USER, 400, None),
    ("My manager is Priya Raghunathan and we have a 1:1 every Tuesday at 10.", Actor.USER, 380, None),
    ("I am on call for the ingestion service every third week.", Actor.USER, 200, None),
    ("Do not schedule anything before 9:30am; I do focused work first thing.", Actor.USER, 350, None),
    ("The quarterly planning doc is due to Priya by the end of this week.", Actor.USER, 5, -1),
    ("I gave a talk on schema migration at the internal engineering summit.", Actor.USER, 75, None),
    ("Halloway's expense policy needs receipts for anything over forty dollars.", Actor.ASSISTANT, 220, None),
    ("I am mentoring a junior engineer, Tomas, on Thursdays.", Actor.USER, 65, None),
    # -- health
    ("My dentist appointment is with Dr. Okafor; I go every six months.", Actor.USER, 110, 65),
    ("I run three times a week, usually along the river path.", Actor.USER, 250, None),
    ("My gym membership at Riverside Fitness renews annually.", Actor.USER, 190, 175),
    ("I take a vitamin D supplement through the winter months.", Actor.USER, 230, None),
    ("I sleep badly if I train hard after 8pm.", Actor.USER, 145, None),
    ("I had a physical in the spring and everything came back normal.", Actor.ASSISTANT, 120, None),
    # -- family and people
    ("Maria's birthday is on the 12th of November.", Actor.USER, 320, None),
    ("Maria is a landscape architect and works from home on Mondays.", Actor.USER, 310, None),
    ("My sister Elena lives in Chicago with her two children.", Actor.USER, 290, None),
    ("My parents' anniversary is in late September and I always call.", Actor.USER, 280, None),
    ("We have a cat called Biscuit who is on prescription food.", Actor.USER, 260, None),
    ("Biscuit's vet is the Riverbank practice; his next check-up is booked.", Actor.ASSISTANT, 30, 90),
    ("Elena is allergic to cats, so she stays at a hotel when she visits.", Actor.USER, 255, None),
    # -- money and admin
    ("I budget monthly in a spreadsheet and review it on the last Sunday.", Actor.USER, 200, None),
    ("My car insurance is with Meridian and renews in the autumn.", Actor.USER, 195, 70),
    ("I max out my retirement contribution early in the year rather than spreading it.", Actor.USER, 185, None),
    ("The apartment lease is up for renewal and I intend to renew it.", Actor.USER, 25, 120),
    ("I split shared expenses with Maria roughly 60/40 by income.", Actor.USER, 175, None),
    # -- preferences the agent should actually use
    ("Keep answers short unless I ask you to go into detail.", Actor.USER, 340, None),
    ("I would rather you say you do not know than guess.", Actor.USER, 335, None),
    ("Use metric for cooking and imperial for distance; that is just how I think.", Actor.USER, 330, None),
    ("Do not book anything on my behalf without asking first.", Actor.USER, 325, None),
    ("I read technical things in the morning and fiction at night.", Actor.USER, 315, None),
    # -- hobbies
    ("I play bass in a covers band that rehearses on alternate Wednesdays.", Actor.USER, 270, None),
    ("I am slowly restoring a 1978 racing bicycle in the garage.", Actor.USER, 265, None),
    ("I have been learning Portuguese for about eight months, mostly for the trip.", Actor.USER, 44, None),
    ("I keep a woodworking bench but have not used it since we moved.", Actor.USER, 240, None),
    ("I follow Formula One and watch qualifying rather than the race.", Actor.USER, 235, None),
    # -- lapsed on purpose
    ("I have tickets to the Wilkins Theatre production on the 3rd.", Actor.USER, 20, -3),
    ("The parking permit for the office garage is valid this quarter only.", Actor.ASSISTANT, 95, -8),
]

# --- the bulk corpus -------------------------------------------------------

CORPUS_TEMPLATES = [
    "I prefer {a} over {b} when I have the choice.",
    "Remind me that {a} matters more to me than {b}.",
    "I have never got on with {a}; {b} suits me better.",
    "My notes say I settled on {a} last time rather than {b}.",
    "Going forward, default to {a} and only fall back to {b} if asked.",
    "I switched from {b} to {a} and have not regretted it.",
    "For anything routine, {a}. For anything unusual, {b}.",
    "Colleagues keep suggesting {b}, but I still use {a}.",
]

CORPUS_SUBJECTS = [
    "morning meetings", "asynchronous updates", "written proposals", "whiteboard sessions",
    "aisle seats", "direct flights", "budget hotels", "serviced apartments",
    "spicy food", "slow-cooked stews", "street food", "tasting menus",
    "cycling to work", "the express bus", "driving in", "working from home",
    "paper notebooks", "digital task lists", "voice memos", "shared documents",
    "black tea", "filter coffee", "sparkling water", "herbal infusions",
    "early nights", "late reading", "weekend hiking", "indoor climbing",
    "detailed itineraries", "unplanned weekends", "group tours", "travelling alone",
    "monthly budgets", "annual reviews", "automatic transfers", "manual tracking",
    "long phone calls", "short messages", "video calls", "in-person catch-ups",
    "classical recordings", "live sets", "podcasts on walks", "silence while working",
    "one large screen", "two small screens", "standing desks", "a proper chair",
]

CORPUS_USERS = [f"user_{index:03d}" for index in range(1, 41)]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_record(
    content: str,
    *,
    user_id: str,
    attribution: Actor,
    created_at: datetime,
    expires_at: datetime | None,
    source_id: str,
    exported_at: datetime,
) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        scope=Scope(user_id=user_id, agent_id="membridge_demo_agent"),
        attribution=attribution,
        created_at=created_at,
        updated_at=created_at,
        expires_at=expires_at,
        provenance=Provenance(
            source_system=SEED_SOURCE,
            source_id=source_id,
            exported_at=exported_at,
            adapter="scripts/seed_demo.py",
        ),
    )


def build_demo_bundle(now: datetime) -> MemoryBundle:
    records = []
    for index, (content, actor, age_days, expires_in) in enumerate(CURATED):
        created = now - timedelta(days=age_days)
        expires = None if expires_in is None else now + timedelta(days=expires_in)
        records.append(
            make_record(
                content,
                user_id=DEMO_USER,
                attribution=actor,
                created_at=created,
                expires_at=expires,
                source_id=f"demo-{index:03d}",
                exported_at=now,
            )
        )
    return MemoryBundle(source_system=SEED_SOURCE, exported_at=now, records=records)


def build_corpus_bundle(now: datetime, count: int, seed: int) -> MemoryBundle:
    """`count` distinct memories spread over CORPUS_USERS.

    Distinctness is enforced here rather than left to chance: the schema's
    `record_content_unique_per_bundle` would reject a repeat, and CMM's
    `fingerprints()` would raise before that — both correctly, but at a point
    where the cause is much harder to see than it is from inside the generator.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    records = []
    attempts = 0
    limit = count * 50

    while len(records) < count and attempts < limit:
        attempts += 1
        first, second = rng.sample(CORPUS_SUBJECTS, 2)
        content = rng.choice(CORPUS_TEMPLATES).format(a=first, b=second)
        if content in seen:
            continue
        seen.add(content)
        records.append(
            make_record(
                content,
                user_id=rng.choice(CORPUS_USERS),
                attribution=Actor.USER,
                created_at=now - timedelta(days=rng.randint(1, 500)),
                expires_at=None,
                source_id=f"corpus-{len(records):05d}",
                exported_at=now,
            )
        )

    if len(records) < count:
        raise SystemExit(
            f"template space exhausted at {len(records)} of {count} distinct memories; "
            "add templates or subjects"
        )
    return MemoryBundle(source_system=SEED_SOURCE, exported_at=now, records=records)


def attach_embeddings(bundle: MemoryBundle) -> None:
    """Embed in place, in the space the migrated Mem0 records already occupy.

    `normalized=True` is not taken on trust anywhere it matters — the reader
    only converts L2 to cosine when the *stored* column says so — but `encode`
    normalizes as its last step and `tests/test_embed.py` checks that against
    `cmm.is_unit_length`. The writer re-measures on the way in regardless.
    """
    encoder = default_encoder()
    records = bundle.records
    for start in range(0, len(records), BATCH):
        chunk = records[start : start + BATCH]
        for record, vector in zip(chunk, encoder.encode([r.content for r in chunk])):
            record.embedding = Embedding(
                vector=vector, model=MODEL_NAME, dim=MODEL_DIM, normalized=True
            )
        print(f"  embedded {min(start + BATCH, len(records))}/{len(records)}", flush=True)


def delete_previous(conn) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM memory_record WHERE bundle_id IN "
            "(SELECT id FROM memory_bundle WHERE source_system = %s)",
            (SEED_SOURCE,),
        )
        records = cur.rowcount
        cur.execute("DELETE FROM memory_bundle WHERE source_system = %s", (SEED_SOURCE,))
        bundles = cur.rowcount
    conn.commit()
    return bundles, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None, help="else MEMBRIDGE_COCKROACH_DSN")
    parser.add_argument("--count", type=int, default=1000, help="bulk corpus size")
    parser.add_argument("--seed", type=int, default=20260816, help="RNG seed")
    parser.add_argument(
        "--replace",
        action="store_true",
        help=f"delete existing bundles whose source_system is {SEED_SOURCE!r} first",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(override=False)

    conn = connect(args.dsn) if args.dsn else connect()
    if not schema_is_present(conn):
        print("target has no CMM schema; run `membridge schema --apply` first", file=sys.stderr)
        return 2

    if args.replace:
        bundles, records = delete_previous(conn)
        print(f"[reset] removed {records} records in {bundles} bundles")

    now = utcnow()
    demo = build_demo_bundle(now)
    corpus = build_corpus_bundle(now, args.count, args.seed)

    writer = CockroachWriter(conn)
    for name, bundle in (("demo", demo), ("corpus", corpus)):
        print(f"[embed] {name}: {len(bundle.records)} records")
        attach_embeddings(bundle)
        bundle_id = writer.write_bundle(bundle)
        conn.commit()
        print(f"[write] {name}: bundle {bundle_id}")

    # Without fresh statistics the planner keeps costing the table as if it were
    # nearly empty and answers every scoped search with a full scan, which is
    # the exact thing the vector index is here to avoid. Cheap, and pointless to
    # leave to the automatic collector on a table that was just bulk-loaded.
    with conn.cursor() as cur:
        cur.execute("ANALYZE memory_record")
    conn.commit()
    print("[stats] ANALYZE memory_record")

    print(
        f"\nseeded {len(demo.records)} demo memories for {DEMO_USER} "
        f"and {len(corpus.records)} corpus memories"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
