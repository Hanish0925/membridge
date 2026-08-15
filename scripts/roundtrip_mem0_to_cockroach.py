"""Mem0 -> CMM -> CockroachDB -> CMM, reporting what survived.

The first end-to-end exercise of both adapters against real instances of both
systems. It is not the fidelity module — that will need alignment strategies,
partial credit and a report format — but it is the thing that fidelity scoring
will be checked against, and it already answers the question MemBridge exists to
ask: of everything the source held, how much of it is in the target?

Field survival is reported per field rather than as a single percentage on
purpose. "97% migrated" is the kind of number that hides which 3%, and whether
the missing part was the content or the timestamps matters enormously.

Requires:
  * the multi-actor Mem0 store (scripts/dump_mem0_multiactor.py)
  * a CockroachDB with sql/schema.sql applied, or reachable so this can apply it

Usage:
    uv run python scripts/roundtrip_mem0_to_cockroach.py
    MEMBRIDGE_COCKROACH_DSN=... uv run python scripts/roundtrip_mem0_to_cockroach.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from membridge.adapters.cockroach import (
    CockroachReader,
    CockroachWriter,
    connect,
    ensure_schema,
    schema_is_present,
)
from membridge.adapters.mem0 import Mem0Reader, build_config
from membridge.cmm import MemoryBundle, MemoryRecord
from membridge.fidelity import render_text, score

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The multi-actor store, because it is the one that exercises every scope field,
#: both attribution paths' output, and expiry. A Phase 0 round trip would look
#: perfect while leaving most of the schema untested.
QDRANT_PATH = REPO_ROOT / "data" / "qdrant_multiactor"
COLLECTION = "membridge_mem0_multiactor"
SCOPE = {"user_id": "john_001", "agent_id": "support_agent_v2", "run_id": "thread_8891"}

#: Every field CMM can carry, so the report is exhaustive rather than a
#: spot-check. Anything absent from this list is a field nobody is checking.
FIELDS: list[tuple[str, Callable[[MemoryRecord], Any]]] = [
    ("content", lambda r: r.content),
    ("scope", lambda r: r.scope.as_key()),
    ("attribution", lambda r: r.attribution),
    ("created_at", lambda r: r.created_at),
    ("updated_at", lambda r: r.updated_at),
    ("expires_at", lambda r: r.expires_at),
    ("metadata", lambda r: r.metadata),
    ("extensions", lambda r: r.extensions),
    ("cmm_version", lambda r: r.cmm_version),
    ("id", lambda r: r.id),
    ("provenance", lambda r: r.provenance),
    ("embedding.model", lambda r: r.embedding.model if r.embedding else None),
    ("embedding.dim", lambda r: r.embedding.dim if r.embedding else None),
    ("embedding.normalized", lambda r: r.embedding.normalized if r.embedding else None),
    ("embedding.vector", lambda r: r.embedding.vector if r.embedding else None),
]


def read_source() -> MemoryBundle:
    from mem0 import Memory

    config = build_config(QDRANT_PATH, collection=COLLECTION)
    reader = Mem0Reader(Memory.from_config(config))
    return reader.read_bundle(**SCOPE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the written bundle in the target instead of deleting it",
    )
    args = parser.parse_args()

    source = read_source()
    print(f"[mem0]   {len(source.records)} records, space={source.embedding_space}")

    conn = connect()
    if not schema_is_present(conn):
        print("[crdb]   schema absent; applying sql/schema.sql")
        ensure_schema(conn)

    bundle_id = CockroachWriter(conn).write_bundle(source)
    print(f"[crdb]   wrote bundle {bundle_id}")

    target = CockroachReader(conn).read_bundle(bundle_id)
    print(f"[crdb]   read back {len(target.records)} records, space={target.embedding_space}")

    # Alignment is by content fingerprint, which is the join key precisely
    # because it survives everything a migration is allowed to change.
    src = source.fingerprints()
    dst = target.fingerprints()
    aligned = sorted(set(src) & set(dst))

    print()
    print("=" * 56)
    print(f"records aligned      : {len(aligned)}/{len(src)}")
    print(f"missing in target    : {len(set(src) - set(dst))}")
    print(f"unexpected in target : {len(set(dst) - set(src))}")
    print("=" * 56)
    print(f"{'field':<24}{'survived':>10}")
    print("-" * 34)

    intact = True
    for name, get in FIELDS:
        survived = sum(1 for fp in aligned if get(src[fp]) == get(dst[fp]))
        flag = "" if survived == len(aligned) else "   <-- LOSS"
        if survived != len(aligned):
            intact = False
        print(f"{name:<24}{survived:>4}/{len(aligned)}{flag}")

    print()
    # Vectors are compared for bit-exactness, not closeness. An "almost equal"
    # vector would hide the float32 text-representation trap that decode_vector
    # exists to handle -- see membridge/adapters/cockroach/types.py.
    print("vectors are compared bit-exactly, not approximately")
    print("ROUND TRIP INTACT" if intact else "ROUND TRIP LOSSY")

    # -- and now the same question, asked by the module ---------------------
    #
    # The comparison above is kept rather than replaced, for the same reason
    # tests/test_cmm_schema.py writes the mem0 mapping twice: it is an
    # independent witness. If membridge.fidelity is ever refactored into
    # agreeing with itself, this loop still disagrees.
    report = score(source, target, target_system="cockroachdb")
    print()
    print(render_text(report))

    if report.intact != intact:
        raise SystemExit(
            f"DISAGREEMENT: this script says {'intact' if intact else 'lossy'}, "
            f"membridge.fidelity says {'intact' if report.intact else 'lossy'}. "
            "One of them is wrong and the scorer is the one nobody has checked."
        )
    print()
    print(f"[check]  membridge.fidelity agrees: {'intact' if report.intact else 'lossy'}")

    if not args.keep:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_bundle WHERE id = %s", (str(bundle_id),))
        conn.commit()
        print(f"[crdb]   removed bundle {bundle_id} (pass --keep to retain)")


if __name__ == "__main__":
    main()
