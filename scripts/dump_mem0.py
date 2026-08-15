"""Produce the ground-truth Mem0 dump that the CMM schema is written against.

This script is the reason `tests/fixtures/mem0_raw_dump.json` exists. We read the
real shape of Mem0's output rather than trusting its documentation, because the
whole MemBridge claim rests on knowing exactly what Mem0 does and does not hold.

Two things are configured EXPLICITLY and must stay that way:

  * the embedder is `sentence-transformers/all-MiniLM-L6-v2` at 384 dims, run
    locally. Mem0's defaults reach for OpenAI; if the source and target embed with
    different models, any semantic fidelity number conflates migration loss with
    embedding drift.
  * the vector store is a local Qdrant at ./data/qdrant, so a dump is reproducible
    on a laptop with no cloud dependency.

The extraction LLM is a third moving part. Mem0 runs an LLM to decide which facts
to extract and how to consolidate them, so the *content* of the dump depends on
which model answers. It defaults to local Ollama here; override with
MEMBRIDGE_LLM_PROVIDER / MEMBRIDGE_LLM_MODEL to reproduce against a hosted model.

Usage:
    uv run python scripts/dump_mem0.py
    uv run python scripts/dump_mem0.py --keep   # do not wipe ./data/qdrant first
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# The Mem0 configuration now lives in the adapter, because the reader has to open
# the same collection with the same embedder this script writes it with. Two
# copies of that config would drift, and the failure mode is a reader that finds
# nothing or mislabels the vectors it finds.
from membridge.adapters.mem0 import (
    COLLECTION,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    build_config,
    widen_ollama_context,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = REPO_ROOT / "data" / "qdrant"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "mem0_raw_dump.json"
META_PATH = REPO_ROOT / "tests" / "fixtures" / "mem0_raw_dump.meta.json"

USER_ID = "john_001"

#: The four-turn conversation the whole Phase 0 demo runs on.
CONVERSATION: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Hi, my name is John, I am looking for some restaurant "
            "recommendations around my area?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Hi John, pleasure to meet you. Here are 5 restaurants near your "
            "current location: Rowdy Burgers, Sage & Salt, Kinara Grill, "
            "The Copper Pot, Noodle Theory."
        ),
    },
    {
        "role": "user",
        "content": (
            "Thanks for the recommendations, can you book the first "
            "recommendation on the list"
        ),
    },
    {
        "role": "assistant",
        "content": "What date and time would you like to book Rowdy Burgers for?",
    },
]

#: Recorded alongside every artifact. Vendor SDKs in this space churn fast; a dump
#: without versions is not reproducible.
TRACKED_PACKAGES = [
    "mem0ai",
    "sentence-transformers",
    "qdrant-client",
    "ollama",
    "openai",
    "pydantic",
    "torch",
    "transformers",
]


def sdk_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def describe_shape(value: Any, depth: int = 0) -> str:
    """Render the *shape* of a value, not its contents."""
    if isinstance(value, dict):
        if not value:
            return "dict(empty)"
        if depth >= 2:
            return f"dict({len(value)} keys)"
        inner = ", ".join(
            f"{k}: {describe_shape(v, depth + 1)}" for k, v in value.items()
        )
        return f"dict{{{inner}}}"
    if isinstance(value, list):
        if not value:
            return "list(empty)"
        return f"list[{len(value)}] of {describe_shape(value[0], depth + 1)}"
    if value is None:
        return "null"
    return type(value).__name__


def probe_embeddings(memory: Any, records: list[dict]) -> dict[str, Any]:
    """Answer: does get_all() hand back vectors, or must we go get them?

    This matters because CMM treats the embedding as a first-class field. If Mem0
    withholds it from its own read API, the adapter has to reach past Mem0 into the
    vector store — which is itself a finding worth recording.
    """
    in_get_all = any(
        any(k in r for k in ("embedding", "vector", "embeddings"))
        for r in records
    )

    result: dict[str, Any] = {
        "returned_by_get_all": in_get_all,
        "vector_store_probe": None,
    }

    try:
        client = memory.vector_store.client
        points, _ = client.scroll(
            collection_name=COLLECTION,
            limit=1,
            with_vectors=True,
            with_payload=True,
        )
        if points:
            vec = points[0].vector
            if isinstance(vec, dict):  # named vectors
                vec = next(iter(vec.values()))
            result["vector_store_probe"] = {
                "reachable": True,
                "dims": len(vec) if vec is not None else None,
                "payload_keys": sorted(points[0].payload.keys())
                if points[0].payload
                else [],
            }
        else:
            result["vector_store_probe"] = {"reachable": True, "dims": None}
    except Exception as exc:  # pragma: no cover - diagnostic path
        result["vector_store_probe"] = {"reachable": False, "error": repr(exc)}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="do not wipe ./data/qdrant before ingesting",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if not args.keep and QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
        print(f"[reset] removed {QDRANT_PATH}")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    from mem0 import Memory  # imported late so --help stays fast

    config = build_config(QDRANT_PATH)
    print(f"[config] embedder : {config['embedder']['config']['model']} "
          f"({EMBEDDING_DIM}d, local)")
    print(f"[config] llm      : {config['llm']['provider']}/"
          f"{config['llm']['config']['model']}")
    print(f"[config] store    : qdrant @ {QDRANT_PATH}")

    memory = Memory.from_config(config)

    if config["llm"]["provider"] == "ollama":
        num_ctx = int(os.environ.get("MEMBRIDGE_OLLAMA_NUM_CTX", "32768"))
        widen_ollama_context(memory, num_ctx)
        print(f"[config] num_ctx  : {num_ctx} (patched; mem0 does not set this)")

    started = datetime.now(timezone.utc)
    add_result = memory.add(CONVERSATION, user_id=USER_ID)
    print(f"[ingest] add() returned: {json.dumps(add_result, default=str)[:400]}")

    # mem0 2.x rejects top-level entity params here; scoping goes through filters.
    raw = memory.get_all(filters={"user_id": USER_ID})

    # Mem0 v2 wraps results in {"results": [...]}; older shapes return a bare list.
    records = raw["results"] if isinstance(raw, dict) and "results" in raw else raw

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(raw, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )

    embedding_info = probe_embeddings(memory, records)

    meta = {
        "generated_at": started.isoformat(),
        "generator": "scripts/dump_mem0.py",
        "user_id": USER_ID,
        "turns_ingested": len(CONVERSATION),
        "records_extracted": len(records),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "llm_provider": config["llm"]["provider"],
        "llm_model": config["llm"]["config"]["model"],
        "sdk_versions": sdk_versions(),
        "embeddings": embedding_info,
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # ---- summary -----------------------------------------------------------
    print()
    print("=" * 72)
    print(f"records extracted : {len(records)} (from {len(CONVERSATION)} turns)")
    print(f"top-level type    : {type(raw).__name__}"
          + (f" with keys {sorted(raw.keys())}" if isinstance(raw, dict) else ""))
    print("=" * 72)

    for i, rec in enumerate(records):
        print(f"\n--- record {i} ---")
        print(f"  field names : {sorted(rec.keys())}")
        for key, val in rec.items():
            print(f"  {key:<14} : {describe_shape(val)}")
            if isinstance(val, dict) and val:
                print(f"  {'':<14}   contents = {json.dumps(val, default=str)}")

    all_fields: set[str] = set()
    for rec in records:
        all_fields.update(rec.keys())
    print(f"\nunion of field names across records: {sorted(all_fields)}")

    print(f"\nembeddings returned by get_all() : {embedding_info['returned_by_get_all']}")
    print(f"vector store probe               : "
          f"{json.dumps(embedding_info['vector_store_probe'], default=str)}")

    print(f"\nwrote {FIXTURE_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {META_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
