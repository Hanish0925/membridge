"""Encoding CMM values into CockroachDB's wire types, and back.

Two of the columns in `sql/schema.sql` are types psycopg has no adapter for —
`VECTOR(384)` and the `cmm_actor` enum — so both directions are spelled out here
rather than left to whatever psycopg guesses. Keeping the pair in one module is
the point: an encoder and a decoder that disagree produce a round trip that looks
successful and is not, which is the precise failure mode MemBridge exists to
detect. If one changes, the other is in view.

There is deliberately no pgvector dependency. Cockroach's vector literal is
`[0.1,0.2,...]` — valid JSON, and a cast on the way in is all the driver needs.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Sequence

# Re-exported rather than reimplemented: both sides of a migration must agree on
# what "normalized" means, so the predicate lives in CMM. It is surfaced here
# because this is the adapter whose index correctness depends on it — see the
# vector index comment in sql/schema.sql.
from membridge.cmm import is_unit_length

__all__ = [
    "encode_vector",
    "decode_vector",
    "as_float32",
    "is_float32_exact",
    "is_unit_length",
]


def encode_vector(vector: Sequence[float]) -> str:
    """CMM vector -> CockroachDB literal, to be used with an explicit `::vector` cast.

    `repr` on a Python float is round-trip exact, so this loses nothing: the
    float parsed back out is bit-identical to the one that went in. Building the
    string by hand rather than via json.dumps keeps that guarantee explicit,
    since it is the whole reason a fidelity score over these vectors means
    anything.
    """
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


def as_float32(value: float) -> float:
    """Round a Python float to the nearest float32, as a float.

    CockroachDB's VECTOR stores float32 (it is pgvector-compatible), so this is
    what storage does to a component whether or not anyone asks for it.
    """
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def is_float32_exact(vector: Sequence[float]) -> bool:
    """Whether every component survives storage in a float32 column unchanged.

    False means writing this vector to CockroachDB *is* lossy — real loss, not
    the representational artifact `decode_vector` handles. Nothing currently
    reports it; see the loss-ledger open question in CLAUDE.md.
    """
    return all(as_float32(component) == float(component) for component in vector)


def decode_vector(value: Any) -> list[float] | None:
    """CockroachDB VECTOR -> list[float], recovering the stored float32 exactly.

    psycopg has no adapter registered for the VECTOR OID, so the value arrives as
    the raw text literal. It is JSON-shaped, so parsing is trivial — but parsing
    alone gives the WRONG NUMBER, in a way that is easy to miss and expensive to
    misread:

    CockroachDB prints each float32 component as the shortest decimal that
    round-trips *as a float32* — nine significant digits, e.g. `0.018742891`.
    Parsed as a Python float (float64) that string is a different value from the
    float32 it came from (`0.018742891028523445`), by around 1e-9 per component.

    The consequence is that a bit-exactness check on a round trip fails on every
    single component while nothing whatsoever has been lost. A fidelity module
    comparing vectors naively would report total embedding loss on a perfect
    migration — exactly the confusion between drift and loss that CMM's
    `Embedding` carries a model name to prevent.

    So: parse, then round to float32, which recovers the stored value bit for
    bit. Verified against CockroachDB v25.4.0 — a Mem0 bundle round-trips with
    all 384 components identical.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [as_float32(component) for component in value]
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"expected a vector literal, got {text[:40]!r}")
        return [as_float32(component) for component in parsed]
    raise TypeError(f"cannot decode a vector from {type(value).__name__}")
