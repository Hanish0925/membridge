"""The Common Memory Model — MemBridge's vendor-neutral intermediate.

Adapters read a source system into a `MemoryBundle` of `MemoryRecord`s and write
that bundle back out to a target. Everything MemBridge claims about fidelity is
a claim about what survives this round trip.
"""

from membridge.cmm.models import (
    CMM_SCHEMA_VERSION,
    UNIT_LENGTH_TOLERANCE,
    Actor,
    Embedding,
    MemoryBundle,
    MemoryRecord,
    Provenance,
    Scope,
    is_unit_length,
)

__all__ = [
    "CMM_SCHEMA_VERSION",
    "UNIT_LENGTH_TOLERANCE",
    "Actor",
    "Embedding",
    "MemoryBundle",
    "MemoryRecord",
    "Provenance",
    "Scope",
    "is_unit_length",
]
