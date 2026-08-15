"""Mem0 adapter: reads a Mem0 store into CMM.

`Mem0Reader` talks to a live instance; `bundle_from_dump` maps an already
captured `get_all()` payload, which is how the recorded fixture exercises the
same mapping offline.
"""

from membridge.adapters.mem0.config import (
    COLLECTION,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    build_config,
    widen_ollama_context,
)
from membridge.adapters.mem0.reader import (
    ADAPTER_NAME,
    SOURCE_SYSTEM,
    Mem0ReadError,
    Mem0Reader,
    bundle_from_dump,
    record_to_cmm,
)

__all__ = [
    "ADAPTER_NAME",
    "COLLECTION",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "Mem0ReadError",
    "Mem0Reader",
    "SOURCE_SYSTEM",
    "build_config",
    "bundle_from_dump",
    "record_to_cmm",
    "widen_ollama_context",
]
