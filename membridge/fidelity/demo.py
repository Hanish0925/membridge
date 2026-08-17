"""Named, deliberate damage, for demonstrating that fidelity scoring works.

Every static ledger looks the same: a clean grid of `5/5`. That is what a
correct migration produces, but it is indistinguishable from a scorer that
would say `5/5` regardless -- a skeptical reader has no way to tell the
detector apart from a rubber stamp until they watch it catch something. These
functions exist so the web demo can show that on demand, against real records,
using the real `score()` this module already ships.

Two of the three corruptions below are applied to a real bundle -- the one
this project actually migrated out of Mem0 -- so the demo scores real data
with one thing changed, not a synthetic example built to look convincing.
`simulate_quantization_loss` is the deliberate exception: real Mem0 vectors
are already float32-exact (measured, Phase 4 of CLAUDE.md), so no live record
exhibits this loss, and forcing one to look quantized would misreport what
production actually holds. It builds a hypothetical pair instead and its name
says so.
"""

from __future__ import annotations

from datetime import timedelta

from membridge.adapters.cockroach import as_float32, is_float32_exact
from membridge.cmm import MemoryBundle, MemoryRecord

#: A small, deliberately irrational-looking offset. Its only job is to push a
#: component off whatever value it already holds by more than float32's ~7
#: significant digits can absorb, so `is_float32_exact` is reliably False on
#: the perturbed source regardless of which component of which record it lands
#: on. Not derived from anything -- there is no "right" delta, only one large
#: enough to survive rounding and small enough not to change what the vector
#: means.
_QUANTIZATION_DELTA = 3.14159e-9


def _replace_record(bundle: MemoryBundle, index: int, record: MemoryRecord) -> MemoryBundle:
    records = list(bundle.records)
    records[index] = record
    # `model_copy` deliberately does not re-run MemoryBundle's validators (see
    # `embedding_space`, `fingerprints`) -- the corrupted copy is meant to
    # reach `score()` exactly as damaged, not get cleaned up on the way there.
    return bundle.model_copy(update={"records": records})


def drop_expiry(bundle: MemoryBundle, *, index: int = 0) -> MemoryBundle:
    """Flip one record's `expires_at`, so the ledger catches a real field vanishing.

    Flips rather than always nulling: a record that already has no expiry
    would otherwise show no change at all, which is a demo that silently does
    nothing depending on which record it lands on.
    """
    record = bundle.records[index]
    flipped = (
        None
        if record.expires_at is not None
        else record.created_at + timedelta(days=3)
    )
    return _replace_record(bundle, index, record.model_copy(update={"expires_at": flipped}))


def corrupt_vector(bundle: MemoryBundle, *, index: int = 0) -> MemoryBundle:
    """Negate one record's embedding, so the ledger catches a vector going bad.

    Negation, specifically, rather than truncation or zeroing:

    * Truncating a dimension changes `Embedding.dim`, which makes the
      corrupted bundle internally inconsistent (`MemoryBundle.embedding_space`
      requires one space across all records) -- `score()` would then refuse to
      compare *any* vector in the bundle as "spaces differ" rather than catch
      this one record as damaged. Correct behaviour, wrong demo.
    * Zeroing collapses the vector's norm to 0, which makes cosine similarity
      undefined (`0/0`) rather than merely bad, and `compare_vector`'s CRITICAL
      detail string formats that value assuming it is a float.

    Negation keeps the dimension and the norm exactly as they were and still
    produces the most damaged similarity a comparable vector can have: cosine
    -1.0, the exact opposite of the original.
    """
    record = bundle.records[index]
    if record.embedding is None:
        raise ValueError(f"record {index} has no embedding to corrupt")
    negated = record.embedding.model_copy(
        update={"vector": [-component for component in record.embedding.vector]}
    )
    return _replace_record(bundle, index, record.model_copy(update={"embedding": negated}))


def simulate_quantization_loss(
    bundle: MemoryBundle, *, index: int = 0
) -> tuple[MemoryBundle, MemoryBundle]:
    """A hypothetical (source, target) pair showing what float32 write-loss looks like.

    Every real vector in this project's live migration is float32-exact, which
    is a genuine, verified finding -- not a gap. The gap `is_float32_exact`
    exists to close is different: nothing surfaces the check's result, so a
    source that genuinely were not float32-exact would quantize silently on
    write. This builds that case: `source` gets one component nudged past
    float32 precision; `target` is exactly what CockroachDB's column would
    store for it. Scoring the pair shows `PRECISION_QUANTIZED`/DEGRADED rather
    than CRITICAL, which is the whole point of that distinction existing.
    """
    record = bundle.records[index]
    if record.embedding is None:
        raise ValueError(f"record {index} has no embedding to perturb")

    vector = list(record.embedding.vector)
    vector[0] = vector[0] + _QUANTIZATION_DELTA
    assert not is_float32_exact(vector), (
        "the perturbation must actually leave float32 precision, or this "
        "demonstrates nothing"
    )

    perturbed = record.embedding.model_copy(update={"vector": vector})
    quantized = record.embedding.model_copy(
        update={"vector": [as_float32(c) for c in vector]}
    )

    source_bundle = _replace_record(
        bundle, index, record.model_copy(update={"embedding": perturbed})
    )
    target_bundle = _replace_record(
        bundle, index, record.model_copy(update={"embedding": quantized})
    )
    return source_bundle, target_bundle


#: Every named corruption the demo can apply, and what each one is supposed to
#: prove. Single source of truth for both the Lambda route and its tests, so
#: adding a mode here is what makes it reachable -- there is nowhere else to
#: forget to wire it in.
MODES = ("intact", "drop_expiry", "corrupt_vector", "simulate_quantization_loss")

__all__ = [
    "MODES",
    "corrupt_vector",
    "drop_expiry",
    "simulate_quantization_loss",
]
