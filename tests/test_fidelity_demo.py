"""The demo's corruptions, checked against the real scorer.

No database, no live Mem0 -- every test here builds a small bundle by hand and
runs it through `membridge.fidelity.score`, the same function the web demo
calls. The thing worth pinning is not "the corruption changed a value" (that's
trivial) but "the real scorer reports the specific loss kind and severity the
demo claims it will show" -- if these drift apart, the page would be narrating
a detection that isn't the one that actually happened.
"""

from __future__ import annotations

from datetime import datetime, timezone

from membridge.cmm import Actor, Embedding, MemoryBundle, MemoryRecord, Provenance, Scope
from membridge.fidelity import LossKind, Severity, score
from membridge.fidelity.demo import corrupt_vector, drop_expiry, simulate_quantization_loss

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)

#: A real MiniLM component profile has plenty of digits; a short hand-written
#: vector is enough to exercise the comparators without pulling in a model.
VECTOR = [0.12, -0.34, 0.56, -0.78, 0.9]


def _record(**overrides: object) -> MemoryRecord:
    base: dict[str, object] = dict(
        content="John is allergic to shellfish.",
        scope=Scope(user_id="john_001"),
        attribution=Actor.USER,
        created_at=NOW,
        updated_at=NOW,
        expires_at=None,
        embedding=Embedding(vector=list(VECTOR), model="test-embedder", dim=len(VECTOR)),
        provenance=Provenance(
            source_system="mem0", source_id="src-1", exported_at=NOW, adapter="tests"
        ),
    )
    base.update(overrides)
    return MemoryRecord(**base)


def _bundle(*records: MemoryRecord) -> MemoryBundle:
    return MemoryBundle(source_system="mem0", exported_at=NOW, records=list(records))


def test_drop_expiry_is_caught_as_a_real_field_change() -> None:
    bundle = _bundle(_record())
    corrupted = drop_expiry(bundle)

    report = score(bundle, corrupted, target_system="cockroachdb")

    assert report.intact is False
    field = next(f for f in report.fields if f.field == "expires_at")
    assert field.survived == 0
    loss = next(loss for loss in report.losses if loss.field == "expires_at")
    assert loss.severity is Severity.CRITICAL


def test_drop_expiry_flips_rather_than_always_nulling() -> None:
    """A record that already has no expiry must still visibly change."""
    bundle = _bundle(_record(expires_at=None))
    corrupted = drop_expiry(bundle)

    assert corrupted.records[0].expires_at is not None
    assert corrupted.records[0].expires_at != bundle.records[0].expires_at


def test_corrupt_vector_is_caught_as_critical_with_a_real_cosine() -> None:
    bundle = _bundle(_record())
    corrupted = corrupt_vector(bundle)

    report = score(bundle, corrupted, target_system="cockroachdb")

    assert report.intact is False
    field = next(f for f in report.fields if f.field == "embedding.vector")
    assert field.survived == 0
    loss = next(loss for loss in report.losses if loss.field == "embedding.vector")
    assert loss.severity is Severity.CRITICAL
    assert loss.kind is LossKind.FIELD_CHANGED
    assert "cosine -1.000000000" in loss.detail, "negation must be the exact opposite"


def test_corrupt_vector_does_not_touch_dimension_or_norm() -> None:
    """Negation must not trip the bundle-level mixed-embedding-space guard."""
    bundle = _bundle(_record())
    corrupted = corrupt_vector(bundle)

    original = bundle.records[0].embedding
    negated = corrupted.records[0].embedding
    assert negated.dim == original.dim
    assert negated.vector == [-c for c in original.vector]
    # Must not raise: a truncated dim would make this bundle internally
    # inconsistent and score() would refuse to compare any vector at all.
    assert corrupted.embedding_space == bundle.embedding_space


def test_simulated_quantization_is_degraded_not_critical() -> None:
    bundle = _bundle(_record())
    source, target = simulate_quantization_loss(bundle)

    report = score(source, target, target_system="cockroachdb")

    field = next(f for f in report.fields if f.field == "embedding.vector")
    assert field.survived == 0, "the perturbed source and quantized target must differ"
    assert field.credit > 0.99, "quantization is bounded loss, not total loss"
    loss = next(loss for loss in report.losses if loss.field == "embedding.vector")
    assert loss.kind is LossKind.PRECISION_QUANTIZED
    assert loss.severity is Severity.DEGRADED
    assert loss not in report.critical()


def test_simulated_quantization_leaves_every_other_field_intact() -> None:
    """Isolating the one open question this demonstrates, nothing else."""
    bundle = _bundle(_record())
    source, target = simulate_quantization_loss(bundle)

    report = score(source, target, target_system="cockroachdb")

    other_fields = [f for f in report.fields if f.field != "embedding.vector"]
    assert all(f.intact for f in other_fields)
