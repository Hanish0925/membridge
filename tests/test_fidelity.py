"""Fidelity scoring.

No database needed anywhere in this file. Scoring compares two CMM bundles, and
making it depend on a live target would mean the thing that measures loss could
only be tested where loss was hard to arrange. Every interesting case here is a
bundle deliberately damaged in one specific way.

The baseline these are written against is
`scripts/roundtrip_mem0_to_cockroach.py`: it already reported the real Mem0 ->
CockroachDB result, so a scorer that disagrees with it on an intact migration is
wrong regardless of how reasonable its output looks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from membridge.cmm import (
    Actor,
    Embedding,
    MemoryBundle,
    MemoryRecord,
    Provenance,
    Scope,
)
from membridge.fidelity import (
    DEFAULT_FIELDS,
    LossKind,
    Severity,
    compare_mapping,
    compare_vector,
    render_text,
    score,
)

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)


def _record(**overrides: Any) -> MemoryRecord:
    base: dict[str, Any] = dict(
        content="John is allergic to shellfish.",
        scope=Scope(user_id="john_001"),
        attribution=Actor.USER,
        created_at=NOW,
        updated_at=NOW,
        provenance=Provenance(
            source_system="mem0", source_id="src-1", exported_at=NOW, adapter="tests"
        ),
    )
    base.update(overrides)
    return MemoryRecord(**base)


def _bundle(*records: MemoryRecord, system: str = "mem0") -> MemoryBundle:
    return MemoryBundle(source_system=system, exported_at=NOW, records=list(records))


def _other(content: str = "John prefers window seats.") -> MemoryRecord:
    return _record(
        content=content,
        provenance=Provenance(
            source_system="mem0", source_id="src-2", exported_at=NOW, adapter="tests"
        ),
    )


# --- the intact case ------------------------------------------------------


def test_a_perfect_migration_scores_intact() -> None:
    source = _bundle(_record(), _other())
    report = score(source, source.model_copy(deep=True), target_system="cockroachdb")

    assert report.intact
    assert report.overall() == 1.0
    assert report.aligned == 2
    assert report.losses == []
    assert all(field.intact for field in report.fields)


def test_the_report_names_the_target_because_the_bundle_cannot() -> None:
    """A bundle read out of CockroachDB still says source_system='mem0'.

    That is deliberate in the adapter -- provenance describes origin, not the
    last system to hold a record -- so the scorer must be told the target's name
    or every report would read "mem0 -> mem0".
    """
    source = _bundle(_record())
    report = score(source, source.model_copy(deep=True), target_system="cockroachdb")
    assert (report.source_system, report.target_system) == ("mem0", "cockroachdb")

    unlabelled = score(source, source.model_copy(deep=True))
    assert "unlabelled" in unlabelled.target_system


# --- records that did not arrive -----------------------------------------


def test_a_missing_record_is_a_critical_loss_naming_its_content() -> None:
    source = _bundle(_record(), _other())
    target = _bundle(_record())

    report = score(source, target, target_system="cockroachdb")

    assert not report.intact
    assert report.missing == 1
    (loss,) = [l for l in report.losses if l.kind is LossKind.RECORD_MISSING]
    assert loss.severity is Severity.CRITICAL
    assert "window seats" in loss.detail, "the ledger must say what was lost, not just count it"


def test_an_unexpected_record_is_reported_because_it_invalidates_the_rates() -> None:
    """Per-field rates are over the aligned subset, so a dirty target matters."""
    source = _bundle(_record())
    target = _bundle(_record(), _other())

    report = score(source, target, target_system="cockroachdb")

    assert report.unexpected == 1
    assert report.aligned == 1
    (loss,) = [l for l in report.losses if l.kind is LossKind.RECORD_UNEXPECTED]
    assert loss.severity is Severity.NOTE
    # Every source record did arrive, so this is not a loss of source data --
    # but it is not "intact" either, because the target holds more than it should.
    assert report.missing == 0


def test_overall_is_the_worst_field_not_the_average() -> None:
    """Fourteen intact fields must not be able to hide a destroyed one.

    `id` is excluded from `rates` here: `_record()`/`_other()` mint a fresh id
    per call, so its rate is 0.0 by construction and would otherwise be the
    "worst field" this test is trying to isolate -- see
    `test_a_fresh_id_never_drags_overall_down` for that behaviour directly.
    """
    source = _bundle(_record(), _other())
    target = _bundle(
        _record(created_at=NOW - timedelta(days=9)),
        _other(),
    )

    report = score(source, target, target_system="cockroachdb")

    rates = [field.rate for field in report.fields if not field.read_scoped]
    assert report.overall() == min(rates)
    assert report.overall() < sum(rates) / len(rates), "an average would flatter this"


# --- read-scoped fields -----------------------------------------------


def test_a_fresh_id_never_drags_overall_down() -> None:
    """Two independent reads of the same store mint two different ids.

    `_record()` defaults `id` via `uuid4()`, so `source` and `target` disagree
    on it by construction here even though nothing else differs -- this is
    exactly the shape of comparing a fresh Mem0 read against an older write,
    which is the Phase 8 near-miss this fix exists to close.
    """
    source = _bundle(_record())
    target = _bundle(_record())  # same content, different fingerprint-excluded id

    report = score(source, target, target_system="cockroachdb")

    id_field = next(f for f in report.fields if f.field == "id")
    assert id_field.read_scoped is True
    assert id_field.rate == 0.0, "the ids really did not match, and that is honest"
    assert id_field.partial_rate == 1.0, "but it earns full credit, not partial"
    assert report.overall() == 1.0
    assert report.intact is True

    loss = next(loss for loss in report.losses if loss.field == "id")
    assert loss.severity is Severity.NOTE
    assert loss not in report.critical()


def test_provenance_exported_at_alone_does_not_break_intact() -> None:
    """The Phase 8 near-miss, pinned directly.

    Scoring a fresh Mem0 read against an older write reported `provenance` 0/5
    on a migration that had lost nothing, because every subfield matched except
    `exported_at` -- which records when each read ran, not what the migration
    did. This is that exact scenario in miniature.
    """
    source = _record()
    target = source.model_copy(
        update={
            "provenance": source.provenance.model_copy(
                update={"exported_at": NOW + timedelta(hours=5)}
            )
        }
    )

    report = score(_bundle(source), _bundle(target), target_system="cockroachdb")

    provenance_field = next(f for f in report.fields if f.field == "provenance")
    assert provenance_field.intact is True
    assert report.intact is True


def test_a_genuine_provenance_mismatch_is_still_critical() -> None:
    """Excluding `exported_at` must not neuter the rest of provenance identity.

    A migration that scrambles `source_id` really has broken lineage, and that
    is not the read-pass artifact this fix is narrowing around.
    """
    source = _record()
    target = source.model_copy(
        update={
            "provenance": source.provenance.model_copy(update={"source_id": "wrong-id"})
        }
    )

    report = score(_bundle(source), _bundle(target), target_system="cockroachdb")

    provenance_field = next(f for f in report.fields if f.field == "provenance")
    assert provenance_field.intact is False
    assert report.intact is False
    loss = next(loss for loss in report.losses if loss.field == "provenance")
    assert loss.severity is Severity.CRITICAL


# --- partial credit -------------------------------------------------------


def test_metadata_earns_partial_credit_per_key() -> None:
    verdict = compare_mapping({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})
    assert not verdict.exact
    assert verdict.credit == pytest.approx(2 / 3)
    assert "dropped ['c']" in verdict.detail
    assert verdict.severity is Severity.DEGRADED


def test_losing_every_key_is_critical_not_merely_degraded() -> None:
    verdict = compare_mapping({"a": 1}, {})
    assert verdict.credit == 0.0
    assert verdict.severity is Severity.CRITICAL


def test_a_target_that_invents_metadata_is_reported() -> None:
    verdict = compare_mapping({"a": 1}, {"a": 1, "spurious": True})
    assert not verdict.exact
    assert verdict.credit == 1.0, "nothing was lost"
    assert "added ['spurious']" in verdict.detail


# --- the float32 case, which is the whole point ---------------------------


def test_float32_quantization_is_named_as_such_not_reported_as_corruption() -> None:
    """The loss-ledger case CLAUDE.md called for.

    Writing a vector that is not float32-exact to CockroachDB quantizes it.
    That is real loss -- but bounded and explainable, and an equality check
    cannot tell it apart from a vector that was corrupted outright.
    """
    import struct

    def f32(v: float) -> float:
        return struct.unpack("f", struct.pack("f", v))[0]

    source = [0.1234567890123456, 0.9876543210987654, 0.5]
    stored = [f32(v) for v in source]
    assert stored != source, "these must genuinely differ, or the test proves nothing"

    verdict = compare_vector(source, stored)

    assert not verdict.exact
    assert verdict.kind is LossKind.PRECISION_QUANTIZED
    assert verdict.severity is Severity.DEGRADED
    assert verdict.credit > 0.9999999, "quantization is tiny; credit must reflect that"
    assert "float32" in verdict.detail


def test_a_corrupted_vector_is_critical_even_when_it_is_close() -> None:
    """Closeness is not the test. A vector nothing explains is corruption."""
    source = [0.1, 0.2, 0.3]
    mangled = [0.1, 0.2, 0.30001]

    verdict = compare_vector(source, mangled)
    assert verdict.kind is LossKind.FIELD_CHANGED
    assert verdict.severity is Severity.CRITICAL


def test_a_vector_present_on_one_side_only_is_total_loss() -> None:
    verdict = compare_vector([0.1, 0.2], None)
    assert verdict.credit == 0.0
    assert verdict.severity is Severity.CRITICAL


# --- embedding spaces -----------------------------------------------------


def test_vectors_from_different_embedders_are_not_scored_at_all() -> None:
    """Reporting drift as loss is the confusion Embedding.model exists to stop."""
    source = _bundle(
        _record(embedding=Embedding(vector=[1.0, 0.0], model="miniLM", dim=2))
    )
    target = _bundle(
        _record(embedding=Embedding(vector=[1.0, 0.0], model="titan", dim=2))
    )

    report = score(source, target, target_system="cockroachdb")

    assert report.embedding_note is not None
    assert "no meaning" in report.embedding_note
    assert not any(field.field == "embedding.vector" for field in report.fields), (
        "a vector comparison across spaces would produce a number that sorts "
        "and means nothing"
    )
    # The model field itself is still compared, and still shows the difference.
    model_field = next(f for f in report.fields if f.field == "embedding.model")
    assert model_field.survived == 0


# --- the accounting test --------------------------------------------------


def test_every_cmm_field_is_scored() -> None:
    """The counterpart to test_cmm_schema's no-field-unaccounted-for test.

    A field CMM carries but nobody scores is a field that can be destroyed by a
    migration without the report noticing -- which is exactly the unfalsifiable
    claim this project exists to avoid. Add a field to MemoryRecord and this
    fails until DEFAULT_FIELDS covers it.
    """
    scored = {name.split(".")[0] for name, _, _ in DEFAULT_FIELDS}
    declared = set(MemoryRecord.model_fields)

    assert declared - scored == set(), f"unscored CMM fields: {sorted(declared - scored)}"
    assert scored - declared == set(), f"scoring a field CMM does not have: {scored - declared}"


# --- the rendered report --------------------------------------------------


def test_the_text_report_shows_intact_fields_too() -> None:
    """A report that lists only problems cannot prove anything survived."""
    source = _bundle(_record(), _other())
    text = render_text(score(source, source.model_copy(deep=True), target_system="crdb"))

    assert "MIGRATION INTACT" in text
    assert "content" in text and "provenance" in text
    assert "2/2" in text


def test_the_text_report_leads_with_the_worst_news() -> None:
    source = _bundle(_record(metadata={"a": 1, "b": 2}))
    target = _bundle(_record(metadata={"a": 1}, content="John is allergic to shellfish."))
    text = render_text(score(source, target, target_system="crdb"))

    assert "LOSS LEDGER" in text
    assert "MIGRATION LOSSY" in text
    assert "metadata" in text
