"""Scoring a migration: what arrived, what changed, and what it cost.

This generalizes `scripts/roundtrip_mem0_to_cockroach.py`, which was the first
thing to ask the question and is still the baseline this module is checked
against. What it adds is the three things that script explicitly does not do:
partial credit, a loss ledger, and a report object rather than printed lines.

The alignment key is `MemoryRecord.fingerprint()`, for the reason CMM computes
it rather than storing it: the join has to survive everything a migration is
legitimately allowed to change. Not the id, not the timestamps, not the
vendor's own hash. That does mean alignment assumes content survives verbatim,
which is false the moment a target re-extracts facts with its own LLM -- see the
open question in CLAUDE.md. When that arrives it needs semantic alignment, and
this module should refuse rather than silently misalign.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from membridge.cmm import MemoryBundle, MemoryRecord
from membridge.fidelity.models import (
    FidelityReport,
    FieldSurvival,
    LossEntry,
    LossKind,
    Severity,
)

#: How many changed-record fingerprints a single field will list before the
#: report starts truncating.
MAX_LISTED = 8


def _float32(value: float) -> float:
    import struct

    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return max(-1.0, min(1.0, dot / (na * nb)))


class Comparison:
    """The verdict on one field of one record."""

    __slots__ = ("exact", "credit", "kind", "severity", "detail")

    def __init__(
        self,
        exact: bool,
        credit: float,
        *,
        kind: LossKind = LossKind.FIELD_CHANGED,
        severity: Severity = Severity.CRITICAL,
        detail: str = "",
    ) -> None:
        self.exact = exact
        self.credit = credit
        self.kind = kind
        self.severity = severity
        self.detail = detail


def compare_exact(source: Any, target: Any) -> Comparison:
    """The default, and the right default.

    Most CMM fields are things a migration must reproduce or admit it did not.
    There is no partial credit for a timestamp.
    """
    if source == target:
        return Comparison(True, 1.0)
    return Comparison(
        False, 0.0, detail=f"{source!r} -> {target!r}", severity=Severity.CRITICAL
    )


def compare_mapping(source: Any, target: Any) -> Comparison:
    """Key-level partial credit for `metadata` and `extensions`.

    These are the two fields where partial credit is meaningful rather than
    misleading: they are bags of independent facts, so losing one key is
    genuinely less bad than losing all of them. Credit is the fraction of source
    items reproduced exactly -- extra keys in the target do not earn credit, but
    they are reported, because a target inventing metadata is worth seeing.
    """
    if source == target:
        return Comparison(True, 1.0)
    if not isinstance(source, dict) or not isinstance(target, dict):
        return compare_exact(source, target)
    if not source:
        return Comparison(False, 0.0, detail=f"target added keys {sorted(target)}")

    kept = sum(1 for key, value in source.items() if key in target and target[key] == value)
    lost = sorted(set(source) - set(target))
    added = sorted(set(target) - set(source))
    parts = []
    if lost:
        parts.append(f"dropped {lost}")
    if added:
        parts.append(f"added {added}")
    changed = sorted(
        key for key in set(source) & set(target) if source[key] != target[key]
    )
    if changed:
        parts.append(f"changed {changed}")
    return Comparison(
        False,
        kept / len(source),
        detail="; ".join(parts) or "differs",
        severity=Severity.CRITICAL if kept == 0 else Severity.DEGRADED,
    )


def compare_vector(source: Any, target: Any) -> Comparison:
    """Bit-exactness first, then a named reason for anything less.

    The distinction this exists to draw: a vector that differs because the
    target's column is float32 has been *quantized*, predictably and boundedly.
    A vector that differs for any other reason has been *corrupted*. Both look
    identical to an equality check and they are not the same finding.

    Note this is not where CockroachDB's nine-digit text representation is
    handled -- that is `decode_vector`'s job, and by the time a bundle reaches
    this module the components are already recovered. If that ever regresses,
    this reports PRECISION_QUANTIZED on a lossless migration, which is why the
    detail line names the largest component delta rather than just asserting.
    """
    if source == target:
        return Comparison(True, 1.0)
    if source is None or target is None:
        return Comparison(
            False,
            0.0,
            detail="vector present on one side only",
            severity=Severity.CRITICAL,
        )
    if len(source) != len(target):
        return Comparison(
            False,
            0.0,
            detail=f"dimension {len(source)} -> {len(target)}",
            severity=Severity.CRITICAL,
        )

    similarity = _cosine(source, target)
    delta = max(abs(s - t) for s, t in zip(source, target))

    if [_float32(v) for v in source] == list(target):
        return Comparison(
            False,
            similarity if similarity is not None else 0.0,
            kind=LossKind.PRECISION_QUANTIZED,
            severity=Severity.DEGRADED,
            detail=(
                f"stored as float32; max component delta {delta:.3g}, "
                f"cosine {similarity:.9f}. Real loss, but bounded: the source "
                "vector was not float32-exact and the target's column is."
            ),
        )

    return Comparison(
        False,
        similarity if similarity is not None else 0.0,
        detail=f"max component delta {delta:.3g}, cosine {similarity:.9f}",
        severity=Severity.CRITICAL,
    )


#: Every field CMM carries, with how to read it and how to judge it. Exhaustive
#: on purpose: a field absent from this list is a field nobody is checking, and
#: the test suite asserts it covers `MemoryRecord`.
FieldSpec = tuple[str, Callable[[MemoryRecord], Any], Callable[[Any, Any], Comparison]]

DEFAULT_FIELDS: list[FieldSpec] = [
    ("content", lambda r: r.content, compare_exact),
    ("scope", lambda r: r.scope.as_key(), compare_exact),
    ("attribution", lambda r: r.attribution, compare_exact),
    ("created_at", lambda r: r.created_at, compare_exact),
    ("updated_at", lambda r: r.updated_at, compare_exact),
    ("expires_at", lambda r: r.expires_at, compare_exact),
    ("metadata", lambda r: r.metadata, compare_mapping),
    ("extensions", lambda r: r.extensions, compare_mapping),
    ("cmm_version", lambda r: r.cmm_version, compare_exact),
    ("id", lambda r: r.id, compare_exact),
    ("provenance", lambda r: r.provenance, compare_exact),
    ("embedding.model", lambda r: r.embedding.model if r.embedding else None, compare_exact),
    ("embedding.dim", lambda r: r.embedding.dim if r.embedding else None, compare_exact),
    (
        "embedding.normalized",
        lambda r: r.embedding.normalized if r.embedding else None,
        compare_exact,
    ),
    (
        "embedding.vector",
        lambda r: r.embedding.vector if r.embedding else None,
        compare_vector,
    ),
]


def score(
    source: MemoryBundle,
    target: MemoryBundle,
    *,
    target_system: str | None = None,
    fields: list[FieldSpec] | None = None,
) -> FidelityReport:
    """Score `target` as a migration of `source`.

    Records are aligned by content fingerprint. Anything in the source with no
    counterpart is a CRITICAL loss; anything in the target with no counterpart
    is reported too, because a target that was not empty when the migration ran
    makes every other number here untrustworthy.

    `target_system` has to be passed by the caller and cannot be read off the
    bundle, which is a consequence of a decision made in the Cockroach adapter
    rather than an oversight: a bundle read back out of CockroachDB still says
    `source_system="mem0"`, because provenance describes where a record came
    from, not the last system to hold it. Defaulting to `target.source_system`
    would therefore label every report "mem0 -> mem0".
    """
    fields = fields or DEFAULT_FIELDS

    src = source.fingerprints()
    dst = target.fingerprints()
    aligned = sorted(set(src) & set(dst))

    losses: list[LossEntry] = []
    for fingerprint in sorted(set(src) - set(dst)):
        losses.append(
            LossEntry(
                kind=LossKind.RECORD_MISSING,
                severity=Severity.CRITICAL,
                fingerprint=fingerprint,
                detail=f"source record not found in target: {src[fingerprint].content[:60]!r}",
            )
        )
    for fingerprint in sorted(set(dst) - set(src)):
        losses.append(
            LossEntry(
                kind=LossKind.RECORD_UNEXPECTED,
                severity=Severity.NOTE,
                fingerprint=fingerprint,
                detail=(
                    "target holds a record the source does not. The target was "
                    "not empty, so per-field rates below are over the aligned "
                    "subset only."
                ),
            )
        )

    # Vectors from different embedders are not comparable, and scoring them
    # would report embedding drift as migration loss -- the confusion CMM's
    # Embedding carries a model name to prevent. Skip, and say so.
    embedding_note = None
    try:
        source_space, target_space = source.embedding_space, target.embedding_space
    except ValueError as exc:  # a bundle that mixes embedders raises
        source_space = target_space = None
        embedding_note = f"embedding space undeterminable: {exc}"

    if embedding_note is None and source_space != target_space:
        embedding_note = (
            f"source is in {source_space} and target in {target_space}; vector "
            "comparison skipped because a distance between embedding spaces is "
            "a number with no meaning."
        )
        fields = [spec for spec in fields if spec[0] != "embedding.vector"]

    results: list[FieldSurvival] = []
    for name, get, compare in fields:
        survived = 0
        credit = 0.0
        changed: list[str] = []
        for fingerprint in aligned:
            verdict = compare(get(src[fingerprint]), get(dst[fingerprint]))
            credit += verdict.credit
            if verdict.exact:
                survived += 1
                continue
            changed.append(fingerprint)
            losses.append(
                LossEntry(
                    kind=verdict.kind,
                    severity=verdict.severity,
                    field=name,
                    fingerprint=fingerprint,
                    detail=verdict.detail,
                )
            )
        results.append(
            FieldSurvival(
                field=name,
                aligned=len(aligned),
                survived=survived,
                credit=credit,
                changed=tuple(changed[:MAX_LISTED]),
                changed_truncated=len(changed) > MAX_LISTED,
            )
        )

    return FidelityReport(
        source_system=source.source_system,
        target_system=target_system or f"{target.source_system} (unlabelled target)",
        source_records=len(source.records),
        target_records=len(target.records),
        aligned=len(aligned),
        fields=results,
        losses=losses,
        embedding_note=embedding_note,
    )
