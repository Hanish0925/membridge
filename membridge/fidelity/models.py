"""What a fidelity score is allowed to say.

The types here exist to make one class of statement impossible: a single
confident percentage. `scripts/roundtrip_mem0_to_cockroach.py` already refuses
that ("97% migrated" hides *which* 3%), and the module that replaces it must
refuse it structurally rather than by convention.

So a report is a list of per-field results plus an explicit ledger of losses,
and there is deliberately no `FidelityReport.score` returning one float. The
closest thing, `overall()`, returns the *worst* field rather than the mean,
because a migration that moved every timestamp and no content is not 93%
successful.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LossKind(str, Enum):
    """Why something did not survive.

    Naming the cause is the whole point of the ledger. A round trip can already
    report *that* a vector changed; without a cause, whoever reads it cannot
    tell a target that quantizes float64 to float32 from one that lost the
    vector entirely, and those call for completely different decisions.
    """

    #: The record is in the source and not in the target at all.
    RECORD_MISSING = "record_missing"
    #: The record is in the target and not in the source. Usually a target that
    #: was not empty when the migration started, which makes every other number
    #: in the report suspect.
    RECORD_UNEXPECTED = "record_unexpected"
    #: A field differs and nothing more specific is known.
    FIELD_CHANGED = "field_changed"
    #: The target's column is narrower than the value. Real loss, but bounded
    #: and predictable -- CockroachDB's float32 VECTOR is the motivating case.
    PRECISION_QUANTIZED = "precision_quantized"
    #: The source held something the target has no way to express, and the
    #: adapter dropped it knowingly. The case `extensions` cannot cover.
    UNREPRESENTABLE = "unrepresentable"


class Severity(str, Enum):
    """How much a loss should worry the person reading the report."""

    #: Data is gone and cannot be recovered from the target.
    CRITICAL = "critical"
    #: Data changed in a bounded, explainable way.
    DEGRADED = "degraded"
    #: Worth recording, not worth blocking on.
    NOTE = "note"


class LossEntry(BaseModel):
    """One thing that did not survive, and why.

    `detail` is free text and is meant to be read by a human deciding whether
    to accept the migration, so it should say what happened in concrete terms
    rather than restate the kind.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: LossKind
    severity: Severity
    field: str | None = None
    #: The content fingerprint of the affected record, or None for bundle-level
    #: losses. Fingerprint rather than id because the id is a thing a migration
    #: is allowed to change, and the ledger has to survive that.
    fingerprint: str | None = None
    detail: str


class FieldSurvival(BaseModel):
    """How one CMM field fared across every aligned record.

    `survived` counts exact matches; `credit` sums partial credit in [0,1]. They
    are kept separate rather than collapsed because "384 of 384 components are
    within 1e-9" and "the vector is the one that was written" are different
    claims, and only the second one means the migration was lossless.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    aligned: int
    survived: int
    credit: float
    #: Fingerprints of records where this field did not match exactly, capped so
    #: a wholesale failure produces a readable report rather than a wall of
    #: hashes. Truncation is flagged by `changed_truncated`.
    changed: tuple[str, ...] = ()
    changed_truncated: bool = False

    @property
    def intact(self) -> bool:
        return self.aligned == self.survived

    @property
    def rate(self) -> float:
        """Exact-match rate. 1.0 for a field with nothing to compare."""
        return 1.0 if self.aligned == 0 else self.survived / self.aligned

    @property
    def partial_rate(self) -> float:
        """Partial-credit rate, always >= `rate`."""
        return 1.0 if self.aligned == 0 else self.credit / self.aligned


class FidelityReport(BaseModel):
    """The result of scoring one migration.

    Deliberately not summarizable to a single number by anything but
    `overall()`, which reports the worst field rather than the average.
    """

    model_config = ConfigDict(extra="forbid")

    source_system: str
    target_system: str
    source_records: int
    target_records: int
    aligned: int
    fields: list[FieldSurvival] = Field(default_factory=list)
    losses: list[LossEntry] = Field(default_factory=list)
    #: Set when the two bundles are not in the same embedding space, in which
    #: case vector comparison is skipped rather than reported as total loss.
    embedding_note: str | None = None

    @property
    def missing(self) -> int:
        return sum(1 for loss in self.losses if loss.kind is LossKind.RECORD_MISSING)

    @property
    def unexpected(self) -> int:
        return sum(1 for loss in self.losses if loss.kind is LossKind.RECORD_UNEXPECTED)

    @property
    def intact(self) -> bool:
        """Every source record arrived and every field matched exactly."""
        return (
            self.missing == 0
            and self.aligned == self.source_records
            and all(field.intact for field in self.fields)
        )

    def overall(self) -> float:
        """The worst field's exact-match rate, not the mean.

        Averaging fields would let fourteen intact fields hide a destroyed one,
        which is precisely the summary this project exists to argue against.
        """
        if self.source_records == 0:
            return 1.0
        alignment = self.aligned / self.source_records
        return min([alignment, *(field.rate for field in self.fields)])

    def critical(self) -> list[LossEntry]:
        return [loss for loss in self.losses if loss.severity is Severity.CRITICAL]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
