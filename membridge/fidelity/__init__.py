"""Fidelity scoring: measuring what actually survived a migration.

The claim MemBridge is built to make falsifiable is "we migrated your memories".
This is the module that does the falsifying. Everything here follows from one
refusal, stated in `models.py` and enforced by the absence of an API for it:
there is no single confident percentage, because "97% migrated" hides which 3%,
and whether the missing part was the content or the timestamps matters enormously.
"""

from membridge.fidelity.models import (
    FidelityReport,
    FieldSurvival,
    LossEntry,
    LossKind,
    Severity,
)
from membridge.fidelity.render import render_text
from membridge.fidelity.score import (
    DEFAULT_FIELDS,
    Comparison,
    compare_exact,
    compare_mapping,
    compare_vector,
    score,
)

__all__ = [
    "Comparison",
    "DEFAULT_FIELDS",
    "FidelityReport",
    "FieldSurvival",
    "LossEntry",
    "LossKind",
    "Severity",
    "compare_exact",
    "compare_mapping",
    "compare_vector",
    "render_text",
    "score",
]
