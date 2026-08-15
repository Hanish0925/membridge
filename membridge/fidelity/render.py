"""Turning a `FidelityReport` into something a person can act on.

The layout is inherited from `scripts/roundtrip_mem0_to_cockroach.py`, whose
output turned out to be the right shape: alignment counts first, then one line
per field, then the verdict. What is added is the loss ledger, which is the part
that says *why* -- and which is the reason this is worth having as a module
rather than a print statement.

Fields that are intact are still printed. A report that only lists problems
cannot be used to prove something survived, and "prove what survived" is the
entire product.
"""

from __future__ import annotations

from membridge.fidelity.models import FidelityReport, Severity

_RULE = "=" * 64


def render_text(report: FidelityReport, *, show_intact: bool = True) -> str:
    lines: list[str] = []
    add = lines.append

    add(_RULE)
    add(f"FIDELITY  {report.source_system} -> {report.target_system}")
    add(_RULE)
    add(f"source records       : {report.source_records}")
    add(f"target records       : {report.target_records}")
    add(f"aligned (fingerprint): {report.aligned}")
    add(f"missing in target    : {report.missing}")
    add(f"unexpected in target : {report.unexpected}")

    if report.embedding_note:
        add("")
        add(f"embeddings           : {report.embedding_note}")

    add("")
    add(f"{'field':<24}{'exact':>12}{'credit':>10}")
    add("-" * 46)
    for field in report.fields:
        if field.intact and not show_intact:
            continue
        marker = "" if field.intact else "  <-- LOSS"
        # Partial credit is shown only where it differs from the exact rate.
        # Printing "1.000" next to "5/5" on every intact row trains the reader
        # to ignore the column that matters on the rows where it does not.
        credit = "" if field.intact else f"{field.partial_rate:.3f}"
        add(f"{field.field:<24}{field.survived:>5}/{field.aligned:<6}{credit:>10}{marker}")

    if report.losses:
        add("")
        add("LOSS LEDGER")
        add("-" * 46)
        # Critical first: the ledger is read top-down by someone deciding
        # whether to accept the migration.
        order = {Severity.CRITICAL: 0, Severity.DEGRADED: 1, Severity.NOTE: 2}
        for loss in sorted(report.losses, key=lambda entry: order[entry.severity]):
            where = f" [{loss.field}]" if loss.field else ""
            who = f" {loss.fingerprint[:12]}…" if loss.fingerprint else ""
            add(f"{loss.severity.value.upper():<9}{loss.kind.value}{where}{who}")
            add(f"          {loss.detail}")

    add("")
    add("vectors are compared bit-exactly, not approximately")
    if report.intact:
        add("MIGRATION INTACT")
    else:
        worst = report.overall()
        add(f"MIGRATION LOSSY — worst field {worst:.3f} ({len(report.critical())} critical)")
    return "\n".join(lines)
