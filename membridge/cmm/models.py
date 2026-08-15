"""The Common Memory Model: MemBridge's vendor-neutral memory record.

CMM is the intermediate every adapter reads into and writes out of. Its whole
job is to be *lossless*: if a source system holds a piece of information and CMM
cannot carry it, the migration silently destroys it and the fidelity score is a
lie about a smaller thing than it claims.

So this schema is written against a real dump, not against documentation. Every
field here exists because `tests/fixtures/mem0_raw_dump.json` (Mem0 2.0.15) or
its Qdrant payload actually contains it. See `scripts/dump_mem0.py` for how that
ground truth is produced, and `tests/test_cmm_schema.py` for the accounting test
that fails the moment a source field has nowhere to go.

Three design rules follow from "lossless":

  1. Anything with a genuine cross-system meaning is a typed field. Speaker
     attribution, scope, timestamps and the embedding are all real concepts in
     every memory store worth bridging, so they get names and types.
  2. Anything vendor-specific goes in `extensions`, namespaced by source system.
     It is preserved verbatim and travels with the record. Nothing is dropped on
     the floor just because CMM has no opinion about it.
  3. Nothing derivable is stored. A stored hash drifts from the content it
     describes; `MemoryRecord.fingerprint()` recomputes instead. The one hash we
     *do* store is the source's own (`Provenance.source_hash`), because that is
     evidence about the source, not a fact about our content.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Bumped whenever a change would make an older bundle read incorrectly rather
#: than merely incompletely. Bundles and records both carry it so a file found on
#: its own is still interpretable.
CMM_SCHEMA_VERSION = "0.1.0"

_WHITESPACE = re.compile(r"\s+")

#: Tolerance for calling a vector unit-length. Embedders return float32 and
#: adapters move vectors through JSON and SQL, so exact equality to 1.0 is the
#: wrong test even for a vector that genuinely is normalized.
UNIT_LENGTH_TOLERANCE = 1e-6


def is_unit_length(
    vector: Sequence[float], *, tolerance: float = UNIT_LENGTH_TOLERANCE
) -> bool:
    """Whether a vector is unit-length.

    Lives in CMM rather than in an adapter because both sides of a migration need
    the same answer, and it is a fact about an embedding rather than about any
    one store.

    It is not decoration. A target may accelerate only *some* distance metrics —
    CockroachDB indexes L2 and not cosine — and L2 ranks identically to cosine
    only for unit-length vectors. So this predicate decides whether a similarity
    query is answered correctly or merely quickly. `Embedding.normalized` records
    it; `None` there means no adapter checked, which must never be read as False.
    """
    magnitude = sum(float(component) * float(component) for component in vector) ** 0.5
    return abs(magnitude - 1.0) <= tolerance


def _require_utc(value: datetime) -> datetime:
    """Normalize to UTC, refusing naive datetimes.

    A naive timestamp is not a timestamp, it is a timestamp plus a guess. Two
    systems guessing differently is exactly the kind of silent corruption a
    fidelity score is supposed to catch, so we reject it at the boundary.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            "timestamp must be timezone-aware; a naive datetime cannot be "
            "compared across systems without inventing an offset"
        )
    return value.astimezone(timezone.utc)


class Actor(str, Enum):
    """Who a memory is attributed to.

    Mem0 2.x records this as `attributed_to` and emits only `user`/`assistant`.
    The wider set is here because the concept generalizes and a target store that
    supports tool or system turns should not have to invent an encoding.

    `UNKNOWN` is deliberate rather than defaulting to `USER`: a source that does
    not track attribution must produce a record that *says so*, otherwise the
    absence becomes an unfalsifiable assertion after one migration hop.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"


class Embedding(BaseModel):
    """A vector together with the space it lives in.

    The model name is not decoration. Cosine similarity between vectors from
    different embedders is a meaningless number that still looks like a score, so
    a bare `list[float]` would let the fidelity module report embedding drift as
    migration loss. Carrying the space with the vector makes that comparison
    refusable instead of merely wrong.

    Mem0 does not return vectors from `get_all()` at all — they have to be read
    out of Qdrant directly — which is why this is optional on the record and why
    a bundle can legitimately be embedding-free.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    vector: list[float]
    model: str = Field(description="embedder identity, e.g. 'sentence-transformers/all-MiniLM-L6-v2'")
    dim: int = Field(gt=0)
    normalized: bool | None = Field(
        default=None,
        description="whether the vector is unit-length; None means the adapter did not check",
    )

    @model_validator(mode="after")
    def _check_dim(self) -> Embedding:
        if len(self.vector) != self.dim:
            raise ValueError(
                f"declared dim {self.dim} but vector has {len(self.vector)} components"
            )
        return self

    @property
    def space(self) -> tuple[str, int]:
        """The comparability key. Two embeddings are comparable iff these match."""
        return (self.model, self.dim)


class Scope(BaseModel):
    """Whose memory this is.

    Mem0 partitions on `user_id` / `agent_id` / `run_id`; most stores have some
    version of the same triple. These are kept as separate typed fields rather
    than an opaque dict because a relational target needs them as indexed columns.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = Field(
        default=None, description="a single run or conversation; Mem0 calls this run_id"
    )
    app_id: str | None = None

    @model_validator(mode="after")
    def _require_one(self) -> Scope:
        if not any((self.user_id, self.agent_id, self.session_id, self.app_id)):
            raise ValueError(
                "scope must name at least one of user_id, agent_id, session_id, app_id; "
                "an unscoped memory cannot be retrieved or migrated safely"
            )
        return self

    def as_key(self) -> tuple[str | None, str | None, str | None, str | None]:
        return (self.user_id, self.agent_id, self.session_id, self.app_id)


class Provenance(BaseModel):
    """Where this record came from, in enough detail to go back and check.

    Vendor SDKs in this space churn fast enough that "it came from Mem0" is not a
    useful claim without a version attached — the same script against 2.0.15 and
    2.1 can produce different field sets.
    """

    model_config = ConfigDict(extra="forbid")

    source_system: str = Field(description="e.g. 'mem0', 'cockroach'")
    source_id: str = Field(description="the record's native identifier in the source")
    source_version: str | None = Field(
        default=None, description="SDK or server version, e.g. 'mem0ai 2.0.15'"
    )
    source_hash: str | None = Field(
        default=None,
        description="the source's own content hash, kept as evidence rather than trusted",
    )
    exported_at: datetime
    adapter: str | None = Field(
        default=None, description="the MemBridge adapter that produced this record"
    )

    _norm_exported_at = field_validator("exported_at")(_require_utc)


class MemoryRecord(BaseModel):
    """One atomic remembered fact, in vendor-neutral form."""

    model_config = ConfigDict(extra="forbid")

    cmm_version: str = CMM_SCHEMA_VERSION

    #: CMM's own identifier. Distinct from `provenance.source_id` on purpose: the
    #: source id is only unique within the source, and a bundle may one day merge
    #: two systems that both hand out UUIDs.
    id: UUID = Field(default_factory=uuid4)

    content: str = Field(description="the remembered fact, as natural language")
    scope: Scope
    attribution: Actor = Actor.UNKNOWN
    created_at: datetime
    updated_at: datetime

    #: When this memory stops being valid, if the source tracks expiry at all.
    #:
    #: Typed rather than left to `extensions` because a memory that should have
    #: expired is worse than a memory that is missing: it is confidently wrong.
    #: A target store that cannot express expiry has to decide what to do about
    #: this field, and that decision should be forced rather than skipped.
    #:
    #: Note that sources differ in granularity — Mem0 records a bare *date* —
    #: so adapters keep the source-exact form in `extensions` as well.
    expires_at: datetime | None = None

    #: Caller-supplied metadata. Mem0 returns `null` here on every record in the
    #: ground-truth dump; that is normalized to `{}` so downstream code never has
    #: to distinguish "no metadata" from "metadata absent".
    metadata: dict[str, Any] = Field(default_factory=dict)

    embedding: Embedding | None = None

    #: Vendor fields with no CMM home, namespaced by source system:
    #: `{"mem0": {"text_lemmatized": "..."}}`. This is the loss-prevention escape
    #: hatch — the accounting test in tests/ asserts every source field lands
    #: either in a typed field or in here.
    extensions: dict[str, dict[str, Any]] = Field(default_factory=dict)

    provenance: Provenance

    @field_validator("metadata", mode="before")
    @classmethod
    def _null_metadata_is_empty(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("content")
    @classmethod
    def _content_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return value

    _norm_created_at = field_validator("created_at")(_require_utc)
    _norm_updated_at = field_validator("updated_at")(_require_utc)
    _norm_expires_at = field_validator("expires_at")(
        lambda v: None if v is None else _require_utc(v)
    )

    @model_validator(mode="after")
    def _check_ordering(self) -> MemoryRecord:
        if self.updated_at < self.created_at:
            raise ValueError(
                f"updated_at {self.updated_at.isoformat()} precedes "
                f"created_at {self.created_at.isoformat()}"
            )
        return self

    @model_validator(mode="after")
    def _check_extension_namespaces(self) -> MemoryRecord:
        for namespace in self.extensions:
            if not namespace or not namespace.strip():
                raise ValueError("extension namespaces must be non-empty source-system names")
        return self

    def fingerprint(self) -> str:
        """A content-identity hash, stable across systems.

        This is the join key the fidelity module uses to line a source record up
        with its migrated counterpart, so it must not depend on anything a
        migration is allowed to change: not the id, not the timestamps, not the
        vendor's own hash function. Unicode is NFC-normalized and runs of
        whitespace collapsed, because a store that round-trips text through JSON
        or a SQL column may legitimately renormalize either one.
        """
        normalized = unicodedata.normalize("NFC", self.content).strip()
        normalized = _WHITESPACE.sub(" ", normalized)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Whether this memory's validity has lapsed. Never expired if unset."""
        if self.expires_at is None:
            return False
        return self.expires_at < (now or datetime.now(timezone.utc))


class MemoryBundle(BaseModel):
    """A set of records exported from one system in one pass.

    The unit of migration and the unit of fidelity scoring. Invariants that only
    make sense across a whole export live here rather than on the record.
    """

    model_config = ConfigDict(extra="forbid")

    cmm_version: str = CMM_SCHEMA_VERSION
    source_system: str
    exported_at: datetime
    records: list[MemoryRecord] = Field(default_factory=list)

    _norm_exported_at = field_validator("exported_at")(_require_utc)

    @model_validator(mode="after")
    def _check_versions(self) -> MemoryBundle:
        mismatched = {r.cmm_version for r in self.records} - {self.cmm_version}
        if mismatched:
            raise ValueError(
                f"bundle is cmm_version {self.cmm_version} but contains records at "
                f"{sorted(mismatched)}; mixed-version bundles cannot be compared"
            )
        return self

    @model_validator(mode="after")
    def _check_unique_ids(self) -> MemoryBundle:
        seen: set[UUID] = set()
        for record in self.records:
            if record.id in seen:
                raise ValueError(f"duplicate record id {record.id} in bundle")
            seen.add(record.id)
        return self

    @property
    def embedding_space(self) -> tuple[str, int] | None:
        """The one embedding space in this bundle, or None if it carries no vectors.

        Raises when records disagree. A bundle spanning two embedders cannot be
        scored for semantic fidelity as a unit, and finding that out at scoring
        time — after the numbers look plausible — is too late.
        """
        spaces = {r.embedding.space for r in self.records if r.embedding is not None}
        if not spaces:
            return None
        if len(spaces) > 1:
            raise ValueError(
                f"bundle mixes embedding spaces {sorted(spaces)}; vectors from "
                "different embedders are not comparable"
            )
        return next(iter(spaces))

    def fingerprints(self) -> dict[str, MemoryRecord]:
        """Records keyed by content fingerprint, for source/target alignment.

        Collisions mean the source holds duplicate content, which is itself a
        finding — it is reported rather than silently deduplicated.
        """
        out: dict[str, MemoryRecord] = {}
        for record in self.records:
            key = record.fingerprint()
            if key in out:
                raise ValueError(
                    f"duplicate content fingerprint {key[:12]}… shared by records "
                    f"{out[key].id} and {record.id}"
                )
            out[key] = record
        return out
