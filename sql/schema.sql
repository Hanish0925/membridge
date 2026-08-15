-- CMM 0.1.0 mapped onto CockroachDB.
--
-- This is the target side of the first real migration pair (Mem0 -> CockroachDB).
-- It is written against membridge/cmm/models.py field by field, and its job is to
-- be the place where CMM's invariants stop being Python validators and become
-- things the database itself refuses to store. A target that accepts a record CMM
-- would have rejected is a target that can quietly end a migration in a state the
-- fidelity score has no way to describe.
--
-- Every CHECK and UNIQUE below corresponds to a named invariant in the CMM
-- module. Where that correspondence is not obvious, it is spelled out.
--
-- Version requirements:
--   * VECTOR type          -- v24.2+ (preview)
--   * VECTOR INDEX/CSPANN  -- v25.2+
-- Everything else here is standard CockroachDB SQL. The vector index is isolated
-- at the bottom of the file so the schema still applies cleanly on v24.2-v25.1,
-- just without ANN acceleration.
--
-- Applied and verified against a real CockroachDB v25.4.0 single node, not
-- written from the docs alone: every constraint below was checked to actually
-- reject what it claims to reject (`sql/verify_constraints.sql`, 14 cases), and
-- the vector index was confirmed to be used via EXPLAIN. Re-applying this file
-- is a no-op. Findings from that exercise are in CLAUDE.md.
--
-- Apply with:
--   cockroach sql --insecure --database=membridge --file sql/schema.sql

-- ---------------------------------------------------------------------------
-- Attribution
-- ---------------------------------------------------------------------------

-- Mirrors cmm.Actor exactly. An enum rather than a STRING because the whole
-- point of Actor.UNKNOWN is that "we don't know who said this" must be a value
-- the schema can hold, not a NULL that later reads as "nobody bothered to set
-- it". A free STRING column would let a future adapter invent a sixth actor and
-- have it survive silently to the next hop.
CREATE TYPE IF NOT EXISTS cmm_actor AS ENUM (
    'user',
    'assistant',
    'system',
    'tool',
    'unknown'
);

-- ---------------------------------------------------------------------------
-- Bundles
-- ---------------------------------------------------------------------------

-- One export pass; the unit of migration and of fidelity scoring.
--
-- The embedding space lives HERE rather than only on the record, because
-- MemoryBundle.embedding_space raises when records disagree. Hoisting the space
-- to the bundle makes that invariant structural: a bundle cannot mix embedders
-- because there is only one row to say which embedder it used. Records point
-- back at it through a composite foreign key (see memory_record).
CREATE TABLE IF NOT EXISTS memory_bundle (
    id                UUID        NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    cmm_version       STRING      NOT NULL,
    source_system     STRING      NOT NULL,

    -- When the SOURCE produced this export. Distinct from imported_at on
    -- purpose: a bundle read in January and written here in March describes
    -- January's memories, and conflating the two would misdate every record.
    exported_at       TIMESTAMPTZ NOT NULL,
    imported_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- NULL for a bundle with no vectors at all, which is a legitimate state:
    -- Mem0's get_all() returns none, so an unenriched Mem0 dump lands here
    -- embedding-free rather than being rejected.
    embedding_model   STRING,
    embedding_dim     INT,

    CONSTRAINT bundle_source_system_nonempty
        CHECK (btrim(source_system) != ''),

    -- The space is (model, dim) together -- cmm.Embedding.space. Half a space is
    -- not a weaker claim, it is an incoherent one.
    CONSTRAINT bundle_embedding_space_is_whole
        CHECK ((embedding_model IS NULL) = (embedding_dim IS NULL)),

    -- The physical vector column below is VECTOR(384). Rejecting a mismatched
    -- declared dim here turns what would be an obscure insert-time cast error on
    -- the record into a clear statement about the bundle.
    CONSTRAINT bundle_embedding_dim_matches_column
        CHECK (embedding_dim IS NULL OR embedding_dim = 384),

    -- Referenced by memory_record's composite FK. A plain PK on id is not enough
    -- to point a multi-column FK at.
    CONSTRAINT bundle_space_unique UNIQUE (id, embedding_model, embedding_dim)
);

-- ---------------------------------------------------------------------------
-- Records
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_record (
    -- CMM's own id, supplied by the writer. The default exists only so the table
    -- is usable by hand; MemBridge always provides it, because CMM's id and the
    -- source's id are deliberately different things.
    --
    -- A UUID primary key is also the right choice for CockroachDB specifically:
    -- ranges are ordered by key, so a sequential PK funnels every insert into one
    -- range and one leaseholder. Random UUIDs spread writes across the cluster.
    id                  UUID        NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,

    bundle_id           UUID        NOT NULL
        REFERENCES memory_bundle (id) ON DELETE CASCADE,

    cmm_version         STRING      NOT NULL,

    content             STRING      NOT NULL,

    -- MemoryRecord.fingerprint(), the content-identity join key.
    --
    -- CMM's third design rule says nothing derivable is stored, and this is a
    -- deliberate, documented exception: it is the key the fidelity module aligns
    -- source and target on, and an alignment key that cannot be indexed forces
    -- every comparison to pull the whole table into memory -- which is exactly
    -- what the streaming open question says will not always be possible.
    --
    -- It is NOT a computed column, and could not be: the fingerprint is defined
    -- over NFC-normalized, whitespace-collapsed text, and SQL has no NFC. A
    -- computed column would therefore compute a DIFFERENT hash than Python and
    -- the two would disagree silently -- worse than not having the column.
    -- Consequence: the writer populates it and the reader must recompute and
    -- verify rather than trust it.
    content_fingerprint STRING      NOT NULL,

    -- cmm.Scope, flattened. Separate columns rather than JSONB because these are
    -- the partition keys every read filters on, and CMM keeps them typed for
    -- precisely this reason.
    scope_user_id       STRING,
    scope_agent_id      STRING,
    scope_session_id    STRING,   -- Mem0 calls this run_id
    scope_app_id        STRING,

    attribution         cmm_actor   NOT NULL DEFAULT 'unknown',

    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL,

    -- NULL means "no expiry", never "unknown". A source that cannot express
    -- expiry produces NULL here and says so in extensions.
    expires_at          TIMESTAMPTZ,

    -- '{}' not NULL, matching CMM's normalization of Mem0's null metadata, so
    -- readers never have to distinguish "empty" from "absent".
    metadata            JSONB       NOT NULL DEFAULT '{}',

    -- The loss-prevention escape hatch, namespaced by source system:
    -- {"mem0": {"text_lemmatized": "...", "actor_id": "maria"}}. JSONB because
    -- its shape is by definition unknown to CMM -- the moment it had a schema it
    -- would belong in a typed column instead.
    extensions          JSONB       NOT NULL DEFAULT '{}',

    -- cmm.Embedding. The vector, plus the space it lives in, plus whether it is
    -- unit-length -- see the vector index at the bottom for why that last column
    -- is load-bearing rather than trivia.
    embedding           VECTOR(384),
    embedding_model     STRING,
    embedding_dim       INT,
    embedding_normalized BOOL,

    -- cmm.Provenance, flattened. Kept on the record rather than in its own table
    -- because it is 1:1 with the record and describes it; a join would buy
    -- nothing and lose the ability to read a record's origin in one scan.
    prov_source_system  STRING      NOT NULL,
    prov_source_id      STRING      NOT NULL,
    prov_source_version STRING,
    prov_source_hash    STRING,
    prov_exported_at    TIMESTAMPTZ NOT NULL,
    prov_adapter        STRING,

    -- -- CMM invariants, enforced ------------------------------------------

    CONSTRAINT record_content_nonempty
        CHECK (btrim(content) != ''),

    -- Scope._require_one: an unscoped memory cannot be retrieved or migrated
    -- safely, so it must not be storable either.
    CONSTRAINT record_scope_names_someone
        CHECK (scope_user_id IS NOT NULL
            OR scope_agent_id IS NOT NULL
            OR scope_session_id IS NOT NULL
            OR scope_app_id IS NOT NULL),

    -- MemoryRecord._check_ordering.
    CONSTRAINT record_updated_not_before_created
        CHECK (updated_at >= created_at),

    -- sha256 hex. Cheap to check and it catches a writer that stores the
    -- source's own hash (Mem0's is a 32-char md5) in the fingerprint column --
    -- a substitution that would otherwise silently break every alignment.
    CONSTRAINT record_fingerprint_is_sha256
        CHECK (content_fingerprint ~ '^[0-9a-f]{64}$'),

    -- Embedding._check_dim, as far as SQL can express it: the declared space
    -- must be present exactly when the vector is.
    CONSTRAINT record_embedding_space_is_whole
        CHECK (num_nonnulls(embedding, embedding_model, embedding_dim) IN (0, 3)),

    CONSTRAINT record_embedding_dim_matches_column
        CHECK (embedding_dim IS NULL OR embedding_dim = 384),

    -- MemoryBundle.embedding_space: records may not disagree with their bundle
    -- about which embedder produced them. MATCH SIMPLE is what makes this work
    -- -- the FK is only checked when all three columns are non-NULL, so a
    -- record with no vector is exempt, while a record WITH one must match the
    -- bundle's declared space or fail.
    CONSTRAINT record_embedding_space_matches_bundle
        FOREIGN KEY (bundle_id, embedding_model, embedding_dim)
        REFERENCES memory_bundle (id, embedding_model, embedding_dim),

    -- MemoryBundle.fingerprints(): duplicate content within one export is
    -- reported, never silently merged. Here that means the second insert fails
    -- loudly rather than the pair quietly collapsing into one row and the
    -- fidelity score reporting a record as "lost".
    CONSTRAINT record_content_unique_per_bundle
        UNIQUE (bundle_id, content_fingerprint),

    -- A source id is unique within its source, and a bundle comes from one
    -- source pass. Re-importing the same export twice is a new bundle, so this
    -- does not block re-migration.
    CONSTRAINT record_source_id_unique_per_bundle
        UNIQUE (bundle_id, prov_source_system, prov_source_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- The scope triple, in the order Mem0 filters on it. STORING content_fingerprint
-- so the commonest fidelity query -- "which fingerprints exist in this scope" --
-- is answered from the index without touching the primary rows.
CREATE INDEX IF NOT EXISTS memory_record_scope_idx
    ON memory_record (scope_user_id, scope_agent_id, scope_session_id)
    STORING (content_fingerprint);

-- Alignment lookups run source-fingerprint -> target-record. Without this the
-- join degrades to a full scan of the target.
CREATE INDEX IF NOT EXISTS memory_record_fingerprint_idx
    ON memory_record (content_fingerprint);

-- Going back to the source to check a specific record.
CREATE INDEX IF NOT EXISTS memory_record_provenance_idx
    ON memory_record (prov_source_system, prov_source_id);

CREATE INDEX IF NOT EXISTS memory_record_bundle_idx
    ON memory_record (bundle_id);

-- Partial: most records never expire, and indexing their NULLs would be pure
-- overhead on the common path.
CREATE INDEX IF NOT EXISTS memory_record_expiry_idx
    ON memory_record (expires_at)
    WHERE expires_at IS NOT NULL;

-- Extensions are queried by namespace ("what did mem0 give us that CMM didn't
-- model"), which is a containment query and therefore wants GIN.
CREATE INDEX IF NOT EXISTS memory_record_extensions_idx
    ON memory_record USING GIN (extensions);

CREATE INDEX IF NOT EXISTS memory_record_metadata_idx
    ON memory_record USING GIN (metadata);

-- ---------------------------------------------------------------------------
-- Vector index -- requires CockroachDB v25.2+
-- ---------------------------------------------------------------------------
--
-- Kept last and separable: on v24.2-v25.1 the VECTOR column and its operators
-- work, only this acceleration is unavailable, and the rest of the schema should
-- still apply.
--
-- Two things about this index are easy to get wrong and expensive to discover
-- later:
--
-- 1. CockroachDB accelerates ONLY L2 distance (`<->`). The cosine operator
--    (`<=>`) is supported but falls back to a full scan. MemBridge's fidelity
--    module wants cosine, so on the face of it this index is useless to it.
--
--    It is not, but only because of a precondition: for unit-length vectors,
--    ||a-b||^2 = 2 - 2*cos(a,b), so L2 and cosine induce the SAME ranking. The
--    Mem0 store's vectors were measured, not assumed -- all-MiniLM-L6-v2 through
--    Mem0's HuggingFace embedder returns vectors of norm 1.0 -- which is what
--    makes an L2 index a valid accelerator for a cosine question here.
--
--    That is a property of the embedder, not of vectors in general. This is why
--    embedding_normalized is a column: a bundle written by some future adapter
--    with unnormalized vectors must be visibly excluded from L2-accelerated
--    search rather than silently ranked wrong. NULL means "the adapter did not
--    check", which is not the same as false and must not be read as true.
--
-- 2. The index is prefix-partitioned on scope_user_id. CockroachDB only uses a
--    vector index when the query's filters match the index's prefix columns, and
--    every MemBridge read is scoped -- there is no unscoped read to serve.
--
-- Both points were confirmed by EXPLAIN on v25.4.0 rather than taken on faith:
--
--    scoped   ... ORDER BY embedding <-> $1  ->  "• vector search"   (accelerated)
--    unscoped ... ORDER BY embedding <-> $1  ->  "• scan, FULL SCAN" (prefix needed)
--    scoped   ... ORDER BY embedding <=> $1  ->  "• scan"            (cosine: no index)
--
CREATE VECTOR INDEX IF NOT EXISTS memory_record_embedding_idx
    ON memory_record (scope_user_id, embedding);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Opt-in, and deliberately NOT the default read path.
--
-- Mem0's mistake was making this the default: show_expired=False silently drops
-- records the store still holds, and a migration built on it loses data without
-- reporting it. The fix is not to hide expiry, it is to make hiding it something
-- a caller has to ask for by name.
CREATE OR REPLACE VIEW memory_record_unexpired AS
    SELECT *
    FROM memory_record
    WHERE expires_at IS NULL OR expires_at >= now();
