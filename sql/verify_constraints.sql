-- Does schema.sql actually enforce what its comments claim?
--
-- Every CHECK, UNIQUE and FOREIGN KEY in schema.sql exists to make one of CMM's
-- Python invariants true at the storage layer. A constraint that is present but
-- does not bite is worse than an absent one, because the comment above it then
-- reads as a guarantee. This file is how that stops being a matter of trust.
--
-- Each case states the outcome it expects. Cases marked "must fail" should emit
-- an ERROR; cases marked OK should insert exactly one row.
--
-- Run against a SCRATCH database -- it writes rows and deletes them again:
--
--   cockroach sql --insecure --database=membridge \
--       --set errexit=false --file sql/verify_constraints.sql
--
-- `--set errexit=false` is required: without it the client stops at case 1 and
-- every later case silently goes unchecked.
--
-- Verified against CockroachDB v25.4.0 (see CLAUDE.md); all 13 cases behave.

-- Clean slate, so a previous run cannot make a later one pass or fail spuriously.
DELETE FROM memory_record WHERE bundle_id IN (
    '11111111-1111-1111-1111-111111111111',
    '22222222-2222-2222-2222-222222222222');
DELETE FROM memory_bundle WHERE id IN (
    '11111111-1111-1111-1111-111111111111',
    '22222222-2222-2222-2222-222222222222');

-- Bundle A declares an embedding space; bundle B declares none. Both are legal:
-- a Mem0 get_all() dump carries no vectors at all.
INSERT INTO memory_bundle (id, cmm_version, source_system, exported_at, embedding_model, embedding_dim)
VALUES ('11111111-1111-1111-1111-111111111111', '0.1.0', 'mem0', now(),
        'sentence-transformers/all-MiniLM-L6-v2', 384);

INSERT INTO memory_bundle (id, cmm_version, source_system, exported_at)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', 'mem0', now());

SELECT '=== 0. OK: a plain valid record ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, attribution, created_at, updated_at,
    prov_source_system, prov_source_id, prov_exported_at)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', 'John is allergic to shellfish.',
        repeat('a', 64), 'john_001', 'user', now(), now(), 'mem0', 'src-1', now());

-- Scope._require_one
SELECT '=== 1. must fail: record naming no scope at all ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', 'x', repeat('b', 64),
        now(), now(), 'mem0', 'src-2', now());

-- MemoryRecord._check_ordering
SELECT '=== 2. must fail: updated_at precedes created_at ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', 'x', repeat('c', 64),
        'u', '2026-01-02T00:00:00Z', '2026-01-01T00:00:00Z', 'mem0', 'src-3', now());

-- MemoryRecord._content_non_empty
SELECT '=== 3. must fail: whitespace-only content ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', '   ', repeat('d', 64),
        'u', now(), now(), 'mem0', 'src-4', now());

-- Catches a writer that stores Mem0's own md5 hash in the fingerprint column.
SELECT '=== 4. must fail: md5-shaped fingerprint (32 chars, not 64) ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', 'x', repeat('e', 32),
        'u', now(), now(), 'mem0', 'src-5', now());

-- MemoryBundle.fingerprints(): duplicates reported, never silently merged.
SELECT '=== 5. must fail: duplicate content fingerprint within one bundle ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', 'different text', repeat('a', 64),
        'u', now(), now(), 'mem0', 'src-6', now());

-- cmm.Embedding.space is (model, dim) together.
SELECT '=== 6. must fail: half an embedding space ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at,
    embedding_model)
VALUES ('11111111-1111-1111-1111-111111111111', '0.1.0', 'x', repeat('f', 64),
        'u', now(), now(), 'mem0', 'src-7', now(), 'some-model');

-- MemoryBundle.embedding_space: the composite FK, the load-bearing case.
SELECT '=== 7. must fail: record embedder disagrees with its bundle ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at,
    embedding, embedding_model, embedding_dim)
VALUES ('11111111-1111-1111-1111-111111111111', '0.1.0', 'x', repeat('9', 64),
        'u', now(), now(), 'mem0', 'src-8', now(),
        ('[' || repeat('0.1,', 383) || '0.1' || ']')::vector, 'a-DIFFERENT-model', 384);

SELECT '=== 8. must fail: vector in a bundle that declares no space ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at,
    embedding, embedding_model, embedding_dim)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', 'x', repeat('8', 64),
        'u', now(), now(), 'mem0', 'src-9', now(),
        ('[' || repeat('0.1,', 383) || '0.1' || ']')::vector,
        'sentence-transformers/all-MiniLM-L6-v2', 384);

SELECT '=== 9. OK: matching embedder is accepted ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at,
    embedding, embedding_model, embedding_dim, embedding_normalized)
VALUES ('11111111-1111-1111-1111-111111111111', '0.1.0', 'vectored', repeat('7', 64),
        'john_001', now(), now(), 'mem0', 'src-10', now(),
        ('[' || repeat('0.1,', 383) || '0.1' || ']')::vector,
        'sentence-transformers/all-MiniLM-L6-v2', 384, true);

-- MATCH SIMPLE: the FK is only checked when all its columns are non-NULL, which
-- is exactly what lets a vector-free record live in a bundle that has a space.
SELECT '=== 10. OK: vector-free record inside a spaced bundle (MATCH SIMPLE) ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at)
VALUES ('11111111-1111-1111-1111-111111111111', '0.1.0', 'no vector here', repeat('6', 64),
        'john_001', now(), now(), 'mem0', 'src-11', now());

-- Uniqueness is per bundle, so re-migrating the same source is not blocked.
SELECT '=== 11. OK: same fingerprint in a different bundle ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at)
VALUES ('11111111-1111-1111-1111-111111111111', '0.1.0', 'John is allergic to shellfish.',
        repeat('a', 64), 'john_001', now(), now(), 'mem0', 'src-12', now());

-- The VECTOR(384) column rejects a wrong-width vector by itself.
SELECT '=== 12. must fail: wrong-width vector ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, prov_source_system, prov_source_id, prov_exported_at,
    embedding, embedding_model, embedding_dim)
VALUES ('11111111-1111-1111-1111-111111111111', '0.1.0', 'y', repeat('4', 64),
        'u', now(), now(), 'mem0', 'src-14', now(),
        '[1,2,3]'::vector, 'sentence-transformers/all-MiniLM-L6-v2', 3);

SELECT '=== 13. the expiry view hides expired rows, the table keeps them ===' AS case;
INSERT INTO memory_record (bundle_id, cmm_version, content, content_fingerprint,
    scope_user_id, created_at, updated_at, expires_at,
    prov_source_system, prov_source_id, prov_exported_at)
VALUES ('22222222-2222-2222-2222-222222222222', '0.1.0', 'expired fact', repeat('5', 64),
        'john_001', now(), now(), '2020-01-01T00:00:00Z', 'mem0', 'src-13', now());

-- Expected: 5 stored, 4 unexpired. The difference is the point -- the expired
-- record is still THERE, which is what Mem0's default read gets wrong.
SELECT count(*) AS rows_stored FROM memory_record;
SELECT count(*) AS rows_unexpired FROM memory_record_unexpired;

-- Tidy up so the file can be run repeatedly.
DELETE FROM memory_record WHERE bundle_id IN (
    '11111111-1111-1111-1111-111111111111',
    '22222222-2222-2222-2222-222222222222');
DELETE FROM memory_bundle WHERE id IN (
    '11111111-1111-1111-1111-111111111111',
    '22222222-2222-2222-2222-222222222222');
