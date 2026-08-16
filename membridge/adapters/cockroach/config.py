"""Connection configuration for the CockroachDB target.

Deliberately thin. Unlike the Mem0 side — where the embedder, vector store and
LLM all had to be pinned because getting any of them wrong silently changes what
a migration means — a SQL target has exactly one thing to get right, which is
which database you are pointed at.

The one thing worth pinning is the *schema* the connection expects, which is
`sql/schema.sql`. A writer talking to a database that has never had that applied
fails on the first insert with a confusing error, so `ensure_schema` exists to
make "did you apply the schema" answerable rather than a guess.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"

#: Matches the single-node insecure cluster the schema was verified against.
#: `sslmode=disable` is right for that and wrong for anything real, which is why
#: this is a default to override rather than a value to rely on.
DEFAULT_DSN = "postgresql://root@localhost:26257/membridge?sslmode=disable"

#: The dimensionality the schema's VECTOR column is declared with. Kept here so
#: the adapter can refuse a mismatched bundle with a clear message instead of
#: letting CockroachDB raise "expected 384 dimensions, not N" from inside a
#: transaction that has already written half a bundle.
SCHEMA_EMBEDDING_DIM = 384

#: Tables and the view the schema defines, in dependency order.
SCHEMA_RELATIONS = ("memory_bundle", "memory_record")


def dsn() -> str:
    return with_trusted_ca(os.environ.get("MEMBRIDGE_COCKROACH_DSN", DEFAULT_DSN))


def with_trusted_ca(raw: str) -> str:
    """Give a verifying DSN a CA bundle it can actually find.

    `sslmode=verify-full` is the right setting and its default trust store is
    not portable. libpq looks in `~/.postgresql/root.crt` and errors when that
    is absent -- which it is on a fresh laptop, and in Lambda, where the error
    helpfully names `/home/sbx_user1051/.postgresql`. The documented escape,
    `sslrootcert=system`, hands the problem to OpenSSL's compiled-in path, and
    psycopg's binary wheel ships its *own* OpenSSL: on macOS that store lacks
    ISRG Root X1, which is what CockroachDB Cloud's certificate chains to, and
    in Lambda it fails verification outright.

    So neither default works in either place, and the two failures look
    different, which makes it tempting to drop to `sslmode=require` -- that
    "fixes" it by disabling verification entirely while still looking encrypted.
    Pointing at certifi's bundle keeps verification full everywhere and depends
    on no host state at all.

    Only applied when the caller is verifying and has not named a bundle, so an
    explicit `sslrootcert=` and the insecure local default both pass through
    untouched.
    """
    verifying = "verify-ca" in raw or "verify-full" in raw
    if "sslrootcert=" in raw or not verifying:
        return raw
    try:
        import certifi
    except ImportError:
        return raw
    return f"{raw}{'&' if '?' in raw else '?'}sslrootcert={certifi.where()}"


def connect(dsn_override: str | None = None, **kwargs: Any) -> Any:
    """Open a connection. Imported late so the module stays importable without a driver."""
    import psycopg

    return psycopg.connect(with_trusted_ca(dsn_override) if dsn_override else dsn(), **kwargs)


def schema_sql() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def ensure_schema(conn: Any) -> None:
    """Apply `sql/schema.sql` if it has not been applied.

    The schema file is written to be idempotent (every statement is IF NOT
    EXISTS / OR REPLACE), so this is safe to call unconditionally — but it is
    still not called automatically by the writer. A migration tool that silently
    creates tables in whatever database you happened to point it at is a
    migration tool that will eventually write someone's memories somewhere they
    did not intend.
    """
    with conn.cursor() as cur:
        cur.execute(schema_sql())
    conn.commit()


def schema_is_present(conn: Any) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FROM information_schema.tables
            WHERE table_name = ANY(%s)
            """,
            (list(SCHEMA_RELATIONS),),
        )
        row = cur.fetchone()
    return bool(row) and row[0] == len(SCHEMA_RELATIONS)
