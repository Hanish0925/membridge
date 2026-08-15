"""CockroachDB adapter: the target side of the first migration pair.

`CockroachWriter` puts a CMM bundle into a database holding `sql/schema.sql`;
`CockroachReader` brings it back. Both are needed before anything can be scored —
a migration that is only written is a claim, and the return trip is the evidence.
"""

from membridge.adapters.cockroach.config import (
    DEFAULT_DSN,
    SCHEMA_EMBEDDING_DIM,
    SCHEMA_PATH,
    connect,
    dsn,
    ensure_schema,
    schema_is_present,
    schema_sql,
)
from membridge.adapters.cockroach.reader import (
    ADAPTER_NAME as READER_ADAPTER_NAME,
    CockroachReadError,
    CockroachReader,
)
from membridge.adapters.cockroach.types import (
    as_float32,
    decode_vector,
    encode_vector,
    is_float32_exact,
    is_unit_length,
)
from membridge.adapters.cockroach.writer import (
    ADAPTER_NAME as WRITER_ADAPTER_NAME,
    CockroachWriteError,
    CockroachWriter,
)

__all__ = [
    "DEFAULT_DSN",
    "SCHEMA_EMBEDDING_DIM",
    "SCHEMA_PATH",
    "connect",
    "dsn",
    "ensure_schema",
    "schema_is_present",
    "schema_sql",
    "CockroachReader",
    "CockroachReadError",
    "READER_ADAPTER_NAME",
    "CockroachWriter",
    "CockroachWriteError",
    "WRITER_ADAPTER_NAME",
    "as_float32",
    "decode_vector",
    "encode_vector",
    "is_float32_exact",
    "is_unit_length",
]
