"""AWS Lambda entry point: the agent, its memory, and nothing else.

Four routes, all POST-or-GET over a Lambda Function URL:

    GET  /health     what this container is holding, and whether the store answers
    POST /search     scoped vector search, no model involved
    POST /ask        the agent: recall, then answer, with the recalls returned
    POST /fidelity   score the live migrated bundle, optionally corrupted first

`/search` exists next to `/ask` for the same reason it does in the CLI. If the
agent gives a poor answer, the interesting question is whether the memory layer
or the model was at fault, and a route that takes the LLM out answers it
directly. A demo that only exposes `/ask` cannot distinguish "retrieved nothing"
from "retrieved the right thing and phrased it badly".

`/fidelity` exists because a static ledger of `5/5` everywhere is
indistinguishable from a scorer that always says so. It runs the real
`membridge.fidelity.score` against whatever is in CockroachDB right now, and
can apply one named, honestly-labelled corruption first -- see
`membridge.fidelity.demo` for what each one actually does and why.

**What is held across invocations, and why.** Lambda reuses a warm container, so
the two expensive things here are created once at module scope and reused: the
ONNX session (a few hundred milliseconds to build) and the CockroachDB
connection (a TLS handshake to us-east-1). Both can die between invocations
while the container survives, so both are checked rather than assumed -- a
connection that Cockroach closed server-side looks perfectly fine to psycopg
until the next execute.

**What is not held: anything about the user.** No conversation history, no
recall cache, no per-user state. That is the same constraint `MemoryAgent`
documents, enforced here at the deployment boundary: every fact the agent uses
came out of CockroachDB during the request that used it. If this process cached
memories between requests, "CockroachDB is the memory layer" would stop being
checkable.
"""

from __future__ import annotations

import base64
import json
import os
import time
import traceback
from typing import Any

#: Where the MiniLM ONNX files are unpacked. Only /tmp is writable in Lambda,
#: and it survives across warm invocations, so a cold start pays the download
#: once per container rather than once per request.
MODEL_CACHE = "/tmp/membridge-onnx"

#: S3 location of the model, as `bucket/prefix`. The model is ~90MB, which is
#: awkward in a deployment zip and trivial in S3 -- and S3 is already in the
#: stack for the static site. Unset means "already on disk", which is how this
#: module is testable locally.
MODEL_S3_ENV = "MEMBRIDGE_ONNX_S3"

CORS_HEADERS = {
    # The static site is served from an S3 website endpoint on a different
    # origin, so the browser preflights every POST. Narrow this to the site's
    # origin if you deploy this beyond a demo.
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

_conn: Any = None
_encoder: Any = None
_cold_started_at = time.time()


# --- lazily built, then reused --------------------------------------------


def _ensure_model() -> None:
    """Put the ONNX model where `membridge.embed` will find it.

    Downloading rather than bundling keeps the deployment package small enough
    to update quickly, which matters more than cold-start latency for a demo.
    """
    from membridge.embed import MODEL_DIR_ENV

    location = os.environ.get(MODEL_S3_ENV)
    if not location:
        return  # MEMBRIDGE_ONNX_DIR is expected to point at a local copy.

    needed = {"model.onnx": os.path.join(MODEL_CACHE, "model.onnx"),
              "tokenizer.json": os.path.join(MODEL_CACHE, "tokenizer.json")}
    if all(os.path.exists(path) for path in needed.values()):
        os.environ[MODEL_DIR_ENV] = MODEL_CACHE
        return

    import boto3  # provided by the Lambda runtime; not a project dependency

    bucket, _, prefix = location.partition("/")
    os.makedirs(MODEL_CACHE, exist_ok=True)
    client = boto3.client("s3")
    for name, path in needed.items():
        if not os.path.exists(path):
            client.download_file(bucket, f"{prefix.rstrip('/')}/{name}", path)
    os.environ[MODEL_DIR_ENV] = MODEL_CACHE


def encoder() -> Any:
    global _encoder
    if _encoder is None:
        _ensure_model()
        from membridge.embed import default_encoder

        _encoder = default_encoder()
    return _encoder


def connection() -> Any:
    """A live connection, reconnecting if the last one died.

    `SELECT 1` rather than `conn.closed`: psycopg only learns a connection was
    dropped server-side when it next uses it, and a Lambda container can sit
    idle for longer than Cockroach's idle timeout. Without this the first
    request after a quiet spell fails and the second succeeds, which is a
    miserable thing to debug from a demo video.
    """
    global _conn
    from membridge.adapters.cockroach import connect

    if _conn is not None:
        try:
            with _conn.cursor() as cur:
                cur.execute("SELECT 1")
            return _conn
        except Exception:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

    _conn = connect()
    # Read-only work: committing after every statement avoids leaving a
    # transaction open across invocations, which would pin MVCC garbage.
    _conn.autocommit = True
    return _conn


# --- request plumbing ------------------------------------------------------


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", **CORS_HEADERS},
        "body": json.dumps(body),
    }


def _parse_body(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _route(event: dict[str, Any]) -> tuple[str, str]:
    context = event.get("requestContext") or {}
    http = context.get("http") or {}
    method = http.get("method") or event.get("httpMethod") or "GET"
    path = http.get("path") or event.get("rawPath") or "/"
    return method.upper(), "/" + path.strip("/")


def _hit_json(hit: Any) -> dict[str, Any]:
    record = hit.record
    mem0_extensions = record.extensions.get("mem0", {})
    return {
        "content": record.content,
        "similarity": hit.similarity,
        "distance": hit.distance,
        "attribution": record.attribution.value,
        "created_at": record.created_at.isoformat(),
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        # The single most interesting field in the payload: it says whether this
        # memory was migrated out of Mem0 or written natively, and the demo's
        # claim is that retrieval cannot tell the difference.
        "source_system": record.provenance.source_system,
        # CMM's scope is user_id/agent_id/session_id/app_id, not just user_id --
        # the migrated records already carry agent_id and session_id (Mem0's
        # run_id), because the fixture that produced them
        # (dump_mem0_multiactor.py) scoped by all three. Omitted when absent so
        # a single-user native record does not grow three empty fields.
        "agent_id": record.scope.agent_id,
        "session_id": record.scope.session_id,
        # `role`/`actor_id` are Mem0's multi-actor attribution, only ever
        # written by the infer=False path -- see the mutually-exclusive
        # attribution finding in CLAUDE.md. Read from extensions rather than a
        # CMM field because CMM has nowhere else to put them.
        "actor_id": mem0_extensions.get("actor_id"),
        "role": mem0_extensions.get("role"),
    }


# --- the routes ------------------------------------------------------------


def _health() -> dict[str, Any]:
    from membridge.adapters.cockroach import schema_is_present

    conn = connection()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM memory_record")
        records = cur.fetchone()[0]
        cur.execute("SELECT source_system, count(*) FROM memory_bundle GROUP BY 1")
        bundles = {row[0]: row[1] for row in cur.fetchall()}

    return {
        "ok": True,
        "schema_present": schema_is_present(conn),
        "records": records,
        "bundles_by_source": bundles,
        "container_age_seconds": round(time.time() - _cold_started_at, 1),
        "encoder_loaded": _encoder is not None,
    }


def _search(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    from membridge.adapters.cockroach import CockroachReader

    query = (body.get("query") or "").strip()
    user_id = (body.get("user_id") or "").strip()
    if not query or not user_id:
        return 400, {"error": "both 'query' and 'user_id' are required"}

    limit = max(1, min(int(body.get("limit") or 5), 20))
    hits = CockroachReader(connection()).search(
        encoder().embed(query),
        user_id=user_id,
        limit=limit,
        include_expired=bool(body.get("include_expired")),
    )
    return 200, {"query": query, "user_id": user_id, "hits": [_hit_json(h) for h in hits]}


#: Seconds held back from the agent's budget so that a degraded answer can
#: still be assembled and returned. The fallback search is an embed plus one
#: scoped query; the rest is JSON. Generous on purpose -- overrunning here means
#: Lambda kills the function and the caller gets nothing at all, which is the
#: one outcome worse than "the model was unreachable".
REPLY_RESERVE_SECONDS = 8.0

#: Used only when there is no Lambda context to ask, i.e. running this module
#: locally. In Lambda the real remaining time is always available and better.
FALLBACK_BUDGET_SECONDS = 40.0


def _ask_budget(context: Any) -> float:
    """How long the agent may run, from the clock that will actually kill it.

    Taken from `get_remaining_time_in_millis` rather than a constant because the
    two costs ahead of it are wildly variable: a cold start pulls 90MB of ONNX
    out of S3 before the agent begins, so a budget measured from the start of
    `ask` can be perfectly reasonable and still overrun the invocation.
    """
    remaining = getattr(context, "get_remaining_time_in_millis", None)
    if remaining is None:
        return FALLBACK_BUDGET_SECONDS
    try:
        return max(1.0, (remaining() / 1000.0) - REPLY_RESERVE_SECONDS)
    except Exception:
        return FALLBACK_BUDGET_SECONDS


#: One label per mode, shown next to the verdict so a viewer knows what was
#: simulated without reading code. Kept beside the route rather than in
#: `membridge.fidelity.demo` because it is prose for this page, not a fact
#: about the corruption itself.
_MODE_LABELS = {
    "intact": "no corruption applied",
    "drop_expiry": "simulated: one record's expires_at was dropped after the read",
    "corrupt_vector": "simulated: one record's embedding was corrupted after the read",
    "simulate_quantization_loss": (
        "hypothetical: a source vector with more precision than float32 can hold "
        "(real Mem0 vectors are already float32-exact, so this cannot happen on "
        "the live data -- it demonstrates what the loss ledger would say if it did)"
    ),
}


def _field_json(field: Any) -> dict[str, Any]:
    return {
        "field": field.field,
        "aligned": field.aligned,
        "survived": field.survived,
        "credit": round(field.credit, 4),
        "rate": round(field.rate, 4),
        "partial_rate": round(field.partial_rate, 4),
        "intact": field.intact,
        "read_scoped": field.read_scoped,
    }


def _loss_json(loss: Any) -> dict[str, Any]:
    return {
        "kind": loss.kind.value,
        "severity": loss.severity.value,
        "field": loss.field,
        "detail": loss.detail,
    }


def _fidelity(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Score the live migrated bundle, optionally corrupted first.

    The point of this route is that it is not the static ledger baked into the
    page at deploy time (see `web/index.html`'s recorded round trip, and the
    reason it stays static: Mem0 is not deployed anywhere this Lambda can reach
    it). This scores whatever is in CockroachDB *right now*, live, with the
    real `membridge.fidelity.score` -- the only thing simulated is the
    corruption itself, named and labelled, never the scoring.
    """
    from membridge.adapters.cockroach import CockroachReader
    from membridge.fidelity import corrupt_vector, drop_expiry, score, simulate_quantization_loss

    mode = body.get("mode") or "intact"
    if mode not in _MODE_LABELS:
        return 400, {"error": f"unknown mode {mode!r}; known: {sorted(_MODE_LABELS)}"}

    reader = CockroachReader(connection())
    bundle_meta = next(
        (b for b in reader.list_bundles() if b["source_system"] == "mem0"), None
    )
    if bundle_meta is None:
        return 404, {"error": "no mem0-sourced bundle in this store"}

    bundle = reader.read_bundle(bundle_meta["id"])

    if mode == "intact":
        source, target = bundle, bundle.model_copy(deep=True)
    elif mode == "drop_expiry":
        source, target = bundle, drop_expiry(bundle)
    elif mode == "corrupt_vector":
        source, target = bundle, corrupt_vector(bundle)
    else:
        source, target = simulate_quantization_loss(bundle)

    report = score(source, target, target_system="cockroachdb")

    return 200, {
        "mode": mode,
        "mode_label": _MODE_LABELS[mode],
        "intact": report.intact,
        "overall": round(report.overall(), 4),
        "source_records": report.source_records,
        "target_records": report.target_records,
        "aligned": report.aligned,
        "fields": [_field_json(f) for f in report.fields],
        "losses": [_loss_json(loss) for loss in report.losses],
        "critical_count": len(report.critical()),
    }


def _ask(body: dict[str, Any], context: Any = None) -> tuple[int, dict[str, Any]]:
    from membridge.adapters.cockroach import CockroachReader
    from membridge.agent import ChatClient, LLMUnavailable, MemoryAgent

    question = (body.get("question") or "").strip()
    user_id = (body.get("user_id") or "").strip()
    if not question or not user_id:
        return 400, {"error": "both 'question' and 'user_id' are required"}

    try:
        llm = ChatClient()
    except LLMUnavailable as exc:
        # Distinct from the model being *unreachable*, which the agent now
        # absorbs. This one means no key was configured, which is a deployment
        # mistake on this side rather than a vendor having a bad afternoon --
        # nothing to degrade to, and nobody but the operator can fix it.
        return 503, {"error": f"model not configured: {exc}"}

    agent = MemoryAgent(
        CockroachReader(connection()),
        encoder(),
        llm,
        budget_seconds=_ask_budget(context),
    )
    answer = agent.ask(question, user_id=user_id)

    # 200 even when degraded, and deliberately: the request succeeded. Memory
    # was searched, the records came back, and they are in this payload. The
    # part that failed is named in `failure` rather than replacing the answer
    # with an error -- a 502 here would report the memory layer as broken on
    # the one occasion it demonstrably was not.
    return 200, {
        "question": question,
        "user_id": user_id,
        "answer": answer.text,
        "turns": answer.turns,
        "degraded": answer.degraded,
        "failure": answer.failure,
        "recalled": [
            {
                "query": recall.query,
                "fallback": recall.fallback,
                "hits": [_hit_json(h) for h in recall.hits],
            }
            for recall in answer.recalled
        ],
        "memories_used": [_hit_json(h) for h in answer.memories_used],
    }


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    method, path = _route(event)

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    try:
        if path in ("/", "/health") and method == "GET":
            return _response(200, _health())
        if path == "/search" and method == "POST":
            status, body = _search(_parse_body(event))
            return _response(status, body)
        if path == "/fidelity" and method == "POST":
            status, body = _fidelity(_parse_body(event))
            return _response(status, body)
        if path == "/ask" and method == "POST":
            status, body = _ask(_parse_body(event), context)
            return _response(status, body)
        return _response(404, {"error": f"no route for {method} {path}"})
    except Exception as exc:
        # The traceback goes to CloudWatch; the caller gets the type and message.
        # A demo that swallows the reason for a failure wastes the time of
        # whoever is watching it.
        traceback.print_exc()
        return _response(500, {"error": f"{type(exc).__name__}: {exc}"})
