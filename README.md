# MemBridge

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Tests: 107 passing](https://img.shields.io/badge/tests-107%20passing-brightgreen.svg)](#tests)

**Migrating agent memory between stores without lying about what survived.**

An agent's memory is the one part of it you cannot regenerate. Prompts can be
rewritten and models swapped, but the accumulated record of a user — their
allergies, their preferences, what they told you last March — exists in exactly
one place, and moving it between vendors is a one-way door. "We migrated your
memories" is an unfalsifiable claim unless somebody measures what actually
arrived.

MemBridge reads a source memory store into a vendor-neutral **Common Memory
Model (CMM)**, writes it to CockroachDB, and **scores the round trip
field-by-field**. Then it runs an agent whose entire memory is that CockroachDB
table — no history buffer, no cache — so the migration's result is something you
can interrogate rather than take on trust.

First real target pair: **Mem0 → CockroachDB**.

**Live demo:** https://membridge-demo-587104705068.s3.us-east-1.amazonaws.com/index.html
**API:** https://mz86k14151.execute-api.us-east-1.amazonaws.com — try
`curl -s https://mz86k14151.execute-api.us-east-1.amazonaws.com/health`

The same memory the web demo queries, from the CLI:

![`membridge --help` listing the schema, bundles, migrate, search and ask commands](docs/img/cli-help.png)

```
Mem0 (Qdrant + local LLM)  ──read──▶  CMM  ──write──▶  CockroachDB Cloud
                                       │                    │
                                       └──── score ─────────┘
                                     15 fields, per record

                       AWS Lambda ──vector search──▶ CockroachDB Cloud
                            ▲                          (us-east-1)
                            │
                    S3 static demo site
```

**Contents:** [The result](#the-result) ·
[Which tools](#which-tools-and-what-each-one-does) ·
[What makes this different](#what-makes-this-different-from-a-rag-demo) ·
[Running it](#running-it) ·
[Layout](#layout) ·
[Tests](#tests) ·
[Team](#team)

---

## The result

Migrating the ground-truth Mem0 store into CockroachDB Cloud:

```
FIDELITY  mem0 -> cockroachdb
source records       : 5      aligned (fingerprint): 5
missing in target    : 0      unexpected in target : 0

content 5/5   scope 5/5   attribution 5/5   created_at 5/5   updated_at 5/5
expires_at 5/5   metadata 5/5   extensions 5/5   cmm_version 5/5   id 5/5
provenance 5/5   embedding.model 5/5   embedding.dim 5/5
embedding.normalized 5/5   embedding.vector 5/5

vectors are compared bit-exactly, not approximately
MIGRATION INTACT
```

Reported per field rather than as one percentage on purpose. "97% migrated"
hides *which* 3%, and whether the lost part was content or timestamps matters
enormously.

Then, against the migrated store — note the third result:

```
$ membridge search "what food can I not eat" --user-id john_001
1. sim 0.5866  I cannot eat shrimp, crab, lobster or anything cooked in the same fryer.
2. sim 0.5054  Maria does not eat meat or fish, but she does eat eggs and dairy.
3. sim 0.3151  My name is John and I am allergic to shellfish.      ← migrated from Mem0
```

The agent cannot tell which of its memories were migrated. That is the point: a
memory layer that treats migrated records as second-class has not really
migrated them.

Five is the size of the ground-truth fixture, not the scale being tested. The
CockroachDB target these five landed in holds 1061 records total — the 5
migrated, 56 written natively for the same user, and 1000 bulk records under
other users — and every search and every fidelity score above runs against
that whole table, not a slice pulled out for the demo.

---

## Which tools, and what each one does

### CockroachDB (3 of the 4 listed tools)

| Tool | What it actually does here |
|---|---|
| **Distributed Vector Indexing** | `memory_record_embedding_idx` is a CSPANN index on `VECTOR(384)`, partitioned by `scope_user_id`. Every memory read the agent makes goes through it. Confirmed used rather than assumed — `EXPLAIN` on a scoped query reports `• vector search … prefix spans: [/'john_001' - /'john_001']`, and the same query unscoped reports `FULL SCAN`. |
| **ccloud CLI (agent-ready)** | Provisioned the whole target: cluster, SQL user, database. The demo cluster was created and configured entirely from the CLI, which is what let this be scripted rather than clicked. |
| **CockroachDB as the memory layer** | Not a cache in front of something else. `membridge/agent/memory_agent.py` holds no conversational state at all — every fact in every answer was retrieved from CockroachDB during the turn it was used in. Restart the process and the agent knows exactly as much as before. |

CockroachDB is also doing schema-level work that a vector store cannot: the CMM
invariants are enforced as `CHECK`, `UNIQUE` and composite `FOREIGN KEY`
constraints, so the target physically cannot hold a record CMM would have
rejected. `sql/verify_constraints.sql` is the falsifier — 14 cases asserting each
constraint rejects what it claims to. All 14 verified on CockroachDB **v26.2.5**
(Cloud) and **v25.4.0** (local).

### AWS (3 services)

| Service | What it actually does here |
|---|---|
| **AWS Lambda** | Runs the agent and the retrieval API. Holds the ONNX encoder and the database connection across warm invocations, and holds nothing about the user between requests — the zero-persistence constraint is enforced at the deployment boundary, not just in a docstring. |
| **Amazon S3** | Serves the static demo site, stores the 90MB MiniLM ONNX model that Lambda pulls into `/tmp` on cold start, and carries the deployment package itself. Keeping the model out of the package is what keeps a code update a ~48MB upload instead of a ~140MB one. |
| **Amazon API Gateway** | The public HTTP entry point. Not the original design — a Lambda Function URL was, and it returns `403 AccessDeniedException` to anonymous callers despite `AuthType: NONE` and a resource policy granting `lambda:InvokeFunctionUrl` to `Principal: "*"`, with no organization or SCP in play. API Gateway invokes Lambda as a service principal rather than publicly, which sidesteps it. |

Both in `us-east-1`, the same region as the cluster: the agent makes several
scoped vector queries per answer, so a cross-region hop would be paid repeatedly
per request rather than once.

---

## What makes this different from a RAG demo

Most agent-memory demos are unfalsifiable in the same way "we migrated your
memories" is. These are the specific places MemBridge refuses to be:

- **Fidelity is scored, not asserted.** `membridge/fidelity/` aligns source and
  target records by a content fingerprint that depends on nothing a migration
  may legitimately change — not the id, not the timestamps, not the vendor's own
  hash — then compares all 15 fields with partial credit and a loss ledger.

- **Vectors are compared bit-exactly.** CockroachDB's `VECTOR` is float32 and
  prints nine significant digits; parsed as a float64 that string is a
  *different number*. All 384 source components were float32-exact and **zero**
  of the naively-returned ones were, so a comparison written without knowing
  this reports total embedding loss on a flawless migration. `decode_vector`
  rounds back through float32 and recovers the stored value exactly.

- **Similarity scores are refused rather than faked.** Cosine between vectors
  from different embedders is a meaningless number that still sorts, so
  `Embedding` carries its model name and a query from the wrong space raises.
  CockroachDB only index-accelerates L2, and L2 ranks identically to cosine
  *only for unit-length vectors* — so `embedding_normalized` is a column, NULL
  means "not checked", and a hit from an unverified space gets `similarity=None`
  rather than a converted number that would be wrong.

- **Expired memories are handled as a decision, not a default.** Mem0's
  `get_all()` silently drops expired records, which is backwards for a
  migration — an expired memory is still data the source holds. MemBridge always
  reads them, and makes hiding them something a *retrieval* caller asks for by
  name. In the demo, the highest-similarity memory for "dinner plans" is a
  reservation that lapsed two days ago, and it is correctly withheld.

- **Vendor data that CMM does not model is preserved, namespaced, verbatim.**
  A test asserts every field Mem0 can emit has a declared destination; add a
  field to Mem0's output and it fails.

- **Migration throughput is not oversold.** `CockroachWriter` inserts one
  record at a time on purpose — CockroachDB's own guidance is that large
  batched `VECTOR` writes degrade — so migrating 1000 records from India to
  `us-east-1` took several minutes, not seconds. A migration of real size wants
  the writer running next to the cluster. Streaming a source larger than
  memory is open work, tracked as such in `CLAUDE.md`, not assumed away.

Every vendor behaviour above was found by reading the installed source, not the
documentation. Several contradict what the documentation implies.

---

## Running it

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd membridge
uv sync
```

### 1. Point at a CockroachDB target

```bash
ccloud cluster create membridge --plan basic          # or use an existing one
ccloud cluster user create membridge membridge -p '<password>'
ccloud cluster database create membridge membridge
ccloud cluster sql membridge --connection-url --database membridge -u membridge
```

Put the connection string in `.env`:

```
MEMBRIDGE_COCKROACH_DSN=postgresql://membridge:<password>@<host>:26257/membridge?sslmode=verify-full
GROQ_API_KEY=<from console.groq.com/keys>          # for local use
GEMINI_API_KEY=<from aistudio.google.com/apikey>   # for the deployed agent
```

> **Two LLM keys, and it is not belt-and-braces.** Groq serves `GET /models`
> from an AWS IP perfectly well and returns `401 Invalid API Key` for
> `POST /chat/completions` with that same key — the key is fine, the source
> address is not. So Groq is the local default and Gemini is what the deployed
> function uses. Either works on a laptop; only Gemini works from Lambda.

> **TLS needs no setup.** `sslmode=verify-full` normally sends libpq looking for
> `~/.postgresql/root.crt`, which does not exist on a fresh machine, and
> `sslrootcert=system` resolves to a trust store that lacks ISRG Root X1 on
> macOS and fails outright in Lambda. `config.with_trusted_ca` points a
> verifying DSN at certifi's bundle instead, so verification stays full and
> depends on nothing about the host.

Then:

```bash
uv run membridge schema --apply       # creates the tables; never implicit
```

### 2. Migrate, and score it

```bash
uv run python scripts/dump_mem0_multiactor.py         # regenerate ground truth

uv run membridge migrate \
    --qdrant ./data/qdrant_multiactor \
    --collection membridge_mem0_multiactor \
    --user-id john_001 --agent-id support_agent_v2 --run-id thread_8891
```

Exits non-zero if anything was lost, so it is usable in a pipeline that should
stop when a migration silently drops something.

### 3. Seed the demo corpus

```bash
uv run python scripts/seed_demo.py --replace
```

The bulk records exist for the query planner, not the demo: CockroachDB only
chooses the vector index once statistics say the table is worth an index scan,
so a store holding five records answers every search with a full scan and proves
nothing.

### 4. Query it

```bash
uv run membridge search "what food can I not eat" --user-id john_001
uv run membridge search "dinner plans" --user-id john_001 --include-expired
uv run membridge ask "What should I know before booking dinner?" --user-id john_001
uv run membridge bundles
```

### 5. Deploy the demo

```bash
aws configure
./scripts/deploy_aws.sh
```

Idempotent; run it again to push new code. It prints the site URL and the API
URL. Note that it makes one S3 bucket publicly readable — a demo URL has to be
reachable without credentials. The DSN and the API key are set only as Lambda
environment variables; they are never in the bucket and never in the page.

Two things in that script look arbitrary and are not:

- **The runtime is `python3.13`, not `python3.11`.** Lambda's 3.11 is Amazon
  Linux 2 (glibc 2.26), which only accepts `manylinux2014` wheels — and the
  newest onnxruntime published for that tag is 1.16.3, compiled against numpy
  1.x. Installed next to the numpy 2.x that resolves alongside it, the function
  dies on import with `_ARRAY_API not found`, which reads as a numpy bug and is
  really a wheel-tag one. 3.13 is Amazon Linux 2023, so `manylinux_2_28` applies
  and onnxruntime 1.28 installs — the pairing the tests already run against.
- **The package goes up via S3, not `--zip-file`.** It is ~48MB against a 50MB
  inline limit; the S3 path allows 250MB, so a dependency bump does not become a
  deploy failure with a misleading "request entity too large".

---

## Layout

```
membridge/cmm/            the Common Memory Model — schema v0.1.0
membridge/adapters/mem0/  read a live Mem0 instance, or a recorded dump
membridge/adapters/cockroach/  write and read CMM, plus scoped vector search
membridge/fidelity/       score a round trip: alignment, per-field, loss ledger
membridge/embed/          MiniLM/384 query encoder via onnxruntime (no torch)
membridge/agent/          the agent, and a dependency-free chat client
membridge/serve/          the Lambda handler
membridge/cli/            typer front end over all of it
sql/schema.sql            CMM mapped onto CockroachDB
sql/verify_constraints.sql  14 cases proving each constraint bites
scripts/                  ground-truth dumps, round trip, seed, deploy
web/index.html            the demo page
```

`CLAUDE.md` is the engineering log: every design decision, every vendor finding,
and the open questions that are still open. It is more honest than this README
and considerably longer.

## Tests

```bash
uv run pytest tests/ -q       # 107 passed, 12 skipped
```

The 12 skips need a live CockroachDB and skip cleanly without one. A skipped
test says so; a mocked database would have confirmed the wrong behaviour.

## Team

- **Pushpak Hanish** — [@Hanish0925](https://github.com/Hanish0925)
- **Varun Nihar** — [@vNihar007](https://github.com/vNihar007)
- **Manideep** — [@manideep0921](https://github.com/manideep0921)

## License

MIT.
