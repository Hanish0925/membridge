"""The MemBridge command line.

Every import inside a command rather than at module scope, deliberately. The
heavy paths here pull in torch (via mem0), onnxruntime, or a database driver,
and a CLI that takes four seconds to print `--help` is one nobody explores. It
also means `membridge ask` works on a machine that could not run `membridge
migrate`, which is exactly the Lambda case.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Migrate agent memory between stores, and measure what survived.",
)


def _err(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)


# --- target management -----------------------------------------------------


@app.command()
def schema(
    dsn: Optional[str] = typer.Option(None, help="CockroachDB DSN; else MEMBRIDGE_COCKROACH_DSN."),
    apply: bool = typer.Option(False, "--apply", help="Create the tables if absent."),
) -> None:
    """Check, or create, the CMM schema in a CockroachDB target.

    Creation is opt-in by flag for the same reason the writer never calls
    `ensure_schema` implicitly: a migration tool that silently creates tables in
    whatever database it was pointed at will eventually write someone's memories
    somewhere they did not intend.
    """
    from membridge.adapters.cockroach import connect, ensure_schema, schema_is_present

    conn = connect(dsn) if dsn else connect()
    present = schema_is_present(conn)
    typer.echo(f"schema present: {present}")

    if present or not apply:
        if not present:
            typer.echo("run again with --apply to create it")
        raise typer.Exit(0 if present else 1)

    ensure_schema(conn)
    typer.secho("schema applied", fg=typer.colors.GREEN)


@app.command()
def bundles(
    dsn: Optional[str] = typer.Option(None, help="CockroachDB DSN."),
) -> None:
    """List the migration bundles held in a target."""
    from membridge.adapters.cockroach import CockroachReader, connect

    conn = connect(dsn) if dsn else connect()
    rows = CockroachReader(conn).list_bundles()
    if not rows:
        typer.echo("no bundles in this target")
        raise typer.Exit(0)

    typer.echo(f"{'bundle':<38}{'source':<12}{'records':>8}  imported")
    for row in rows:
        typer.echo(
            f"{str(row['id']):<38}{row['source_system']:<12}"
            f"{row['records']:>8}  {row['imported_at']:%Y-%m-%d %H:%M}"
        )


# --- the migration ---------------------------------------------------------


@app.command()
def migrate(
    qdrant: str = typer.Option(..., help="Path to the Mem0 Qdrant store."),
    collection: str = typer.Option(..., help="Qdrant collection name."),
    user_id: str = typer.Option(..., help="Mem0 user_id to migrate."),
    agent_id: Optional[str] = typer.Option(None, help="Mem0 agent_id, if scoped."),
    run_id: Optional[str] = typer.Option(None, help="Mem0 run_id, if scoped."),
    dsn: Optional[str] = typer.Option(None, help="CockroachDB DSN."),
    score_it: bool = typer.Option(True, "--score/--no-score", help="Score the round trip."),
    json_out: Optional[str] = typer.Option(None, "--json", help="Write the report here."),
) -> None:
    """Mem0 -> CMM -> CockroachDB, then read it back and score it.

    Scoring is on by default and that is the whole argument of this tool: a
    migration that reports only "written, no errors" is a claim, and the return
    trip is the evidence. Pass --no-score if you have a reason.
    """
    from mem0 import Memory

    from membridge.adapters.cockroach import (
        CockroachReader,
        CockroachWriter,
        connect,
        schema_is_present,
    )
    from membridge.adapters.mem0 import Mem0Reader, build_config
    from membridge.fidelity import render_text, score as score_bundles

    scope = {"user_id": user_id}
    if agent_id:
        scope["agent_id"] = agent_id
    if run_id:
        scope["run_id"] = run_id

    config = build_config(qdrant, collection=collection)
    source = Mem0Reader(Memory.from_config(config)).read_bundle(**scope)
    typer.echo(f"[mem0]  {len(source.records)} records, space={source.embedding_space}")

    conn = connect(dsn) if dsn else connect()
    if not schema_is_present(conn):
        _err("target has no CMM schema; run `membridge schema --apply` first")
        raise typer.Exit(2)

    bundle_id = CockroachWriter(conn).write_bundle(source)
    # write_bundle opens its own transaction but leaves the commit to the
    # caller, so that a caller can migrate and verify before making either
    # visible. A CLI has no such caller: without this, the process exits, the
    # transaction rolls back, and the report above cheerfully says INTACT about
    # a bundle that no longer exists. Found by running `migrate` and then
    # `search` and getting nothing back.
    conn.commit()
    typer.secho(f"[crdb]  wrote bundle {bundle_id}", fg=typer.colors.GREEN)

    if not score_it:
        return

    target = CockroachReader(conn).read_bundle(bundle_id)
    report = score_bundles(source, target, target_system="cockroachdb")
    typer.echo("")
    typer.echo(render_text(report))

    if json_out:
        import json
        from pathlib import Path

        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2))
        typer.echo(f"\nreport written to {json_out}")

    # Non-zero on loss, so this is usable in a pipeline that should stop when a
    # migration silently drops something.
    raise typer.Exit(0 if report.intact else 1)


# --- retrieval and the agent ----------------------------------------------


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for."),
    user_id: str = typer.Option(..., help="Whose memory to search."),
    limit: int = typer.Option(5, help="How many memories to return."),
    dsn: Optional[str] = typer.Option(None, help="CockroachDB DSN."),
    include_expired: bool = typer.Option(False, help="Also return expired memories."),
) -> None:
    """Semantic recall straight against the store, with no model in the way.

    Useful precisely because it takes the LLM out: if `membridge ask` gives a
    poor answer, this says whether the memory layer or the model is at fault.
    """
    from membridge.adapters.cockroach import CockroachReader, connect
    from membridge.embed import default_encoder

    conn = connect(dsn) if dsn else connect()
    embedding = default_encoder().embed(query)
    hits = CockroachReader(conn).search(
        embedding, user_id=user_id, limit=limit, include_expired=include_expired
    )

    if not hits:
        typer.echo("no memories found")
        raise typer.Exit(0)

    for index, hit in enumerate(hits, start=1):
        score_text = (
            f"sim {hit.similarity:.4f}"
            if hit.similarity is not None
            else f"L2 {hit.distance:.4f}  (space not known unit-length)"
        )
        typer.echo(f"{index}. {score_text}  {hit.record.content}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="What to ask."),
    user_id: str = typer.Option(..., help="Whose memory the agent may read."),
    dsn: Optional[str] = typer.Option(None, help="CockroachDB DSN."),
    provider: Optional[str] = typer.Option(None, help="groq or gemini."),
    model: Optional[str] = typer.Option(None, help="Override the model name."),
    show_memory: bool = typer.Option(True, help="Print what was recalled."),
) -> None:
    """Ask the agent, and see which memories it used to answer.

    The recalled memories are printed by default because the answer alone is
    not the interesting part. An assistant that says the right thing while
    having retrieved nothing is guessing, and this is how you can tell.
    """
    from membridge.adapters.cockroach import CockroachReader, connect
    from membridge.agent import ChatClient, LLMUnavailable, MemoryAgent
    from membridge.embed import default_encoder

    conn = connect(dsn) if dsn else connect()
    try:
        llm = ChatClient(provider=provider, model=model)
    except LLMUnavailable as exc:
        _err(str(exc))
        raise typer.Exit(2)

    agent = MemoryAgent(CockroachReader(conn), default_encoder(), llm)
    try:
        answer = agent.ask(question, user_id=user_id)
    except LLMUnavailable as exc:
        _err(f"memory was reachable, the model was not: {exc}")
        raise typer.Exit(3)

    if show_memory:
        for recall in answer.recalled:
            typer.secho(
                f"  recall({recall.query!r}) -> {len(recall.hits)} memories",
                fg=typer.colors.BRIGHT_BLACK,
            )
        for hit in answer.memories_used:
            score_text = (
                f"{hit.similarity:.3f}" if hit.similarity is not None else f"L2 {hit.distance:.3f}"
            )
            typer.secho(f"    [{score_text}] {hit.record.content}", fg=typer.colors.BRIGHT_BLACK)
        typer.echo("")

    typer.echo(answer.text)


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
