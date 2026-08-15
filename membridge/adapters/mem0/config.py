"""Mem0 configuration, shared by the dump script and the reader.

This lives in the adapter rather than in `scripts/dump_mem0.py` because the two
must agree. The dump writes a Qdrant collection with a particular embedder at a
particular dimensionality; the reader has to open *that* collection with *that*
embedder or it reads nothing, or worse, reads vectors it then mislabels.

Nothing here is left to Mem0's defaults where it matters. Mem0's defaults reach
for OpenAI; if the source and target embed with different models, any semantic
fidelity number conflates migration loss with embedding drift.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: The Qdrant collection the Phase 0 demo writes and reads.
COLLECTION = "membridge_mem0"

#: Pinned deliberately. Local, small, and reproducible on a laptop with no cloud
#: dependency — and identical on both sides of a migration by construction.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

#: Mem0 2.x's extraction system prompt is ~8.5k tokens. Ollama's default context
#: is far smaller and it truncates from the front, silently. See
#: `widen_ollama_context` for the full story.
DEFAULT_OLLAMA_NUM_CTX = 32768

#: qwen2.5-coder:14b, not llama3: llama3's 8192-token maximum cannot hold Mem0's
#: extraction prompt at all, at any context setting.
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:14b"


def build_config(qdrant_path: Path | str, *, collection: str = COLLECTION) -> dict[str, Any]:
    """Mem0 configuration with nothing left to defaults that matters.

    The extraction LLM is a third moving part: Mem0 runs an LLM to decide which
    facts to extract and how to consolidate them, so the *content* of a dump
    depends on which model answers. It defaults to local Ollama; override with
    MEMBRIDGE_LLM_PROVIDER / MEMBRIDGE_LLM_MODEL to reproduce against a hosted
    model. Reads do not invoke the LLM, but Mem0 constructs one regardless, so
    the reader needs this section present and valid too.
    """
    provider = os.environ.get("MEMBRIDGE_LLM_PROVIDER", "ollama")

    if provider == "ollama":
        llm = {
            "provider": "ollama",
            "config": {
                "model": os.environ.get("MEMBRIDGE_LLM_MODEL", DEFAULT_OLLAMA_MODEL),
                "temperature": 0.0,
                "max_tokens": 4000,
                "ollama_base_url": os.environ.get(
                    "OLLAMA_BASE_URL", "http://localhost:11434"
                ),
            },
        }
    else:
        llm = {
            "provider": provider,
            "config": {
                "model": os.environ.get("MEMBRIDGE_LLM_MODEL", "gpt-4o-mini"),
                "temperature": 0.0,
            },
        }

    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "embedding_model_dims": EMBEDDING_DIM,
                "path": str(qdrant_path),
                "on_disk": True,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": EMBEDDING_MODEL,
                "embedding_dims": EMBEDDING_DIM,
            },
        },
        "llm": llm,
    }


def widen_ollama_context(memory: Any, num_ctx: int = DEFAULT_OLLAMA_NUM_CTX) -> None:
    """Force a large Ollama context window for Mem0's extraction call.

    This is not a tuning nicety, it is required for correctness. Mem0 2.x runs a
    single "additive extraction" call whose system prompt is ~33k characters
    (~8.5k tokens). Mem0's Ollama provider sets only temperature, num_predict and
    top_p -- it never sets `num_ctx`, so Ollama applies its small default and
    silently truncates the prompt from the front. The model then sees the INPUTS
    section without the OUTPUT-FORMAT section, returns something shaped nothing
    like `{"memory": [...]}`, and Mem0 swallows the parse failure and reports zero
    extracted facts. The failure is completely silent: `add()` returns
    `{"results": []}` and nothing is logged.

    Note this also rules out any model whose maximum context is below ~9k tokens,
    regardless of this setting -- llama3:latest tops out at 8192 and cannot hold
    the prompt at all.

    Only needed on the write path; `get_all()` never reaches the LLM.
    """
    client = memory.llm.client
    original_chat = client.chat

    def chat_with_context(**params: Any) -> Any:
        options = dict(params.get("options") or {})
        options.setdefault("num_ctx", num_ctx)
        params["options"] = options
        return original_chat(**params)

    client.chat = chat_with_context
