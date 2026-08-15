"""Embedding a query in the same space the migrated memories already live in.

This module exists because of a constraint that is easy to get backwards. The
memories in CockroachDB carry vectors that came *out of Mem0 bit-exactly* -- that
is the migration's central claim, and re-embedding them with something else at
write time would quietly destroy it. So the store's embedding space is fixed at
`sentence-transformers/all-MiniLM-L6-v2` / 384d, and anything that wants to
search it must produce vectors in that same space. The query embedder is not a
free choice; it is determined by what was migrated.

The obvious way to honour that is to run sentence-transformers itself. That is
what `scripts/dump_mem0*.py` do, and it pulls in torch -- around 2.5GB, which
does not fit in a Lambda zip and makes a container cold-start take tens of
seconds. So this runs the *same weights* through onnxruntime instead: about
90MB of model and no torch at all.

"Same weights" is a claim, and this project does not ship those unmeasured.
`tests/test_embed.py` pins the agreement against sentence-transformers directly:
cosine 1.0 to float32, max component delta ~1.5e-7. That is what licenses this
encoder to declare `model="sentence-transformers/all-MiniLM-L6-v2"` and be
accepted by `CockroachReader.search`, which refuses vectors from any other
space. If the two ever diverge past float32 noise, that test fails and the
declaration becomes a lie the search would not catch.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from membridge.cmm import Embedding

#: The exact string Mem0 records in its Qdrant payload, and therefore the string
#: stored in `memory_record.embedding_model`. It has to match character for
#: character or `CockroachReader.search` will refuse the query -- correctly.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIM = 384

#: MiniLM's trained maximum. Mem0 truncates at the same point, so a query longer
#: than this is embedded from its first 256 tokens on both sides.
MAX_TOKENS = 256

#: Set to a directory holding `model.onnx` and `tokenizer.json` to skip the
#: Hugging Face download. This is how the Lambda package supplies the model:
#: there is no network egress budget for a 90MB fetch on a cold start.
MODEL_DIR_ENV = "MEMBRIDGE_ONNX_DIR"


class EncoderUnavailable(RuntimeError):
    """Raised when the ONNX runtime or the model files cannot be loaded."""


def _resolve_model_files() -> tuple[str, str]:
    """(model.onnx, tokenizer.json), from a local directory or the HF hub."""
    local = os.environ.get(MODEL_DIR_ENV)
    if local:
        directory = Path(local)
        model, tokenizer = directory / "model.onnx", directory / "tokenizer.json"
        missing = [p.name for p in (model, tokenizer) if not p.exists()]
        if missing:
            raise EncoderUnavailable(
                f"{MODEL_DIR_ENV}={local} but {', '.join(missing)} is not there"
            )
        return str(model), str(tokenizer)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise EncoderUnavailable(
            f"no {MODEL_DIR_ENV} set and huggingface_hub is not installed"
        ) from exc

    return (
        hf_hub_download(MODEL_NAME, "onnx/model.onnx"),
        hf_hub_download(MODEL_NAME, "tokenizer.json"),
    )


class MiniLMEncoder:
    """MiniLM through onnxruntime, producing CMM `Embedding` objects.

    Deliberately not a general-purpose embedding interface. It encodes into one
    named space and says which, because an encoder that can silently produce
    vectors in the wrong space is the failure `Embedding.model` exists to make
    impossible.
    """

    def __init__(self) -> None:
        try:
            import numpy
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise EncoderUnavailable(
                "MiniLMEncoder needs onnxruntime, tokenizers and numpy: "
                "install membridge[agent]"
            ) from exc

        self._np = numpy
        model_path, tokenizer_path = _resolve_model_files()

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(MAX_TOKENS)

        # One thread: Lambda gives a small vCPU slice, and onnxruntime's default
        # thread pool spends more time coordinating than embedding on a batch
        # this size.
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Mean-pooled, L2-normalized sentence vectors.

        Mean pooling over the attention mask then normalizing is what
        all-MiniLM-L6-v2's sentence-transformers configuration does; the ONNX
        graph emits token vectors only, so the pooling has to happen here. Get
        it wrong -- pool without the mask, or skip the normalize -- and the
        vectors still look plausible and rank badly, which is the kind of bug
        that shows up as "the agent's memory seems vague".
        """
        np = self._np
        if not texts:
            return []

        encodings = [self._tokenizer.encode(text) for text in texts]
        width = max(len(encoding.ids) for encoding in encodings)
        ids = np.array(
            [e.ids + [0] * (width - len(e.ids)) for e in encodings], dtype=np.int64
        )
        mask = np.array(
            [e.attention_mask + [0] * (width - len(e.attention_mask)) for e in encodings],
            dtype=np.int64,
        )

        tokens = self._session.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            },
        )[0]

        weights = mask[..., None].astype(np.float32)
        pooled = (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled.astype(np.float32).tolist()

    def embed(self, text: str) -> Embedding:
        """One query, as the CMM object `CockroachReader.search` expects.

        `normalized=True` is asserted rather than measured here only because
        `encode` normalizes as its last step; the test suite checks it against
        `cmm.is_unit_length` rather than taking this line's word for it.
        """
        (vector,) = self.encode([text])
        return Embedding(
            vector=vector, model=MODEL_NAME, dim=MODEL_DIM, normalized=True
        )


@lru_cache(maxsize=1)
def default_encoder() -> MiniLMEncoder:
    """A process-wide encoder.

    Loading the ONNX session takes a noticeable fraction of a second, and a
    Lambda container serves many invocations. Caching at module scope is what
    turns a warm invocation into a few milliseconds of work.
    """
    return MiniLMEncoder()
