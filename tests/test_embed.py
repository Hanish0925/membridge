"""The query encoder, and the claim that licenses it.

`MiniLMEncoder` declares `model="sentence-transformers/all-MiniLM-L6-v2"`, which
is what lets `CockroachReader.search` accept its vectors against a store
migrated from Mem0. That declaration is only true if running the weights through
onnxruntime gives what running them through sentence-transformers gives. This
file is where that stops being an assumption.

It is the same move as measuring Mem0's vector norms instead of assuming them:
the difference between "the same embedding space" and "a space that looks
similar" is invisible until retrieval is quietly wrong.

Skips cleanly where onnxruntime or the model files are unavailable — the model
is a 90MB download, and a mocked encoder would confirm nothing.
"""

from __future__ import annotations

import pytest

from membridge.cmm import is_unit_length

pytest.importorskip("onnxruntime", reason="membridge[agent] not installed")

from membridge.embed import MODEL_DIM, MODEL_NAME, EncoderUnavailable, MiniLMEncoder

TEXTS = [
    "John is allergic to shellfish.",
    "The customer prefers window seats on morning flights.",
    "Refund was issued on Tuesday after the escalation.",
]


@pytest.fixture(scope="module")
def encoder() -> MiniLMEncoder:
    try:
        return MiniLMEncoder()
    except EncoderUnavailable as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"encoder unavailable: {exc}")


def test_vectors_are_the_shape_the_schema_expects(encoder: MiniLMEncoder) -> None:
    vectors = encoder.encode(TEXTS)
    assert len(vectors) == len(TEXTS)
    assert {len(v) for v in vectors} == {MODEL_DIM}


def test_vectors_are_unit_length_by_cmms_own_test(encoder: MiniLMEncoder) -> None:
    """Not "we called normalize", but "cmm.is_unit_length agrees".

    The Cockroach schema's L2 index only ranks like cosine for unit vectors, and
    `embedding_normalized` is the column that records whether anyone checked.
    This is the check.
    """
    for vector in encoder.encode(TEXTS):
        assert is_unit_length(vector)


def test_embed_declares_the_space_it_produced(encoder: MiniLMEncoder) -> None:
    embedding = encoder.embed(TEXTS[0])
    assert (embedding.model, embedding.dim) == (MODEL_NAME, MODEL_DIM)
    assert embedding.normalized is True
    assert is_unit_length(embedding.vector), "the declaration must be true, not asserted"


def test_onnx_agrees_with_sentence_transformers(encoder: MiniLMEncoder) -> None:
    """The load-bearing test of this module.

    If this fails, `MiniLMEncoder` is claiming an embedding space it does not
    actually produce vectors in, and `CockroachReader.search` -- which checks
    the space by *name* -- would accept them and rank the results wrongly
    without any error anywhere.

    Measured on the real models: cosine 1.0 to float32 precision, max component
    delta ~1.5e-7. The threshold below is deliberately far tighter than "close
    enough to retrieve well", because the failure being guarded against is a
    different model, not a noisier one.
    """
    sentence_transformers = pytest.importorskip(
        "sentence_transformers", reason="reference implementation not installed"
    )
    import numpy as np

    reference = sentence_transformers.SentenceTransformer(MODEL_NAME)
    expected = reference.encode(TEXTS, normalize_embeddings=True)
    actual = np.array(encoder.encode(TEXTS))

    for index, text in enumerate(TEXTS):
        cosine = float(np.dot(actual[index], expected[index]))
        delta = float(np.abs(actual[index] - expected[index]).max())
        assert cosine == pytest.approx(1.0, abs=1e-6), f"diverged on {text!r}"
        assert delta < 1e-5, f"component delta {delta:.3g} on {text!r}"


def test_an_empty_batch_is_not_an_error(encoder: MiniLMEncoder) -> None:
    assert encoder.encode([]) == []
