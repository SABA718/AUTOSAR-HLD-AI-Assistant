"""Unit tests for Dense Text Embeddings module (Stage 5)."""

import math
import numpy as np
import pytest

from core.embeddings.embedder import SentenceTransformerEmbedder, l2_normalize


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    """Fixture providing a cached SentenceTransformerEmbedder instance."""
    return SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")


def test_embedding_dimensionality(embedder: SentenceTransformerEmbedder):
    """Verify vector dimensionality matches expected 384 dimensions for all-MiniLM-L6-v2."""
    assert embedder.dimension == 384

    text = "Software Component provides brake torque arbitration services."
    vector = embedder.embed_text(text)

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


def test_embedding_l2_normalization(embedder: SentenceTransformerEmbedder):
    """Verify that generated vectors are strictly L2 normalized (unit length)."""
    texts = [
        "AUTOSAR Classic Platform Software Component",
        "Sender-Receiver interface signal mapping",
        "RPort and PPort communication specifications",
    ]

    # Test single text embedding normalization
    vec_single = embedder.embed_text(texts[0])
    norm_single = np.linalg.norm(vec_single)
    assert norm_single == pytest.approx(1.0, rel=1e-5)

    # Test batch embedding normalization
    vectors = embedder.embed_batch(texts)
    assert len(vectors) == len(texts)
    for vec in vectors:
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, rel=1e-5)


def test_l2_normalize_helper_and_edge_cases():
    """Verify l2_normalize function on arbitrary vectors and edge cases."""
    # Standard vector [3.0, 4.0] has norm 5.0 -> [0.6, 0.8]
    raw_vec = [3.0, 4.0]
    normalized = l2_normalize(raw_vec)
    assert normalized[0] == pytest.approx(0.6)
    assert normalized[1] == pytest.approx(0.8)
    assert np.linalg.norm(normalized) == pytest.approx(1.0)

    # Zero vector does not divide by zero or raise NaN
    zero_vec = [0.0, 0.0, 0.0]
    norm_zero = l2_normalize(zero_vec)
    assert norm_zero == [0.0, 0.0, 0.0]


def test_embedding_batch_behavior(embedder: SentenceTransformerEmbedder):
    """Verify batch handling: empty lists, single items, batch size chunking, and consistency."""
    # 1. Empty batch
    empty_result = embedder.embed_batch([])
    assert empty_result == []

    # 2. Single item batch
    single_batch = embedder.embed_batch(["Single automotive sensor signal"])
    assert len(single_batch) == 1
    assert len(single_batch[0]) == 384

    # 3. Equivalence: embed_text(x) vs embed_batch([x])[0]
    sample_text = "ECU abstraction layer communicates directly with MCAL drivers."
    single_vector = embedder.embed_text(sample_text)
    batch_vector = embedder.embed_batch([sample_text])[0]

    np.testing.assert_allclose(single_vector, batch_vector, atol=1e-5)

    # 4. Batch size splitting: 5 items with batch_size=2
    multi_texts = [
        f"AUTOSAR architecture component description number {i}"
        for i in range(5)
    ]
    chunked_batch_vectors = embedder.embed_batch(multi_texts, batch_size=2)
    assert len(chunked_batch_vectors) == 5
    for vec in chunked_batch_vectors:
        assert len(vec) == 384
        assert np.linalg.norm(vec) == pytest.approx(1.0, rel=1e-5)


def test_embedding_semantic_similarity(embedder: SentenceTransformerEmbedder):
    """Verify semantic ordering: related automotive terms score higher than unrelated terms."""
    vec_query = np.array(embedder.embed_text("Require Port for wheel speed sensor"))
    vec_related = np.array(embedder.embed_text("RPort receiving wheel speed data"))
    vec_unrelated = np.array(embedder.embed_text("Delicious recipe for chocolate fudge cake"))

    # Because vectors are L2-normalized, cosine similarity equals the dot product
    sim_related = float(np.dot(vec_query, vec_related))
    sim_unrelated = float(np.dot(vec_query, vec_unrelated))

    assert sim_related > sim_unrelated
    assert sim_related > 0.6
