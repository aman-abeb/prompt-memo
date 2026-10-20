"""Unit tests for MinHash shingling, signature stability, and similarity estimation."""

import pytest
from prompt_memo.minhash import (
    MinHasher,
    compute_shingles,
    deserialize_signature,
    estimate_similarity,
    exact_jaccard,
    serialize_signature,
)


def test_compute_shingles_char_mode() -> None:
    text = "hello"
    shingles = compute_shingles(text, k=3, mode="char")
    assert shingles == {"hel", "ell", "llo"}


def test_compute_shingles_word_mode() -> None:
    text = "the quick brown fox jumps"
    shingles = compute_shingles(text, k=2, mode="word")
    assert shingles == {
        "the quick",
        "quick brown",
        "brown fox",
        "fox jumps",
    }


def test_compute_shingles_short_text() -> None:
    # Text shorter than k returns the entire collapsed string
    shingles_char = compute_shingles("hi", k=3, mode="char")
    assert shingles_char == {"hi"}

    shingles_word = compute_shingles("hello", k=3, mode="word")
    assert shingles_word == {"hello"}


def test_compute_shingles_empty_text() -> None:
    assert compute_shingles("", k=3) == set()
    assert compute_shingles("   \n\t  ", k=3) == set()


def test_compute_shingles_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        compute_shingles("text", k=0)

    with pytest.raises(ValueError, match="Unsupported shingle mode"):
        compute_shingles("text", k=3, mode="invalid_mode")


def test_exact_jaccard_similarity() -> None:
    set1 = {"a", "b", "c"}
    set2 = {"b", "c", "d"}
    # intersection: {'b', 'c'} (2), union: {'a', 'b', 'c', 'd'} (4) -> 0.5
    assert exact_jaccard(set1, set2) == 0.5

    # Identical sets
    assert exact_jaccard(set1, set1) == 1.0

    # Completely disjoint
    assert exact_jaccard({"x", "y"}, {"a", "b"}) == 0.0

    # Empty sets
    assert exact_jaccard(set(), set()) == 1.0
    assert exact_jaccard(set1, set()) == 0.0


def test_minhasher_signature_stability_and_determinism() -> None:
    hasher1 = MinHasher(num_perm=64, k=3, mode="char", seed=42)
    hasher2 = MinHasher(num_perm=64, k=3, mode="char", seed=42)
    hasher_diff_seed = MinHasher(num_perm=64, k=3, mode="char", seed=99)

    text = "Explain the theory of relativity in simple terms."
    sig1_a = hasher1.compute_signature(text)
    sig1_b = hasher1.compute_signature(text)
    sig2 = hasher2.compute_signature(text)
    sig_diff = hasher_diff_seed.compute_signature(text)

    assert len(sig1_a) == 64
    assert sig1_a == sig1_b  # Idempotent
    assert sig1_a == sig2    # Independent instances with same seed
    assert sig1_a != sig_diff  # Different seed produces different permutations


def test_minhasher_empty_signature() -> None:
    hasher = MinHasher(num_perm=64)
    empty_sig = hasher.compute_signature("")
    assert empty_sig == [0] * 64


def test_signature_serialization_roundtrip() -> None:
    hasher = MinHasher(num_perm=32, seed=42)
    sig = hasher.compute_signature("Test signature roundtrip serialization")

    # Comma-separated serialization
    serialized = serialize_signature(sig)
    deserialized = deserialize_signature(serialized)
    assert deserialized == sig

    # JSON-formatted string deserialization support
    import json
    json_serialized = json.dumps(sig)
    deserialized_json = deserialize_signature(json_serialized)
    assert deserialized_json == sig

    assert deserialize_signature("") == []


def test_similarity_estimation_near_duplicates() -> None:
    hasher = MinHasher(num_perm=64, k=3, mode="char", seed=42)

    text1 = "What is the capital of France?"
    text2 = "What is the capital of France!"
    text3 = "What is the capital of France, please?"
    unrelated = "How do I write a binary search algorithm in Python?"

    sim_identical = hasher.similarity_of_texts(text1, text1)
    sim_punctuation = hasher.similarity_of_texts(text1, text2)
    sim_near = hasher.similarity_of_texts(text1, text3)
    sim_unrelated = hasher.similarity_of_texts(text1, unrelated)

    assert sim_identical == 1.0
    assert sim_punctuation >= 0.90
    assert sim_near >= 0.70
    assert sim_unrelated <= 0.15


def test_minhash_vs_exact_jaccard_retrieval_accuracy() -> None:
    hasher = MinHasher(num_perm=64, k=3, mode="char", seed=42)

    pairs = [
        ("The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog!"),
        ("Explain machine learning in simple words.", "Explain machine learning in simple terms."),
        ("Docker container networking guide", "Kubernetes pod networking tutorial"),
        ("How to boil an egg", "Python list comprehension examples"),
    ]

    for text_a, text_b in pairs:
        shingles_a = hasher.compute_shingles(text_a)
        shingles_b = hasher.compute_shingles(text_b)
        exact = exact_jaccard(shingles_a, shingles_b)
        estimated = hasher.similarity_of_texts(text_a, text_b)

        # Standard error for 64 permutations is approx 1/sqrt(64) = 0.125
        assert abs(exact - estimated) < 0.20
